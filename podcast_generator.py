"""
podcast_generator.py — CLI voice-clone podcast generator.

Improvements over original:
- Uses shared utils.py (no duplicated helpers)
- Checkpoint / resume: interrupted generations can be continued
- Loudness normalisation at -16 LUFS with peak limiting
- Temp files cleaned up even on crash/Ctrl-C
- Better progress reporting with elapsed time
"""

import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from utils import (
    check_audio_file_health,
    check_audio_health,
    limit_audio_peak,
    load_checkpoint,
    make_output_filename,
    normalise_loudness,
    save_checkpoint,
    smooth_join_audio,
    split_text,
    _safe_remove,
    convert_wav_to_mp3,
    validate_generated_audio,
)
from tts_backends import (
    QWEN_DEFAULT_MODEL,
    VOXCPM_DEFAULT_MODEL,
    cleanup_reference_audio,
    generate_backend_chunk,
    load_backend_model,
    prepare_reference_audio,
)

MODEL_PATH = QWEN_DEFAULT_MODEL
MAX_CHUNK_ATTEMPTS = 3
DEFAULT_SPEED = 1.03
DEFAULT_TEMPERATURE = 0.85


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _generate_validated_chunk(
    *,
    backend: str,
    model,
    chunk: str,
    clean_audio: str,
    ref_text: str,
    speed: float,
    temperature: float,
    chunk_index: int,
) -> tuple[np.ndarray, int, list[str]]:
    """Generate one chunk, retrying unsafe waveform output with conservative settings."""

    warnings: list[str] = []
    last_error = ""

    for attempt in range(1, MAX_CHUNK_ATTEMPTS + 1):
        attempt_temperature = min(temperature, 0.85) if attempt > 1 else temperature
        attempt_speed = min(speed, 1.05) if attempt > 1 else speed

        audio_result = generate_backend_chunk(
            backend=backend,
            model=model,
            task_mode="clone",
            chunk=chunk,
            ref_audio_path=clean_audio,
            ref_text=ref_text,
            custom_speaker="Vivian",
            custom_instruction="Normal tone",
            design_instruction="",
            speed=attempt_speed,
            temperature=attempt_temperature,
        )

        audio_arr = np.asarray(audio_result.audio, dtype=np.float32)
        sample_rate = int(audio_result.sample_rate)
        warnings = validate_generated_audio(
            audio_arr,
            sample_rate,
            label=f"chunk {chunk_index}",
        )
        if not warnings:
            return audio_arr, sample_rate, warnings

        last_error = "; ".join(warnings)
        if attempt < MAX_CHUNK_ATTEMPTS:
            print(f"    ! 音频健康检查失败，重试 {attempt}/{MAX_CHUNK_ATTEMPTS - 1}: {last_error}")

    raise RuntimeError(f"chunk {chunk_index} generated unsafe audio after retries: {last_error}")


def _segment_dir_for_output(output_path: str) -> str:
    output = Path(output_path)
    return str(output.with_name(f".{output.stem}_segments"))


def generate_podcast(
    model,
    backend: str,
    ref_audio_path: str,
    ref_text: str,
    target_text: str,
    output_path: str,
    speed: float = DEFAULT_SPEED,
    temperature: float = DEFAULT_TEMPERATURE,
    chunk_max_chars: int = 80,
    checkpoint_path: str | None = None,
    checkpoint_metadata: dict | None = None,
    output_format: str = "mp3",
    normalise: bool = True,
) -> str | None:
    """Generate podcast audio for *target_text* by cloning the voice in *ref_audio_path*.

    Supports checkpoint / resume: pass *checkpoint_path* to save progress after
    each chunk.  If the file exists from a previous run the already-generated
    chunks are reused automatically.

    Returns the saved output path on success, or ``None`` on failure.
    """
    print(f"\n{'=' * 60}")
    print("播客音频生成")
    print(f"{'=' * 60}")
    print(f"文本长度: {len(target_text)} 字符")
    print(f"{'=' * 60}\n")

    # -- Reference audio --------------------------------------------------
    clean_audio = prepare_reference_audio("clone", ref_audio_path)
    if not clean_audio:
        print("✗ 参考音频转换失败（检查路径和 ffmpeg）")
        return None

    # -- Text chunking ----------------------------------------------------
    chunks = split_text(target_text, max_chars=chunk_max_chars)
    print(f"分成 {len(chunks)} 段生成\n")

    # -- Checkpoint: load existing progress --------------------------------
    completed_indices: list[int] = []
    segment_paths: list[str] = []

    checkpoint_metadata = checkpoint_metadata or {}
    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt and ckpt["chunks"] == chunks and ckpt.get("metadata", {}) == checkpoint_metadata:
            completed_indices = ckpt.get("completed", [])
            segment_paths = ckpt.get("segments", [])
            if completed_indices:
                print(f"⟳ 断点续生成: 已完成 {len(completed_indices)}/{len(chunks)} 段\n")

    # Pad segment_paths to length of chunks so indices align
    while len(segment_paths) < len(chunks):
        segment_paths.append("")

    segment_dir = _segment_dir_for_output(output_path)

    # -- Per-chunk generation ---------------------------------------------
    total_start = time.time()

    try:
        for i, chunk in enumerate(chunks):
            if i in completed_indices:
                print(f"[{i+1}/{len(chunks)}] (已有) {chunk[:50]}")
                continue

            label = chunk[:50] + ("…" if len(chunk) > 50 else "")
            print(f"[{i+1}/{len(chunks)}] {label}")
            seg_start = time.time()

            try:
                audio_arr, sample_rate, audio_warnings = _generate_validated_chunk(
                    backend=backend,
                    model=model,
                    chunk=chunk,
                    clean_audio=clean_audio,
                    ref_text=ref_text,
                    speed=speed,
                    temperature=temperature,
                    chunk_index=i + 1,
                )
                elapsed = time.time() - seg_start
                duration = len(audio_arr) / sample_rate
                print(f"    ✓ {elapsed:.0f}s → {duration:.1f}s 音频")
                for warning in audio_warnings:
                    print(f"    [audio warning] {warning}")

                # Save segment to temp file for checkpoint
                seg_path = os.path.join(
                    segment_dir,
                    f"_seg_{i:04d}.wav",
                )
                os.makedirs(segment_dir, exist_ok=True)
                sf.write(seg_path, audio_arr, sample_rate)
                segment_paths[i] = seg_path
                completed_indices.append(i)

                if checkpoint_path:
                    save_checkpoint(checkpoint_path, chunks, completed_indices, segment_paths, checkpoint_metadata)

            except KeyboardInterrupt:
                print("\n⎹ 中断！保存断点...")
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, chunks, completed_indices, segment_paths, checkpoint_metadata)
                raise
            except Exception as e:
                print(f"    ✗ 失败: {e}")
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, chunks, completed_indices, segment_paths, checkpoint_metadata)
                return None

    finally:
        cleanup_reference_audio(ref_audio_path, clean_audio)

    # -- Assemble final audio ---------------------------------------------
    all_audio: list[np.ndarray] = []
    final_sample_rate: int | None = None
    for i, seg_path in enumerate(segment_paths):
        if i not in completed_indices or not seg_path or not os.path.exists(seg_path):
            continue
        data, sample_rate = sf.read(seg_path)
        if final_sample_rate is None:
            final_sample_rate = int(sample_rate)
        elif final_sample_rate != int(sample_rate):
            print(f"\n✗ 片段采样率不一致: {final_sample_rate} vs {sample_rate}")
            return None
        all_audio.append(data)

    if not all_audio:
        print("\n✗ 没有成功生成任何片段")
        return None

    final_audio = smooth_join_audio(all_audio, final_sample_rate)
    final_sample_rate = final_sample_rate or 24000

    # -- Loudness normalisation -------------------------------------------
    if normalise:
        print("\n🔊 响度标准化 (-16 LUFS，峰值限制 0.95 peak)…")
        final_audio = normalise_loudness(final_audio, final_sample_rate)
    final_audio = limit_audio_peak(final_audio)

    # -- Audio health check ------------------------------------------------
    for warning in check_audio_health(final_audio, final_sample_rate, label="final audio before write"):
        print(f"[audio warning] {warning}")

    # -- Save output -------------------------------------------------------
    output_format = output_format if output_format in {"wav", "mp3", "both"} else "mp3"
    if output_format == "wav" and not output_path.lower().endswith(".wav"):
        output_path = output_path.rsplit(".", 1)[0] + ".wav"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wav_path = output_path
    if output_format in {"mp3", "both"} and not output_path.lower().endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"
    sf.write(wav_path, final_audio, final_sample_rate)

    for warning in check_audio_file_health(wav_path, label="written WAV"):
        print(f"[audio warning] {warning}")

    result_path = wav_path
    if output_format in {"mp3", "both"}:
        mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
        convert_wav_to_mp3(wav_path, mp3_path)
        print(f"✓ 已保存 MP3: {mp3_path}")
        for warning in check_audio_file_health(mp3_path, label="written MP3"):
            print(f"[audio warning] {warning}")
        if output_format == "mp3":
            _safe_remove(wav_path)
            result_path = mp3_path

    # Segment files are checkpoint artifacts during generation; remove them
    # only after a successful final assembly.
    for seg_path in segment_paths:
        if seg_path:
            _safe_remove(seg_path)
    if os.path.isdir(segment_dir):
        try:
            shutil.rmtree(segment_dir)
        except OSError:
            pass

    # Remove checkpoint on successful completion
    if checkpoint_path and os.path.exists(checkpoint_path):
        _safe_remove(checkpoint_path)

    total_elapsed = time.time() - total_start
    duration = len(final_audio) / final_sample_rate
    print(f"\n{'=' * 60}")
    print(f"✓ 已保存: {result_path}")
    print(f"✓ 总时长: {duration:.1f}s ({duration / 60:.1f} 分钟)")
    print(f"✓ 总耗时: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} 分钟)")
    print(f"{'=' * 60}")
    return result_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Magic Box（声音克隆）")
    parser.add_argument("--text", "-t", help="要生成的文本内容")
    parser.add_argument("--file", "-f", help="文本文件路径")
    parser.add_argument("--output", "-o", default="", help="输出文件路径（默认自动命名）")
    parser.add_argument(
        "--ref-audio", "-r",
        default="",
        help="参考音频路径（请使用你自己录制或上传的声音样本）",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="参考音频文本（参考音频里说的内容）",
    )
    parser.add_argument("--speed", "-s", type=float, default=DEFAULT_SPEED, help=f"语速 (默认 {DEFAULT_SPEED})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"随机性 (默认 {DEFAULT_TEMPERATURE})")
    parser.add_argument("--no-normalise", action="store_true", help="跳过响度标准化")
    parser.add_argument(
        "--format",
        choices=["wav", "mp3", "both"],
        default="mp3",
        help="输出格式: wav, mp3, both (默认 mp3)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从上次断点继续（需要保留 checkpoint 文件）",
    )
    parser.add_argument(
        "--checkpoint", default="",
        help="断点文件路径（默认: outputs/<output_name>.ckpt）",
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL_PATH,
        help="模型路径（默认 1.7B-Base）",
    )
    parser.add_argument(
        "--backend",
        choices=["qwen", "voxcpm", "chatterbox"],
        default="qwen",
        help="后端类型（默认 qwen；voxcpm/chatterbox 仅用于隔离实验）",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # -- Text input --------------------------------------------------------
    if args.file:
        if not os.path.exists(args.file):
            print(f"✗ 文件不存在: {args.file}")
            return
        with open(args.file, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    elif args.text:
        target_text = args.text
    else:
        parser.print_help()
        print("\n示例: python podcast_generator.py --file 稿件.txt --ref-audio 我的声音.m4a")
        return

    if not target_text:
        print("✗ 文本内容为空")
        return

    # -- Reference audio ---------------------------------------------------
    if not args.ref_audio:
        print("✗ 请提供参考音频: --ref-audio path/to/your_voice.wav")
        return
    if not args.ref_text.strip():
        print("✗ 请提供参考音频文本: --ref-text \"参考音频里实际说的内容\"")
        return
    if not os.path.exists(args.ref_audio):
        print(f"✗ 参考音频不存在: {args.ref_audio}")
        return

    # -- Output path -------------------------------------------------------
    output_path = args.output
    if not output_path:
        os.makedirs("outputs", exist_ok=True)
        snippet = target_text[:20]
        ext = "mp3" if args.format == "mp3" else "wav"
        output_path = os.path.join("outputs", make_output_filename(snippet, ext=ext))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # -- Checkpoint --------------------------------------------------------
    checkpoint_path = args.checkpoint or (output_path.rsplit(".", 1)[0] + ".ckpt")
    if not args.resume and os.path.exists(checkpoint_path):
        print(f"注意: 发现已有断点文件 {checkpoint_path}")
        ans = input("是否从断点继续？[y/N] ").strip().lower()
        if ans != "y":
            _safe_remove(checkpoint_path)
            checkpoint_path = None

    # -- Model -------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Magic Box")
    print(f"{'=' * 60}")

    if args.backend == "voxcpm" and args.model == MODEL_PATH:
        args.model = VOXCPM_DEFAULT_MODEL
    elif args.backend == "chatterbox" and args.model == MODEL_PATH:
        args.model = "chatterbox-multilingual-v3"

    print(f"\n后端: {args.backend}")
    print(f"加载模型: {args.model}")
    try:
        model, _ = load_backend_model(args.backend, args.model, task_mode="clone")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    print("✓ 模型加载成功")
    if args.backend == "voxcpm":
        print("ℹ VoxCPM 为实验后端：不支持 Qwen 的 speed/temperature 控制，按原生推理参数运行。")
    elif args.backend == "chatterbox":
        print("ℹ Chatterbox 为实验后端：建议在独立环境安装 chatterbox-tts 后测试。")

    # -- Generate ----------------------------------------------------------
    generate_podcast(
        model=model,
        backend=args.backend,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        target_text=target_text,
        output_path=output_path,
        speed=args.speed,
        temperature=args.temperature,
        checkpoint_path=checkpoint_path,
        output_format=args.format,
        normalise=not args.no_normalise,
    )


if __name__ == "__main__":
    main()
