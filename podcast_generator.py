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
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

from utils import (
    SAMPLE_RATE,
    audio_quality_issues,
    check_audio_file_health,
    check_audio_health,
    compute_audio_quality_metrics,
    crossfade_concat,
    limit_audio_peak,
    load_checkpoint,
    make_output_filename,
    normalise_loudness,
    remove_dc_offset,
    sanitise_tts_text,
    save_checkpoint,
    split_text,
    safe_remove,
    smooth_sample_jumps,
    convert_wav_to_mp3,
    trim_silence,
    write_quality_report,
)
from tts_backends import (
    QWEN_DEFAULT_MODEL,
    VOXCPM_DEFAULT_MODEL,
    cleanup_reference_audio,
    coerce_prepared_reference,
    generate_backend_chunk,
    load_backend_model,
    prepare_reference_audio,
)

MODEL_PATH = QWEN_DEFAULT_MODEL
QWEN_CLONE_BASELINE_CHARS_PER_SECOND = 6.3
MIN_RATE_CALIBRATION_REF_SECONDS = 3.0
MIN_RATE_CALIBRATION_REF_CHARS = 10
MIN_REASONABLE_REF_CHARS_PER_SECOND = 2.0
MAX_REASONABLE_REF_CHARS_PER_SECOND = 8.5


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _model_metadata(model) -> dict:
    config = getattr(model, "config", None)
    return {
        "class": type(model).__name__,
        "tts_model_type": _json_safe(getattr(config, "tts_model_type", None)),
        "sample_rate": _json_safe(getattr(model, "sample_rate", None)),
    }


def _file_fingerprint(path: str | os.PathLike[str] | None) -> dict:
    if not path:
        return {"path": None, "size": 0, "sha256": None}
    file_path = Path(path)
    payload = {"path": str(file_path), "size": 0, "sha256": None}
    try:
        stat = file_path.stat()
        payload["size"] = stat.st_size
        hasher = hashlib.sha256()
        with open(file_path, "rb") as fp:
            while chunk := fp.read(1 << 20):
                hasher.update(chunk)
        payload["sha256"] = hasher.hexdigest()
    except OSError:
        pass
    return payload


def _text_digest(text: str | None) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def _audio_duration_seconds(path: str | os.PathLike[str] | None) -> float:
    if not path:
        return 0.0
    try:
        info = sf.info(path)
    except Exception:
        return 0.0
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _reference_rate_calibration(
    *,
    backend: str,
    requested_speed: float,
    clean_audio_path: str,
    clean_ref_text: str,
) -> dict[str, object]:
    """Map the UI speed to the model's speed scale for cloned Qwen voices.

    Qwen's ``speed=1.0`` is the model baseline, not "match my reference".
    The UI treats 1.00 as reference-relative, so a slower reference needs a
    lower model speed before generation starts.
    """
    payload: dict[str, object] = {
        "enabled": False,
        "mode": "model_native",
        "requested_speed": float(requested_speed),
        "model_speed": float(requested_speed),
        "baseline_chars_per_second": QWEN_CLONE_BASELINE_CHARS_PER_SECOND,
    }
    if backend != "qwen":
        payload["reason"] = "backend does not expose compatible speed control"
        return payload

    duration = _audio_duration_seconds(clean_audio_path)
    metrics = _speech_rate_metrics(clean_ref_text, duration)
    text_chars = int(metrics["text_chars"])
    ref_cps = float(metrics["chars_per_second"])
    payload["reference"] = {
        "duration": duration,
        "text_chars": text_chars,
        "chars_per_second": ref_cps,
    }

    if duration < MIN_RATE_CALIBRATION_REF_SECONDS or text_chars < MIN_RATE_CALIBRATION_REF_CHARS:
        payload["reason"] = "reference is too short for reliable speed calibration"
        return payload
    if not np.isfinite(ref_cps) or not (
        MIN_REASONABLE_REF_CHARS_PER_SECOND <= ref_cps <= MAX_REASONABLE_REF_CHARS_PER_SECOND
    ):
        payload["reason"] = "reference speech rate is outside the safe calibration range"
        return payload

    model_speed = float(requested_speed) * (ref_cps / QWEN_CLONE_BASELINE_CHARS_PER_SECOND)
    model_speed = float(np.clip(model_speed, 0.5, 1.5))
    payload.update(
        {
            "enabled": True,
            "mode": "reference_relative",
            "model_speed": model_speed,
            "scale": model_speed / float(requested_speed) if requested_speed else 0.0,
        }
    )
    return payload


def _build_checkpoint_metadata(
    *,
    provided: dict | None,
    backend: str,
    model_ref: str | None,
    speed: float,
    model_speed: float,
    temperature: float,
    chunk_max_chars: int,
    normalise: bool,
    clean_audio_path: str,
    clean_ref_text: str,
    reference_report: dict,
    rate_calibration: dict | None = None,
) -> dict:
    provided = provided or {}
    profile_metadata = provided.get("profile") if isinstance(provided.get("profile"), dict) else None
    if profile_metadata is None and any(str(key).startswith("voice_profile_") for key in provided):
        profile_metadata = {
            key: value
            for key, value in provided.items()
            if str(key).startswith("voice_profile_")
        }

    reference_quality = reference_report.get("audit")
    if not isinstance(reference_quality, dict):
        reference_quality = {}

    return {
        "version": 5,
        "backend": backend,
        "model_ref": model_ref,
        "speed": float(speed),
        "model_speed": float(model_speed),
        "rate_calibration": _json_safe(rate_calibration or {}),
        "temperature": float(temperature),
        "chunk_max_chars": int(chunk_max_chars),
        "normalise": bool(normalise),
        "chunk_declick": {
            "enabled": True,
            "threshold": 0.25,
            "radius": 2,
            "passes": 5,
        },
        "post_loudness_declick": {
            "enabled": True,
            "threshold": 0.5,
            "radius": 2,
            "passes": 5,
        },
        "reference": {
            "audio": _file_fingerprint(clean_audio_path),
            "ref_text_sha256": _text_digest(clean_ref_text),
            "quality_ok": reference_quality.get("ok"),
            "quality_warnings": reference_quality.get("warnings", []),
        },
        "profile": _json_safe(profile_metadata or {}),
    }


def _audio_file_metrics(path: str, *, label: str) -> tuple[dict | None, list[str], list[str]]:
    try:
        audio, sample_rate = sf.read(path)
    except Exception as exc:
        message = f"{label}: could not read audio file ({exc})"
        return None, [message], [message]
    metrics = compute_audio_quality_metrics(audio, int(sample_rate))
    return metrics, audio_quality_issues(metrics, label=label), check_audio_health(audio, int(sample_rate), label=label)


def _split_transient_jump_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    """Treat post-gain sample-jump findings as diagnostics, not hard failures."""
    hard: list[str] = []
    transient: list[str] = []
    for issue in issues:
        if "sample jumps" in issue:
            transient.append(issue)
        else:
            hard.append(issue)
    return hard, transient


def _split_pre_trim_chunk_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    """Keep raw chunk pause findings as diagnostics; final audio is gated post-trim."""
    hard: list[str] = []
    diagnostics: list[str] = []
    for issue in issues:
        if "audible pause/dropout" in issue:
            diagnostics.append(issue)
        else:
            hard.append(issue)
    return hard, diagnostics


def _speech_rate_metrics(text: str, duration: float) -> dict[str, float | int]:
    text_chars = len("".join(str(text or "").split()))
    chars_per_second = float(text_chars / duration) if duration > 0 else float("inf")
    return {
        "text_chars": text_chars,
        "chars_per_second": chars_per_second,
    }


def _speech_rate_issues(
    text: str,
    duration: float,
    *,
    label: str,
    min_chars: int = 30,
    max_chars_per_second: float = 14.0,
) -> list[str]:
    """Catch truncated/early-EOS TTS output that is too short for the text."""
    metrics = _speech_rate_metrics(text, duration)
    text_chars = int(metrics["text_chars"])
    chars_per_second = float(metrics["chars_per_second"])
    if text_chars >= min_chars and chars_per_second > max_chars_per_second:
        return [
            f"{label}: audio is too short for {text_chars} text chars "
            f"({chars_per_second:.1f} chars/s > {max_chars_per_second:.1f}); generation likely truncated"
        ]
    return []


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_podcast(
    model,
    backend: str,
    ref_audio_path: str,
    ref_text: str,
    target_text: str,
    output_path: str,
    speed: float = 1.0,
    temperature: float = 1.0,
    chunk_max_chars: int = 80,
    checkpoint_path: str | None = None,
    checkpoint_metadata: dict | None = None,
    output_format: str = "mp3",
    normalise: bool = True,
    keep_segments: bool = True,
    quality_gate: bool = True,
    quality_report_path: str | None = None,
    model_ref: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> str | None:
    """Generate podcast audio for *target_text* by cloning the voice in *ref_audio_path*.

    Supports checkpoint / resume: pass *checkpoint_path* to save progress after
    each chunk.  If the file exists from a previous run the already-generated
    chunks are reused automatically.

    *progress_callback*, if given, is called as ``(current, total, label)`` after
    each chunk completes.

    Returns the saved output path on success, or ``None`` on failure.
    """
    target_text = sanitise_tts_text(target_text).strip()
    ref_text = sanitise_tts_text(ref_text).strip()

    print(f"\n{'=' * 60}")
    print("播客音频生成")
    print(f"{'=' * 60}")
    print(f"文本长度: {len(target_text)} 字符")
    print(f"{'=' * 60}\n")

    output_dir = os.path.dirname(output_path) or "."
    output_stem = Path(output_path).stem
    segment_dir = os.path.join(output_dir, f".{output_stem}_segments")
    report_path = quality_report_path or os.path.join(output_dir, "quality_report.json")
    quality_report: dict = {
        "status": "running",
        "failed": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_path": output_path,
        "segment_dir": segment_dir,
        "backend": backend,
        "generation": {
            "model_ref": model_ref,
            "model_metadata": _model_metadata(model),
            "checkpoint_metadata": _json_safe(checkpoint_metadata or {}),
            "profile_metadata": _json_safe((checkpoint_metadata or {}).get("profile") if checkpoint_metadata else None),
            "speed": speed,
            "temperature": temperature,
            "chunk_max_chars": chunk_max_chars,
            "normalise": normalise,
            "loudness": {
                "target_lufs": -23.0,
                "peak_ceiling": 0.95,
                "limiter_ceiling": 0.95,
            },
            "quality_gate": quality_gate,
            "output_format": output_format,
            "requested_speed": speed,
            "model_speed": speed,
            "rate_calibration": None,
        },
        "reference": None,
        "sample_rate": None,
        "chunks": [],
        "final": None,
        "issues": [],
    }

    def finish_report(status: str, issues: list[str], result_path: str | None = None) -> None:
        quality_report["status"] = status
        quality_report["failed"] = status == "failed"
        quality_report["issues"] = issues
        if result_path is not None:
            quality_report["result_path"] = result_path
        write_quality_report(report_path, quality_report)

    # -- Reference audio --------------------------------------------------
    prepared_ref = coerce_prepared_reference(
        prepare_reference_audio("clone", ref_audio_path, ref_text),
        fallback_ref_text=ref_text,
    )
    if not prepared_ref:
        print("✗ 参考音频转换失败（检查路径和 ffmpeg）")
        return None
    clean_audio = prepared_ref.clean_audio_path
    clean_ref_text = (prepared_ref.clean_ref_text or "").strip()
    quality_report["reference"] = prepared_ref.to_report_dict()
    if not clean_ref_text:
        finish_report("failed", ["reference: missing audited reference transcript"])
        print(f"✗ 质量报告: {report_path}")
        cleanup_reference_audio(ref_audio_path, prepared_ref)
        return None

    rate_calibration = _reference_rate_calibration(
        backend=backend,
        requested_speed=speed,
        clean_audio_path=clean_audio,
        clean_ref_text=clean_ref_text,
    )
    model_speed = float(rate_calibration.get("model_speed", speed))
    quality_report["generation"]["model_speed"] = model_speed
    quality_report["generation"]["rate_calibration"] = _json_safe(rate_calibration)
    if bool(rate_calibration.get("enabled")):
        ref_info = rate_calibration.get("reference")
        ref_cps = ref_info.get("chars_per_second") if isinstance(ref_info, dict) else None
        print(
            "↔ 参考语速校准: UI "
            f"{float(speed):.2f} → 模型 speed {model_speed:.2f}"
            + (f"（参考约 {float(ref_cps):.2f} 字/秒）" if isinstance(ref_cps, (int, float)) else "")
        )
    else:
        reason = rate_calibration.get("reason")
        if reason:
            print(f"↔ 参考语速校准未启用: {reason}")

    checkpoint_metadata = _build_checkpoint_metadata(
        provided=checkpoint_metadata,
        backend=backend,
        model_ref=model_ref,
        speed=speed,
        model_speed=model_speed,
        temperature=temperature,
        chunk_max_chars=chunk_max_chars,
        normalise=normalise,
        clean_audio_path=clean_audio,
        clean_ref_text=clean_ref_text,
        reference_report=quality_report["reference"],
        rate_calibration=rate_calibration,
    )
    quality_report["generation"]["checkpoint_metadata"] = _json_safe(checkpoint_metadata)
    quality_report["generation"]["profile_metadata"] = _json_safe(checkpoint_metadata.get("profile"))

    # -- Text chunking ----------------------------------------------------
    chunks = split_text(target_text, max_chars=chunk_max_chars)
    print(f"分成 {len(chunks)} 段生成\n")

    # -- Checkpoint: load existing progress --------------------------------
    completed_indices: list[int] = []
    segment_paths: list[str] = []

    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt and ckpt["chunks"] == chunks and ckpt.get("metadata", {}) == checkpoint_metadata:
            completed_indices = ckpt.get("completed", [])
            segment_paths = ckpt.get("segments", [])
            if completed_indices:
                print(f"⟳ 断点续生成: 已完成 {len(completed_indices)}/{len(chunks)} 段\n")
        elif ckpt:
            print("↻ 断点参数或参考音频已变化，将重新生成全部片段。")

    # Pad segment_paths to length of chunks so indices align
    while len(segment_paths) < len(chunks):
        segment_paths.append("")

    os.makedirs(segment_dir, exist_ok=True)

    # -- Per-chunk generation ---------------------------------------------
    # Track the sample rate of the *first successfully produced* chunk so we
    # can catch a backend that silently switches rate mid-generation (e.g.
    # a future VoxCPM2 variant that returns a different rate per chunk).
    # Mirrors the check already enforced in benchmark_generate_case.
    expected_sample_rate: int | None = None
    total_start = time.time()
    generation_errors: list[str] = []
    chunk_declick_stats_by_index: dict[int, dict[str, int | float]] = {}

    try:
        for i, chunk in enumerate(chunks):
            if i in completed_indices:
                print(f"[{i+1}/{len(chunks)}] (已有) {chunk[:50]}")
                continue

            label = chunk[:50] + ("…" if len(chunk) > 50 else "")
            print(f"[{i+1}/{len(chunks)}] {label}")
            seg_start = time.time()

            try:
                audio_result = generate_backend_chunk(
                    backend=backend,
                    model=model,
                    task_mode="clone",
                    chunk=chunk,
                    ref_audio_path=clean_audio,
                    ref_text=clean_ref_text,
                    custom_speaker="Vivian",
                    custom_instruction="Normal tone",
                    design_instruction="",
                    speed=model_speed,
                    temperature=temperature,
                )
                audio_arr = np.asarray(audio_result.audio, dtype=np.float32)
                sample_rate = audio_result.sample_rate
                if expected_sample_rate is None:
                    expected_sample_rate = int(sample_rate)
                elif int(sample_rate) != expected_sample_rate:
                    raise ValueError(
                        f"后端在第 {i+1} 段切换了采样率: 已生成段为 "
                        f"{expected_sample_rate} Hz, 本段为 {int(sample_rate)} Hz。 "
                        "若继续拼接, 输出将以错误速率写入 WAV, 表现为播放加速/减速。"
                    )
                elapsed = time.time() - seg_start
                audio_arr, chunk_declick_stats = smooth_sample_jumps(audio_arr, threshold=0.25, radius=2)
                if int(chunk_declick_stats.get("repaired_jumps", 0) or 0) > 0:
                    print(
                        "    🧹 De-click: repaired "
                        f"{chunk_declick_stats['repaired_jumps']}/{chunk_declick_stats['detected_jumps']} sample jumps"
                    )
                chunk_declick_stats_by_index[i] = chunk_declick_stats
                duration = len(audio_arr) / sample_rate
                print(f"    ✓ {elapsed:.0f}s → {duration:.1f}s 音频")

                # Per-chunk audio health check. Catches NaN/Inf, all-silence,
                # peak clipping, and very-low RMS *as soon as the chunk is
                # produced* — the final-audio check at the end of the pipeline
                # would otherwise let one bad chunk silently smear into the
                # final mix (where it's much harder to attribute to a segment).
                for warning in check_audio_health(
                    audio_arr, int(sample_rate), label=f"chunk {i+1}/{len(chunks)}"
                ):
                    print(f"    [audio warning] {warning}")

                # Save segment evidence for checkpointing and quality reports.
                seg_path = os.path.join(segment_dir, f"_seg_{i:04d}.wav")
                sf.write(seg_path, audio_arr, sample_rate)
                segment_paths[i] = seg_path
                completed_indices.append(i)

                if progress_callback is not None:
                    progress_callback(i + 1, len(chunks), label)

                if checkpoint_path:
                    if not save_checkpoint(checkpoint_path, chunks, completed_indices, segment_paths, checkpoint_metadata):
                        print("    ! 断点保存失败（磁盘满或权限不足），下次启动无法继续生成。")

            except KeyboardInterrupt:
                print("\n⎹ 中断！保存断点...")
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, chunks, completed_indices, segment_paths, checkpoint_metadata)
                raise
            except Exception as e:
                print(f"    ✗ 失败: {e}")
                generation_errors.append(f"chunk {i+1}/{len(chunks)}: {e}")

    finally:
        cleanup_reference_audio(ref_audio_path, prepared_ref)

    # -- Assemble final audio ---------------------------------------------
    all_audio: list[np.ndarray] = []
    final_sample_rate: int | None = None
    raw_total_ms = 0
    for i, seg_path in enumerate(segment_paths):
        if i not in completed_indices or not seg_path or not os.path.exists(seg_path):
            continue
        data, sample_rate = sf.read(seg_path)
        if final_sample_rate is None:
            final_sample_rate = int(sample_rate)
        elif final_sample_rate != int(sample_rate):
            print(f"\n✗ 片段采样率不一致: {final_sample_rate} vs {sample_rate}")
            finish_report(
                "failed",
                [f"segment sample-rate mismatch: {final_sample_rate} vs {int(sample_rate)}"],
            )
            print(f"✗ 质量报告: {report_path}")
            return None
        all_audio.append(np.asarray(data, dtype=np.float32))
        raw_total_ms += int(len(data) / sample_rate * 1000)
        metrics = compute_audio_quality_metrics(data, int(sample_rate))
        metrics.update(_speech_rate_metrics(chunks[i], float(metrics.get("duration", 0.0) or 0.0)))
        chunk_issues = audio_quality_issues(metrics, label=f"chunk {i+1}/{len(chunks)}")
        chunk_issues, pre_trim_diagnostics = _split_pre_trim_chunk_issues(chunk_issues)
        chunk_issues.extend(
            _speech_rate_issues(
                chunks[i],
                float(metrics.get("duration", 0.0) or 0.0),
                label=f"chunk {i+1}/{len(chunks)}",
            )
        )
        quality_report["chunks"].append(
            {
                "index": i,
                "label": f"chunk {i+1}/{len(chunks)}",
                "text": chunks[i],
                "path": seg_path,
                "sample_rate": int(sample_rate),
                "metrics": metrics,
                "issues": chunk_issues,
                "pre_trim_diagnostics": pre_trim_diagnostics,
                "declick": chunk_declick_stats_by_index.get(i),
            }
        )

    if expected_sample_rate is not None:
        quality_report["sample_rate"] = expected_sample_rate
    elif final_sample_rate is not None:
        quality_report["sample_rate"] = final_sample_rate

    if not all_audio:
        print("\n✗ 没有成功生成任何片段")
        finish_report("failed", generation_errors + ["final: no generated segments"])
        print(f"✗ 质量报告: {report_path}")
        return None

    if generation_errors or len(completed_indices) != len(chunks):
        missing = sorted(set(range(len(chunks))) - set(completed_indices))
        issues = generation_errors + [f"missing chunks: {missing}"]
        finish_report("failed", issues)
        print(f"\n✗ 生成未完成，已保留片段证据: {segment_dir}")
        print(f"✗ 质量报告: {report_path}")
        return None

    # -- Trim long silence padding from each chunk ------------------------
    # Qwen3-TTS (and VoxCPM2) can emit silence padding, but real cloned speech
    # often has a very low RMS floor. Keep this conservative: only true near-
    # silence is shortened, and trim_silence refuses aggressive chunk removal.
    trimmed_audio: list[np.ndarray] = []
    trimmed_ms = 0
    for seg in all_audio:
        before_ms = int(len(seg) / (final_sample_rate or SAMPLE_RATE) * 1000)
        t = trim_silence(
            seg,
            final_sample_rate or SAMPLE_RATE,
            threshold=0.006,
            min_silence_ms=350.0,
            keep_silence_ms=450.0,
            max_trim_ratio=0.70,
            head_trim_ms=80.0,
            tail_trim_ms=80.0,
            pad_ms=30.0,
        )
        after_ms = int(len(t) / (final_sample_rate or SAMPLE_RATE) * 1000)
        trimmed_audio.append(t)
        trimmed_ms += after_ms
    saved_ms = raw_total_ms - trimmed_ms
    if saved_ms > 0:
        print(
            f"\n✂  去掉了 {saved_ms / 1000:.1f}s silence padding "
            f"({raw_total_ms / 1000:.1f}s → {trimmed_ms / 1000:.1f}s)"
        )

    # Crossfade-concat (defensive — chunks already end/start near zero after
    # trim_silence pads both ends with 30 ms of silence, but a 30 ms linear
    # crossfade still removes any sub-millisecond step that would otherwise
    # become an audible click after loudness amplification).
    final_audio = crossfade_concat(trimmed_audio, fade_ms=30, sample_rate=final_sample_rate or SAMPLE_RATE)
    final_sample_rate = final_sample_rate or 24000

    # De-DC (typically <1e-3 but cumulative across many chunks).
    final_audio = remove_dc_offset(final_audio)
    final_audio, declick_stats = smooth_sample_jumps(final_audio, threshold=0.25, radius=2)
    if int(declick_stats.get("repaired_jumps", 0) or 0) > 0:
        print(
            "\n🧹 De-click: repaired "
            f"{declick_stats['repaired_jumps']}/{declick_stats['detected_jumps']} sample jumps"
        )

    # -- Audio quality gate ------------------------------------------------
    # Gate on the post-trim/post-crossfade signal *before* loudness gain.
    # Loudness normalisation intentionally amplifies the waveform; running the
    # sample-jump gate after that gain turns acceptable model output into false
    # failures because every slope is multiplied along with the speech.
    final_metrics = compute_audio_quality_metrics(final_audio, final_sample_rate)
    final_metrics.update(_speech_rate_metrics(target_text, float(final_metrics.get("duration", 0.0) or 0.0)))
    final_issues = audio_quality_issues(final_metrics, label="final audio before loudness")
    final_issues.extend(
        _speech_rate_issues(
            target_text,
            float(final_metrics.get("duration", 0.0) or 0.0),
            label="final audio before loudness",
        )
    )
    quality_report["final"] = {
        "sample_rate": final_sample_rate,
        "metrics": final_metrics,
        "issues": final_issues,
        "declick": declick_stats,
    }
    quality_issues: list[str] = []
    for chunk_report in quality_report["chunks"]:
        quality_issues.extend(chunk_report["issues"])
    quality_issues.extend(final_issues)

    for warning in check_audio_health(final_audio, final_sample_rate, label="final audio before write"):
        print(f"[audio warning] {warning}")

    if quality_gate and quality_issues:
        finish_report("failed", quality_issues)
        print(f"\n✗ 音频质量检测失败，已保留片段证据: {segment_dir}")
        print(f"✗ 质量报告: {report_path}")
        return None

    # -- Loudness normalisation -------------------------------------------
    if normalise:
        print("\n🔊 响度标准化 (-23 LUFS，软限幅 0.95 peak)…")
        final_audio = normalise_loudness(final_audio, final_sample_rate)
    final_audio = limit_audio_peak(final_audio)
    final_audio, post_loudness_declick = smooth_sample_jumps(final_audio, threshold=0.5, radius=2)
    if int(post_loudness_declick.get("repaired_jumps", 0) or 0) > 0:
        print(
            "🧹 Post-loudness de-click: repaired "
            f"{post_loudness_declick['repaired_jumps']}/{post_loudness_declick['detected_jumps']} sample jumps"
        )

    output_metrics = compute_audio_quality_metrics(final_audio, final_sample_rate)
    output_warnings = check_audio_health(final_audio, final_sample_rate, label="final audio after loudness")
    output_issues = audio_quality_issues(output_metrics, label="final audio after loudness")
    quality_report["final"]["output_metrics"] = output_metrics
    quality_report["final"]["output_warnings"] = output_warnings
    quality_report["final"]["output_issues"] = output_issues
    quality_report["final"]["output_diagnostics"] = []
    quality_report["final"]["post_loudness_declick"] = post_loudness_declick

    # -- Save output -------------------------------------------------------
    output_format = output_format if output_format in {"wav", "mp3", "both"} else "mp3"
    if output_format == "wav" and not output_path.lower().endswith(".wav"):
        output_path = output_path.rsplit(".", 1)[0] + ".wav"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wav_path = output_path
    if output_format in {"mp3", "both"} and not output_path.lower().endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"
    sf.write(wav_path, final_audio, final_sample_rate)

    for warning in output_warnings:
        print(f"[audio warning] {warning}")
    for warning in check_audio_file_health(wav_path, label="written WAV"):
        print(f"[audio warning] {warning}")
    wav_metrics, wav_issues, wav_warnings = _audio_file_metrics(wav_path, label="written WAV")
    quality_report["final"]["written_wav_metrics"] = wav_metrics
    quality_report["final"]["written_wav_issues"] = wav_issues
    quality_report["final"]["written_wav_warnings"] = wav_warnings
    quality_report["final"]["written_wav_diagnostics"] = []
    final_delivery_issues = output_issues + wav_issues
    needs_mp3_decode = output_format in {"mp3", "both"}
    if quality_gate and final_delivery_issues and not needs_mp3_decode:
        quality_issues.extend(final_delivery_issues)
        quality_report["final"]["path"] = wav_path
        finish_report("failed", quality_issues, result_path=wav_path)
        print(f"\n✗ 最终 WAV 音频质量检测失败，已保留片段证据: {segment_dir}")
        print(f"✗ 质量报告: {report_path}")
        return None

    result_path = wav_path
    if output_format in {"mp3", "both"}:
        mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
        convert_wav_to_mp3(wav_path, mp3_path)
        print(f"✓ 已保存 MP3: {mp3_path}")
        for warning in check_audio_file_health(mp3_path, label="written MP3"):
            print(f"[audio warning] {warning}")
        mp3_metrics, mp3_issues, mp3_warnings = _audio_file_metrics(mp3_path, label="decoded MP3")
        quality_report["final"]["decoded_mp3_metrics"] = mp3_metrics
        quality_report["final"]["decoded_mp3_issues"] = mp3_issues
        quality_report["final"]["decoded_mp3_warnings"] = mp3_warnings
        quality_report["final"]["decoded_mp3_diagnostics"] = []
        final_delivery_issues.extend(mp3_issues)
        if quality_gate and final_delivery_issues:
            quality_issues.extend(final_delivery_issues)
            quality_report["final"]["path"] = wav_path
            quality_report["final"]["mp3_path"] = mp3_path
            finish_report("failed", quality_issues, result_path=mp3_path)
            print(f"\n✗ 最终交付音频质量检测失败，已保留片段证据: {segment_dir}")
            print(f"✗ 质量报告: {report_path}")
            return None
        if output_format == "mp3":
            safe_remove(wav_path)
            result_path = mp3_path

    quality_report["final"]["path"] = wav_path
    finish_report("passed", quality_issues, result_path=result_path)
    print(f"✓ 质量报告: {report_path}")

    # Segment files are checkpoint artifacts during generation; remove them
    # only after a successful final assembly.
    if not keep_segments:
        for seg_path in segment_paths:
            if seg_path:
                safe_remove(seg_path)
        try:
            os.rmdir(segment_dir)
        except OSError:
            pass

    # Remove checkpoint on successful completion
    if checkpoint_path and os.path.exists(checkpoint_path):
        safe_remove(checkpoint_path)

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
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="语速倍率 (默认 1.00；Qwen clone 会按参考音频语速校准)")
    parser.add_argument("--temperature", type=float, default=1.0, help="随机性 (默认 1.0)")
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
        "--checkpoint-metadata-file",
        default="",
        help="包含断点元数据的 JSON 文件（供 Web UI 子进程调用）。",
    )
    parser.add_argument("--chunk-max-chars", type=int, default=80, help="每段最大字符数 (默认 80)")
    parser.add_argument(
        "--model", "-m",
        default=MODEL_PATH,
        help="模型路径（默认 1.7B-Base）",
    )
    parser.add_argument(
        "--backend",
        choices=["qwen", "voxcpm"],
        default="qwen",
        help="后端类型（默认 qwen；voxcpm 仅用于隔离实验）",
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
            safe_remove(checkpoint_path)
            checkpoint_path = None

    checkpoint_metadata = None
    if args.checkpoint_metadata_file:
        try:
            with open(args.checkpoint_metadata_file, "r", encoding="utf-8") as f:
                checkpoint_metadata = json.load(f)
        except Exception as exc:
            print(f"✗ 断点元数据读取失败: {exc}")
            sys.exit(1)

    # -- Model -------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Magic Box")
    print(f"{'=' * 60}")

    if args.backend == "voxcpm" and args.model == MODEL_PATH:
        args.model = VOXCPM_DEFAULT_MODEL

    print(f"\n后端: {args.backend}")
    print(f"加载模型: {args.model}")
    try:
        model, _ = load_backend_model(args.backend, args.model)
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    print("✓ 模型加载成功")
    if args.backend == "voxcpm":
        print("ℹ VoxCPM 为实验后端：不支持 Qwen 的 speed/temperature 控制，按原生推理参数运行。")

    # -- Generate ----------------------------------------------------------
    result = generate_podcast(
        model=model,
        backend=args.backend,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        target_text=target_text,
        output_path=output_path,
        speed=args.speed,
        temperature=args.temperature,
        chunk_max_chars=args.chunk_max_chars,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        output_format=args.format,
        normalise=not args.no_normalise,
        model_ref=args.model,
    )
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logging.exception("生成失败: %s", exc)
        sys.exit(1)
