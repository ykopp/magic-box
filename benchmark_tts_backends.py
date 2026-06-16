import argparse
import json
import os
from pathlib import Path

import soundfile as sf

from tts_backends import (
    CHATTERBOX_MULTILINGUAL_MODEL,
    QWEN_DEFAULT_MODEL,
    VOXCPM_DEFAULT_MODEL,
    benchmark_generate_case,
)

BENCHMARK_CASES = [
    {
        "id": "cn_podcast",
        "task_mode": "clone",
        "text": "大家好，欢迎收听今天的播客节目。今天我们聊一个我最近反复听的主题：为什么真正打动人的声音，不只是清晰，还要有节奏、呼吸和情绪的起伏。",
        "notes": "中文单人播客段落",
    },
    {
        "id": "mixed_en",
        "task_mode": "clone",
        "text": "今天这期节目，我们会聊到 lo-fi, indie pop, and live session 这些关键词，也会顺带提到 Beatles 和 Radiohead 这两支我最近又重新回听的乐队。",
        "notes": "中英混排段落",
    },
    {
        "id": "energetic_intro",
        "task_mode": "clone",
        "text": "哈喽大家好，欢迎来到今天这一期节目。今天这首歌我真的想立刻分享给你，因为它不是那种安静地好听，而是一开口就会把人抓住。",
        "notes": "情绪更强的播客开场",
    },
]


def run_case(backend: str, model_ref: str, case: dict, ref_audio: str, ref_text: str, output_dir: Path):
    audio_result, metrics = benchmark_generate_case(
        backend=backend,
        model_ref=model_ref,
        task_mode=case["task_mode"],
        text=case["text"],
        ref_audio_path=ref_audio,
        ref_text=ref_text,
    )
    audio = audio_result.audio
    sample_rate = audio_result.sample_rate
    output_path = output_dir / f"{backend}__{case['id']}.wav"
    sf.write(output_path, audio, sample_rate)
    audio_seconds = len(audio) / sample_rate if sample_rate > 0 else 0
    metrics.update(
        {
            "case_id": case["id"],
            "notes": case["notes"],
            "output_path": str(output_path),
            "sample_rate": sample_rate,
            "audio_seconds": round(audio_seconds, 3),
            "realtime_factor": round(metrics["generate_seconds"] / audio_seconds, 3)
            if audio_seconds > 0
            else None,
            "subjective_similarity": "",
            "subjective_naturalness": "",
            "subjective_podcast_feel": "",
            "subjective_english_stability": "",
        }
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="统一对比 Qwen / Chatterbox / VoxCPM 的 benchmark 入口")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["qwen", "chatterbox", "voxcpm"],
        default=["qwen", "chatterbox", "voxcpm"],
        help="需要跑的后端列表",
    )
    parser.add_argument("--ref-audio", required=True, help="统一参考音频")
    parser.add_argument("--ref-text", required=True, help="统一参考音频文本")
    parser.add_argument("--qwen-model", default=QWEN_DEFAULT_MODEL, help="Qwen 模型路径")
    parser.add_argument("--chatterbox-model", default=CHATTERBOX_MULTILINGUAL_MODEL, help="Chatterbox 模型变体")
    parser.add_argument("--voxcpm-model", default=VOXCPM_DEFAULT_MODEL, help="VoxCPM 模型路径或 HF ID")
    parser.add_argument("--output-dir", default="outputs/benchmarks", help="benchmark 输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_map = {
        "qwen": args.qwen_model,
        "chatterbox": args.chatterbox_model,
        "voxcpm": args.voxcpm_model,
    }

    all_metrics = []
    for backend in args.backends:
        print(f"[backend] {backend} -> {model_map[backend]}")
        for case in BENCHMARK_CASES:
            print(f"  [case] {case['id']}")
            try:
                metrics = run_case(
                    backend=backend,
                    model_ref=model_map[backend],
                    case=case,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    output_dir=output_dir,
                )
                all_metrics.append(metrics)
                print(f"    saved: {metrics['output_path']}")
            except Exception as exc:
                all_metrics.append(
                    {
                        "backend": backend,
                        "model_ref": model_map[backend],
                        "case_id": case["id"],
                        "notes": case["notes"],
                        "error": str(exc),
                    }
                )
                print(f"    error: {exc}")

    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# TTS Benchmark Results",
        "",
        "| backend | case | audio_seconds | load_seconds | generate_seconds | realtime_factor | output | error |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in all_metrics:
        md_lines.append(
            "| {backend} | {case_id} | {audio_seconds} | {load_seconds} | {generate_seconds} | {realtime_factor} | {output} | {error} |".format(
                backend=item.get("backend", ""),
                case_id=item.get("case_id", ""),
                audio_seconds=item.get("audio_seconds", ""),
                load_seconds=item.get("load_seconds", ""),
                generate_seconds=item.get("generate_seconds", ""),
                realtime_factor=item.get("realtime_factor", ""),
                output=item.get("output_path", ""),
                error=item.get("error", ""),
            )
        )
    (output_dir / "benchmark_results.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[done] results -> {json_path}")


if __name__ == "__main__":
    main()
