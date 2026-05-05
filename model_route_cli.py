#!/usr/bin/env python3
import argparse

from model_manager import model_manager


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 模型协调工具")
    parser.add_argument(
        "--task",
        default="clone",
        choices=["clone", "custom", "design"],
        help="任务类型",
    )
    parser.add_argument(
        "--objective",
        default="quality",
        choices=["quality", "balanced", "speed"],
        help="优化目标",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="如果没有可用模型则自动下载推荐模型",
    )
    parser.add_argument(
        "--check-updates",
        action="store_true",
        help="先检查一次更新",
    )
    args = parser.parse_args()

    if args.check_updates:
        update = model_manager.check_all_updates(force=True)
        print(f"[更新] {update.get('summary')}")

    result = model_manager.ensure_model_for_task(
        task=args.task,
        objective=args.objective,
        auto_download=args.auto_download,
    )

    print(f"[任务] {args.task} | [目标] {args.objective}")
    print(f"[结果] {result.message}")
    print("[候选链路]")
    if not result.route:
        print("  - 无可用模型")
        return

    for index, model in enumerate(result.route, start=1):
        status = "已下载" if model.downloaded else "未下载"
        source = model.source
        print(
            f"  {index}. {model.name} ({status}, source={source}, Q{model.quality_score}/S{model.speed_score})"
        )


if __name__ == "__main__":
    main()
