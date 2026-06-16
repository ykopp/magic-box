import argparse
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("请先安装 huggingface_hub: pip install huggingface_hub")
    sys.exit(1)

MODELS_DIR = "models"

MODELS = {
    "1": {
        "name": "Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "desc": "Voice Cloning (Pro bf16 - 1.7B, higher fidelity)"
    },
    "2": {
        "name": "Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "desc": "Voice Cloning (Lite bf16 - 0.6B, stable fallback)"
    },
    "3": {
        "name": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "desc": "Voice Cloning (Pro - 1.7B)"
    },
    "4": {
        "name": "Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "desc": "Voice Cloning (Lite - 0.6B)"
    },
}

REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
)


def validate_model_dir(local_dir):
    missing = [
        rel_path
        for rel_path in REQUIRED_FILES
        if not os.path.exists(os.path.join(local_dir, rel_path))
    ]
    if missing:
        return False, missing
    return True, []

def download_model(model_key):
    if model_key not in MODELS:
        print(f"无效选择: {model_key}")
        return False
    
    model_info = MODELS[model_key]
    local_dir = os.path.join(MODELS_DIR, model_info["name"])
    
    print(f"\n{'='*50}")
    print(f"下载模型: {model_info['desc']}")
    print(f"仓库: {model_info['repo']}")
    print(f"目标目录: {local_dir}")
    print(f"{'='*50}\n")
    
    try:
        snapshot_download(
            repo_id=model_info["repo"],
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        valid, missing = validate_model_dir(local_dir)
        if not valid:
            print(f"\n✗ 下载后模型仍不完整，缺失: {', '.join(missing)}")
            print("  缺少 speech_tokenizer/model.safetensors 会导致输出变成噪音。")
            return False
        print(f"\n✓ 下载完成: {local_dir}")
        return True
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False

def print_model_list():
    print("\n" + "="*50)
    print("Qwen3-TTS 模型下载工具")
    print("="*50)
    print("\n可用模型:")
    for key, info in MODELS.items():
        print(f"  {key}. {info['desc']}")
    print("\n推荐下载:")
    print("  - 1.7B Base bf16: 主力高保真 voice cloning")
    print("  - 0.6B Base bf16: 速度/内存优先的稳定备选")
    print("  - 8bit: 磁盘/内存紧张时使用")
    print("="*50)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 模型下载工具")
    parser.add_argument(
        "model",
        nargs="?",
        choices=sorted(MODELS.keys()),
        help="要下载的模型编号",
    )
    parser.add_argument("--list", action="store_true", help="列出可用模型后退出")
    return parser.parse_args()


def main():
    args = parse_args()
    print_model_list()

    if args.list:
        return 0

    choice = args.model
    if choice is None:
        if not sys.stdin.isatty():
            print("\n非交互环境请显式指定模型编号，例如: python3 download_model.py 3", file=sys.stderr)
            return 2
        choice = input("\n请选择要下载的模型 (输入数字): ").strip()

    return 0 if download_model(choice) else 1


if __name__ == "__main__":
    sys.exit(main())
