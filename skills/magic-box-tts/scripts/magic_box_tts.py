#!/usr/bin/env python3
"""CLI wrapper for Magic Box TTS generation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_REPO = Path(os.environ.get("MAGIC_BOX_DIR", "/Users/liuchang/Apprun/chenxi/Magic Box"))
DEFAULT_MODEL = "/Users/liuchang/podcast_generator_models/Qwen3-TTS-12Hz-1.7B-Base-bf16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Magic Box TTS from text or a text file.")
    parser.add_argument("--repo", default=str(DEFAULT_REPO), help="Magic Box checkout path.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="Input text file.")
    source.add_argument("--text", help="Input text directly.")
    parser.add_argument("--output", required=True, help="Output path, relative to repo or absolute.")
    parser.add_argument("--profile", help="Saved profile id/name under voices/profiles.")
    parser.add_argument("--ref-audio", help="Reference audio path when not using --profile.")
    parser.add_argument("--ref-text", help="Reference transcript when not using --profile.")
    parser.add_argument("--backend", default="qwen", choices=["qwen", "voxcpm"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--speed", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--chunk-max-chars", type=int, default=260)
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "both"])
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    parser.add_argument("--no-normalise", action="store_true", help="Disable loudness normalization.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print an equivalent shell command for Hermes/remote execution instead of running.",
    )
    return parser.parse_args()


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path


def read_profile(repo: Path, profile: str) -> tuple[Path, str]:
    profiles_dir = repo / "voices" / "profiles"
    candidates = [profiles_dir / profile]
    if profiles_dir.exists():
        folded = profile.casefold()
        candidates.extend(path for path in profiles_dir.iterdir() if path.name.casefold() == folded)

    profile_dir = next((path for path in candidates if path.exists() and path.is_dir()), None)
    if profile_dir is None:
        raise FileNotFoundError(f"Profile not found under {profiles_dir}: {profile}")

    audio_path = profile_dir / "reference_clean.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Profile is missing reference_clean.wav: {profile_dir}")

    for text_name in ("reference_clean.txt", "transcript.txt"):
        text_path = profile_dir / text_name
        if text_path.exists():
            ref_text = text_path.read_text(encoding="utf-8").strip()
            if ref_text:
                return audio_path, ref_text

    metadata_path = profile_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        ref_text = str(metadata.get("ref_text") or metadata.get("transcript") or "").strip()
        if ref_text:
            return audio_path, ref_text

    raise FileNotFoundError(f"Profile is missing reference text: {profile_dir}")


def build_command(args: argparse.Namespace, text_path: Path, ref_audio: Path, ref_text: str) -> list[str]:
    repo = Path(args.repo).expanduser().resolve()
    python_path = repo / ".venv" / "bin" / "python"
    generator = repo / "podcast_generator.py"
    output_path = resolve_path(repo, args.output)
    checkpoint_path = output_path.with_suffix(".ckpt")

    command = [
        str(python_path),
        "-u",
        str(generator),
        "--file",
        str(text_path),
        "--ref-audio",
        str(ref_audio),
        "--ref-text",
        ref_text,
        "--output",
        str(output_path),
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--speed",
        str(args.speed),
        "--temperature",
        str(args.temperature),
        "--chunk-max-chars",
        str(args.chunk_max_chars),
        "--checkpoint",
        str(checkpoint_path),
        "--format",
        args.format,
        "--force-exit-after-run",
    ]
    if not args.no_resume:
        command.append("--resume")
    if args.no_normalise:
        command.append("--no-normalise")
    return command


def build_wrapper_command(args: argparse.Namespace) -> list[str]:
    repo = Path(args.repo).expanduser().resolve()
    wrapper = repo / "skills" / "magic-box-tts" / "scripts" / "magic_box_tts.py"
    command = ["python3", str(wrapper), "--repo", str(repo)]
    if args.file:
        command.extend(["--file", args.file])
    else:
        command.extend(["--text", args.text])
    command.extend(["--output", args.output])
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.ref_audio:
        command.extend(["--ref-audio", args.ref_audio])
    if args.ref_text:
        command.extend(["--ref-text", args.ref_text])
    command.extend(
        [
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--speed",
            str(args.speed),
            "--temperature",
            str(args.temperature),
            "--chunk-max-chars",
            str(args.chunk_max_chars),
            "--format",
            args.format,
        ]
    )
    if args.no_resume:
        command.append("--no-resume")
    if args.no_normalise:
        command.append("--no-normalise")
    return command


def shell_command(repo: Path, command: list[str]) -> str:
    return "cd " + shlex.quote(str(repo)) + " && " + " ".join(shlex.quote(part) for part in command)


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    if not repo.exists():
        print(f"Magic Box repo not found: {repo}", file=sys.stderr)
        return 2
    python_path = repo / ".venv" / "bin" / "python"
    if not python_path.exists():
        print(f"Project virtualenv not found: {python_path}", file=sys.stderr)
        return 2

    if args.print_command:
        print(shell_command(repo, build_wrapper_command(args)))
        return 0

    temp_file: tempfile.NamedTemporaryFile[str] | None = None
    if args.file:
        text_path = resolve_path(repo, args.file)
    else:
        runtime_dir = repo / "runtime" / "skill"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", dir=runtime_dir, delete=False)
        temp_file.write(args.text.strip())
        temp_file.close()
        text_path = Path(temp_file.name)

    if not text_path.exists():
        print(f"Input text file not found: {text_path}", file=sys.stderr)
        return 2

    if args.profile:
        ref_audio, ref_text = read_profile(repo, args.profile)
    else:
        if not args.ref_audio or not args.ref_text:
            print("Use --profile or provide both --ref-audio and --ref-text.", file=sys.stderr)
            return 2
        ref_audio = resolve_path(repo, args.ref_audio)
        ref_text = args.ref_text.strip()

    command = build_command(args, text_path, ref_audio, ref_text)

    print("Running Magic Box TTS:")
    print(" ".join(shlex.quote(part) for part in command))
    process = subprocess.Popen(command, cwd=str(repo))
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
