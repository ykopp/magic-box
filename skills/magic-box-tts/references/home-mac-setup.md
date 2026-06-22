# Home Mac Setup For Magic Box TTS

Use this when preparing another Mac to run Magic Box generation locally.

## Required Local State

- macOS on Apple Silicon.
- Git checkout of `https://github.com/ykopp/magic-box.git`, branch `codex/interactive-web-app`.
- Project virtualenv at `.venv`.
- `ffmpeg` available for MP3 output.
- Qwen model directory available locally, commonly:

```text
/Users/liuchang/podcast_generator_models/Qwen3-TTS-12Hz-1.7B-Base-bf16
```

- A saved voice profile copied to:

```text
voices/profiles/<profile_id>/
  reference_clean.wav
  reference_clean.txt or transcript.txt
  metadata.json
```

## First-Time Setup Sketch

```bash
mkdir -p "/Users/liuchang/Apprun/chenxi"
cd "/Users/liuchang/Apprun/chenxi"
git clone https://github.com/ykopp/magic-box.git "Magic Box"
cd "Magic Box"
git checkout codex/interactive-web-app
python3.11 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e .
brew install ffmpeg
```

Model download and profile transfer are intentionally not automated here because model files and personal voice data are large/private.

## Hermes Pattern

When Hermes is available on the target Mac, generate or send a shell command that:

1. `cd`s into the Magic Box checkout.
2. Runs `python3 skills/magic-box-tts/scripts/magic_box_tts.py ...`.
3. Writes logs to a file under `outputs/` or Hermes output storage.
4. Returns the final output path and `quality_report.json` status.

Prefer this over running long generation on the current laptop.
