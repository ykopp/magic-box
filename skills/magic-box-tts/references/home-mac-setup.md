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

Recommended one-command setup from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/ykopp/magic-box/codex/interactive-web-app/scripts/bootstrap_mac.sh | bash
```

Equivalent manual setup:

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

## Sync Updates From GitHub

After a Magic Box update is pushed, run this on the target Mac:

```bash
cd "/Users/liuchang/Apprun/chenxi/Magic Box"
make sync-update
```

or:

```bash
bash scripts/sync_update.sh
```

The sync script pulls the configured branch with `--ff-only`, refreshes `.venv`, and relinks the bundled skill into `~/.codex/skills/` when that folder exists. It leaves `models/`, `voices/profiles/`, `outputs/`, and `runtime/` local.

## Web On A Mac

To run the Streamlit production UI on a Mac:

```bash
cd "/Users/liuchang/Apprun/chenxi/Magic Box"
make run-streamlit
```

The default URL is `http://127.0.0.1:8507/`.

## Hermes Pattern

When Hermes is available on the target Mac, generate or send a shell command that:

1. `cd`s into the Magic Box checkout.
2. Runs `python3 skills/magic-box-tts/scripts/magic_box_tts.py ...`.
3. Writes logs to a file under `outputs/` or Hermes output storage.
4. Returns the final output path and `quality_report.json` status.

Prefer this over running long generation on the current laptop.

Example command to send through Hermes:

```bash
cd "/Users/liuchang/Apprun/chenxi/Magic Box" && \
python3 skills/magic-box-tts/scripts/magic_box_tts.py \
  --text "要朗读的正文" \
  --profile 刘畅 \
  --output outputs/story.mp3
```

To generate an escaped command without running locally:

```bash
python3 skills/magic-box-tts/scripts/magic_box_tts.py \
  --text "要朗读的正文" \
  --profile 刘畅 \
  --output outputs/story.mp3 \
  --print-command
```
