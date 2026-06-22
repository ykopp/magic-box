---
name: magic-box-tts
description: Generate narration or story audio with the local Magic Box TTS project from text files, pasted text, saved voice profiles, or remote/Hermes task handoff. Use when the user wants Magic Box CLI audio generation, batch TTS, running generation on another Mac, checking Magic Box outputs, preparing Hermes-executable commands, or avoiding keeping the current computer busy during long Qwen/VoxCPM generation.
---

# Magic Box TTS

Use this skill to run Magic Box as a CLI-first TTS production tool. Prefer the bundled wrapper `scripts/magic_box_tts.py` instead of hand-writing long `podcast_generator.py` commands.

## Quick Start

From the Magic Box checkout:

```bash
python3 skills/magic-box-tts/scripts/magic_box_tts.py \
  --repo "/Users/liuchang/Apprun/chenxi/Magic Box" \
  --file story.txt \
  --profile 刘畅 \
  --output outputs/story.mp3
```

For pasted text:

```bash
python3 skills/magic-box-tts/scripts/magic_box_tts.py \
  --text "要朗读的正文" \
  --profile 刘畅 \
  --output outputs/story.mp3
```

Defaults are production-oriented:

- backend `qwen`
- format `mp3`
- speed `0.82`
- temperature `0.85`
- chunk max chars `260`
- resume enabled
- loudness normalization enabled

## Workflow

1. Locate the Magic Box checkout. Use `--repo`, `$MAGIC_BOX_DIR`, or the default `/Users/liuchang/Apprun/chenxi/Magic Box`.
2. Use a saved profile with `--profile <id-or-name>` when available. The wrapper reads `reference_clean.wav` plus `reference_clean.txt` or `transcript.txt`.
3. If no profile exists on the target Mac, require `--ref-audio` and `--ref-text`.
4. Run generation in the target Mac's local `.venv`; do not use system Python.
5. Keep `--resume` enabled for long stories. Reusing the same output path continues from the checkpoint.
6. After generation, check the final MP3 plus `outputs/quality_report.json` and the hidden segment folder `outputs/.<stem>_segments/`.

## Hermes / Remote Mac Handoff

If the user wants generation on a different Mac that has Hermes deployed:

1. Ensure that Mac has the Magic Box repo, `.venv`, model files, `ffmpeg`, and the desired voice profile.
2. Create the exact command with `--print-command`.
3. Send that command through the user's Hermes command/execution channel for the target Mac.
4. Ask Hermes to return the final output path and the last 30 log lines. If the target Mac posts files back, request the MP3; otherwise report the path on that Mac.

Example command generation:

```bash
python3 skills/magic-box-tts/scripts/magic_box_tts.py \
  --repo "/Users/liuchang/Apprun/chenxi/Magic Box" \
  --file story.txt \
  --profile 刘畅 \
  --output outputs/story.mp3 \
  --print-command
```

Do not assume a private Hermes API. Treat Hermes as the transport that runs a shell command on the target Mac unless the user provides a specific Hermes CLI/API.

For target Mac setup details, read `references/home-mac-setup.md`.

## Guardrails

- Never commit `outputs/`, `runtime/`, `audio_samples/`, `voices/profiles/`, or model files.
- Do not delete `models/`, `.venv/`, or active voice profiles during cleanup.
- If generation looks stalled, inspect process state, checkpoint, and segment files before killing it.
- Green tests or a successful process exit are not enough for production audio; inspect/listen to the resulting MP3 when quality matters.
