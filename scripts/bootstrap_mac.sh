#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${MAGIC_BOX_REPO_URL:-https://github.com/ykopp/magic-box.git}"
BRANCH="${MAGIC_BOX_BRANCH:-codex/interactive-web-app}"
INSTALL_DIR="${MAGIC_BOX_DIR:-$HOME/Workspaces/Magic Box}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

mkdir -p "$(dirname "$INSTALL_DIR")"

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -e .

if command -v brew >/dev/null 2>&1 && ! command -v ffmpeg >/dev/null 2>&1; then
  brew install ffmpeg
fi

if [[ -d "$HOME/.codex/skills" ]]; then
  ln -sfn "$INSTALL_DIR/skills/magic-box-tts" "$HOME/.codex/skills/magic-box-tts"
fi

cat <<EOF
Magic Box is ready at:
  $INSTALL_DIR

Next steps:
  1. Put Qwen/VoxCPM model files on this Mac.
  2. Copy private voice profiles into voices/profiles/.
  3. Start Web: make run-streamlit
  4. For Hermes/CLI: python3 skills/magic-box-tts/scripts/magic_box_tts.py --help
EOF
