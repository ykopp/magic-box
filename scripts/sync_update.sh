#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BRANCH="${MAGIC_BOX_BRANCH:-codex/interactive-web-app}"
REMOTE="${MAGIC_BOX_REMOTE:-origin}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

cd "$REPO_DIR"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  if git remote get-url fork >/dev/null 2>&1; then
    REMOTE="fork"
  else
    echo "Git remote not found: $REMOTE" >&2
    exit 2
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree has local code changes. Commit or stash them before syncing." >&2
  exit 2
fi

git fetch "$REMOTE" "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only "$REMOTE" "$BRANCH"

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -e .

if [[ -d "$HOME/.codex/skills" ]]; then
  ln -sfn "$REPO_DIR/skills/magic-box-tts" "$HOME/.codex/skills/magic-box-tts"
fi

echo "Magic Box updated on branch $BRANCH from $REMOTE."
