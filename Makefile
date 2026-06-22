.PHONY: install test lint format run-streamlit run-cli route benchmark sync-update install-skill clean help

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python

help:
	@echo "Magic Box - Apple Silicon TTS podcast studio"
	@echo ""
	@echo "Targets:"
	@echo "  install         Create venv and install runtime deps"
	@echo "  test            Run pytest test suite"
	@echo "  lint            Run ruff + mypy"
	@echo "  format          Auto-format with ruff"
	@echo "  run-streamlit   Launch the Streamlit UI"
	@echo "  run-cli         Run the CLI generator (pass ARGS=...)"
	@echo "  route           Inspect model routing decisions"
	@echo "  benchmark       Run TTS backend benchmark"
	@echo "  sync-update     Pull latest GitHub code and refresh local venv"
	@echo "  install-skill   Link the bundled Codex/Hermes skill into ~/.codex/skills"
	@echo "  clean           Remove caches, runtime, and __pycache__"

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

test:
	$(PY) -m pytest tests/

lint:
	$(PY) -m ruff check .
	$(PY) -m mypy streamlit_app.py podcast_generator.py tts_backends.py utils.py

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

run-streamlit:
	$(PY) -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8507

run-cli:
	$(PY) podcast_generator.py $(ARGS)

route:
	$(PY) model_route_cli.py $(ARGS)

benchmark:
	$(PY) benchmark_tts_backends.py $(ARGS)

sync-update:
	bash scripts/sync_update.sh

install-skill:
	mkdir -p "$(HOME)/.codex/skills"
	ln -sfn "$(CURDIR)/skills/magic-box-tts" "$(HOME)/.codex/skills/magic-box-tts"

clean:
	find outputs -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
	find runtime -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
	find audio_samples -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -f .DS_Store models/.DS_Store audio_samples/.DS_Store outputs/.DS_Store
	find . -name '*.pyc' -delete
