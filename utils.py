"""
Shared utility functions for Qwen3-TTS.
Centralises helpers previously duplicated across main.py, podcast_generator.py,
and web_interface.py.
"""

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
FILENAME_MAX_LEN = 20


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_smart_path(folder_name: str) -> Optional[str]:
    """Resolve the best local model path from a folder name.

    Handles two layouts:
    - Direct folder: ``models/Foo/model.safetensors``
    - Snapshot folder (HuggingFace cache): ``models/Foo/snapshots/<hash>/``

    Always returns the **latest** snapshot when multiple exist.
    Searches in ``cwd()/models`` → ``~/podcast_generator_models``.
    """
    search_roots = [
        os.path.join(os.getcwd(), folder_name),
        os.path.join(os.getcwd(), "models", folder_name),
        os.path.join(str(Path.home()), "podcast_generator_models", folder_name),
    ]
    for full_path in search_roots:
        if not os.path.exists(full_path):
            continue
        snapshots_dir = os.path.join(full_path, "snapshots")
        if os.path.exists(snapshots_dir):
            subfolders = sorted(
                [f for f in os.listdir(snapshots_dir) if not f.startswith(".")],
                reverse=True,          # latest hash first (alphabetical = chronological for SHA)
            )
            if subfolders:
                return os.path.join(snapshots_dir, subfolders[0])
        return full_path
    return None


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def convert_audio_if_needed(input_path: str) -> Optional[str]:
    """Convert any audio file to a clean 24 kHz mono WAV.

    Returns the original path if it is already a valid WAV, or a path to a
    *temporary* WAV file that the caller is responsible for deleting via the
    returned context.  The function internally writes to a NamedTemporaryFile
    so the OS will clean it up on crash if the caller forgets.

    Raises nothing; returns ``None`` on failure.
    """
    if not os.path.exists(input_path):
        print(f"[audio] Input not found: {input_path}")
        return None

    _, ext = os.path.splitext(input_path)

    # Fast-path: already a valid WAV
    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, "rb") as f:
                if f.getnframes() > 0 and f.getsampwidth() > 0:
                    return input_path
        except wave.Error:
            pass  # Fall through to ffmpeg conversion

    # Slow-path: convert with ffmpeg into a named temp file
    # Using delete=False so we can pass the path to external tools; caller must
    # remove the file when done.
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="qwen_tts_",
            dir=os.getcwd(),
            delete=False,
        )
        tmp.close()
        tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", input_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            tmp_path,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print("[audio] ffmpeg ran but output is empty.")
            _safe_remove(tmp_path)
            return None
        return tmp_path
    except subprocess.CalledProcessError as e:
        print(f"[audio] ffmpeg error: {e.stderr.strip()}")
        _safe_remove(tmp_path if "tmp_path" in dir() else None)
        return None
    except FileNotFoundError:
        print("[audio] ffmpeg not found. Install with: brew install ffmpeg")
        return None


def _safe_remove(path: Optional[str]):
    """Delete a file silently if it exists."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_text(text: str, max_chars: int = 80) -> list[str]:
    """Split text into chunks suitable for TTS generation.

    Handles both Chinese (splits on 。！？，) and English (splits on . ! ?).
    Never splits in the middle of a word.  Chunks that exceed *max_chars*
    due to a single long sentence are kept as-is (not further split).
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Insert split markers after Chinese/English sentence terminators
    text = re.sub(r"([。！？!?])\s*", r"\1\n", text)
    # Also split on Chinese comma-sized pauses for very long runs
    text = re.sub(r"([，,])\s*", r"\1\n", text)

    raw_sentences = [s.strip() for s in text.split("\n") if s.strip()]

    chunks: list[str] = []
    current = ""

    for sent in raw_sentences:
        if not current:
            current = sent
        elif len(current) + len(sent) <= max_chars:
            current += sent
        else:
            chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Loudness normalisation (pyloudnorm, optional)
# ---------------------------------------------------------------------------

def normalise_loudness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Apply EBU R128 loudness normalisation to *audio*.

    Target: -16 LUFS (podcast/streaming standard).
    Falls back silently to the original audio if pyloudnorm is not installed.
    """
    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        if np.isinf(loudness):          # silence or too-short clip
            return audio
        normalised = pyln.normalize.loudness(audio, loudness, target_lufs)
        # Clamp to [-1, 1] to avoid hard clipping
        return np.clip(normalised, -1.0, 1.0)
    except ImportError:
        return audio
    except Exception as e:
        print(f"[loudness] Normalisation skipped: {e}")
        return audio


# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------

def make_output_filename(text_snippet: str, ext: str = "wav") -> str:
    """Return a timestamped, filesystem-safe filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean = re.sub(r"[^\w\s-]", "", text_snippet)
    clean = clean.strip().replace(" ", "_")[:FILENAME_MAX_LEN] or "audio"
    return f"{timestamp}_{clean}.{ext}"


# ---------------------------------------------------------------------------
# Checkpoint helpers for long-form generation
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION = 1


def save_checkpoint(checkpoint_path: str, chunks: list[str], completed_indices: list[int], audio_segments: list[str]):
    """Persist generation progress to disk.

    *audio_segments* is a list of absolute paths to per-chunk WAV files.
    """
    data = {
        "version": _CHECKPOINT_VERSION,
        "saved_at": datetime.now().isoformat(),
        "chunks": chunks,
        "completed": completed_indices,
        "segments": audio_segments,
    }
    tmp = checkpoint_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, checkpoint_path)
    except OSError as e:
        print(f"[checkpoint] Save failed: {e}")


def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Load a checkpoint file.  Returns None if missing or corrupt."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != _CHECKPOINT_VERSION:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Thread-safe model cache
# ---------------------------------------------------------------------------

class ModelCache:
    """Singleton cache that holds the currently loaded MLX model.

    Thread-safe: a ``threading.Lock`` guards all load / read operations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._path: Optional[str] = None

    def get(self, resolved_path: str):
        """Return the cached model if *resolved_path* matches the loaded one."""
        with self._lock:
            if self._path == resolved_path:
                return self._model
            return None

    def set(self, resolved_path: str, model):
        with self._lock:
            self._model = model
            self._path = resolved_path

    def clear(self):
        with self._lock:
            self._model = None
            self._path = None

    @property
    def current_path(self) -> Optional[str]:
        with self._lock:
            return self._path


model_cache = ModelCache()
