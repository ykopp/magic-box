"""
Shared utility functions for Qwen3-TTS.
Centralises helpers used by the Streamlit app and CLI generator.
"""

import json
import os
import re
import math
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
FILENAME_MAX_LEN = 20
APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = APP_DIR / "runtime"


@dataclass(frozen=True)
class AudioMetrics:
    duration_seconds: float
    sample_rate: int
    peak: float
    rms: float
    max_adjacent_jump: float
    p999_adjacent_jump: float


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_smart_path(folder_name: str) -> Optional[str]:
    """Resolve the best local model path from a folder name.

    Handles two layouts:
    - Direct folder: ``models/Foo/model.safetensors``
    - Snapshot folder (HuggingFace cache): ``models/Foo/snapshots/<hash>/``

    Uses ``refs/main`` for snapshot models when present, then falls back to the
    newest snapshot directory by modification time.
    Searches in the app directory and ``~/podcast_generator_models``.
    """
    search_roots = [
        os.path.join(str(APP_DIR), folder_name),
        os.path.join(str(APP_DIR), "models", folder_name),
        os.path.join(str(Path.home()), "podcast_generator_models", folder_name),
    ]
    for full_path in search_roots:
        if not os.path.exists(full_path):
            continue
        snapshots_dir = os.path.join(full_path, "snapshots")
        if os.path.exists(snapshots_dir):
            refs_main = os.path.join(full_path, "refs", "main")
            if os.path.exists(refs_main):
                try:
                    snapshot_name = Path(refs_main).read_text(encoding="utf-8").strip()
                    snapshot_path = os.path.join(snapshots_dir, snapshot_name)
                    if snapshot_name and os.path.exists(snapshot_path):
                        return snapshot_path
                except OSError:
                    pass
            subfolders = [
                os.path.join(snapshots_dir, f)
                for f in os.listdir(snapshots_dir)
                if not f.startswith(".")
            ]
            if subfolders:
                return max(subfolders, key=lambda path: os.path.getmtime(path))
        return full_path
    return None


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def convert_audio_if_needed(input_path: str) -> Optional[str]:
    """Convert any audio file to a clean 24 kHz mono WAV.

    Returns the original path only if it is already a valid 24 kHz mono PCM
    WAV, or a temporary WAV path that the caller should delete via
    cleanup_reference_audio.

    Raises nothing; returns ``None`` on failure.
    """
    if not os.path.exists(input_path):
        print(f"[audio] Input not found: {input_path}")
        return None

    _, ext = os.path.splitext(input_path)

    # Fast-path: already exactly what Qwen expects.
    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, "rb") as f:
                if (
                    f.getnframes() > 0
                    and f.getframerate() == SAMPLE_RATE
                    and f.getnchannels() == 1
                    and f.getsampwidth() in {2, 3, 4}
                ):
                    return input_path
        except wave.Error:
            pass  # Fall through to ffmpeg conversion

    # Slow-path: convert with ffmpeg into a named temp file
    # Using delete=False so we can pass the path to external tools; caller must
    # remove the file when done.
    try:
        runtime_dir = RUNTIME_DIR
        runtime_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="qwen_tts_",
            dir=str(runtime_dir),
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
        _safe_remove(tmp_path if "tmp_path" in dir() else None)
        return None


def check_audio_health(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    label: str = "audio",
    low_rms_threshold: float = 1e-4,
    low_peak_threshold: float = 1e-4,
    high_peak_threshold: float = 0.98,
) -> list[str]:
    """Return lightweight audio health warnings without raising errors."""
    warnings: list[str] = []
    arr = np.asarray(audio)

    if sample_rate <= 0:
        warnings.append(f"{label}: invalid sample rate ({sample_rate})")

    if arr.size == 0:
        warnings.append(f"{label}: empty audio")
        return warnings

    if not np.issubdtype(arr.dtype, np.number):
        warnings.append(f"{label}: non-numeric audio dtype ({arr.dtype})")
        return warnings

    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        warnings.append(f"{label}: contains NaN or Inf samples")

    finite = arr[finite_mask].astype(np.float64, copy=False)
    if finite.size == 0:
        warnings.append(f"{label}: no finite samples")
        return warnings

    peak = float(np.max(np.abs(finite)))
    rms = float(math.sqrt(np.mean(np.square(finite))))

    if rms < low_rms_threshold:
        warnings.append(f"{label}: very low RMS ({rms:.2e})")
    if peak < low_peak_threshold:
        warnings.append(f"{label}: very low peak ({peak:.2e})")
    if peak > 1.0:
        warnings.append(f"{label}: peak exceeds full scale ({peak:.3f})")
    elif peak >= high_peak_threshold:
        warnings.append(f"{label}: peak is near clipping ({peak:.3f})")

    return warnings


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float32, copy=True)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=True)
    if arr.ndim == 2:
        return np.mean(arr, axis=1, dtype=np.float64).astype(np.float32)
    return arr.reshape(-1).astype(np.float32, copy=True)


def measure_audio(audio: np.ndarray, sample_rate: int) -> AudioMetrics:
    """Return simple mono audio metrics for generated TTS samples."""
    arr = _to_mono_float(audio)
    if arr.size == 0:
        return AudioMetrics(
            duration_seconds=0.0,
            sample_rate=int(sample_rate),
            peak=0.0,
            rms=0.0,
            max_adjacent_jump=0.0,
            p999_adjacent_jump=0.0,
        )

    finite = arr[np.isfinite(arr)].astype(np.float64, copy=False)
    if finite.size == 0:
        return AudioMetrics(
            duration_seconds=float(arr.size / sample_rate) if sample_rate > 0 else 0.0,
            sample_rate=int(sample_rate),
            peak=float("nan"),
            rms=float("nan"),
            max_adjacent_jump=float("nan"),
            p999_adjacent_jump=float("nan"),
        )

    jumps = np.abs(np.diff(finite))
    max_jump = float(np.max(jumps)) if jumps.size else 0.0
    p999_jump = float(np.percentile(jumps, 99.9)) if jumps.size else 0.0

    return AudioMetrics(
        duration_seconds=float(arr.size / sample_rate) if sample_rate > 0 else 0.0,
        sample_rate=int(sample_rate),
        peak=float(np.max(np.abs(finite))),
        rms=float(math.sqrt(np.mean(np.square(finite)))),
        max_adjacent_jump=max_jump,
        p999_adjacent_jump=p999_jump,
    )


def validate_generated_audio(
    audio: np.ndarray,
    sample_rate: int,
    label: str = "audio",
    *,
    min_duration_seconds: float = 0.05,
    low_rms_threshold: float = 1e-4,
    low_peak_threshold: float = 1e-4,
    near_clip_threshold: float = 0.98,
) -> list[str]:
    """Return warnings/errors that should reject bad generated audio chunks."""
    warnings: list[str] = []
    arr = np.asarray(audio)

    if sample_rate <= 0:
        warnings.append(f"{label}: invalid sample rate ({sample_rate})")

    if arr.size == 0:
        warnings.append(f"{label}: empty audio")
        return warnings

    if not np.issubdtype(arr.dtype, np.number):
        warnings.append(f"{label}: non-numeric audio dtype ({arr.dtype})")
        return warnings

    if not np.all(np.isfinite(arr)):
        warnings.append(f"{label}: contains NaN or Inf samples")

    metrics = measure_audio(arr, sample_rate)

    if sample_rate > 0 and metrics.duration_seconds < min_duration_seconds:
        warnings.append(f"{label}: too short ({metrics.duration_seconds:.3f}s)")

    if np.isfinite(metrics.rms) and metrics.rms < low_rms_threshold:
        warnings.append(f"{label}: very low RMS ({metrics.rms:.2e})")
    if np.isfinite(metrics.peak) and metrics.peak < low_peak_threshold:
        warnings.append(f"{label}: very low peak ({metrics.peak:.2e})")

    if np.isfinite(metrics.peak):
        if metrics.peak > 1.0:
            warnings.append(f"{label}: peak exceeds full scale ({metrics.peak:.3f})")
        elif metrics.peak >= near_clip_threshold:
            warnings.append(f"{label}: peak is near clipping ({metrics.peak:.3f})")

    return warnings


def smooth_join_audio(
    chunks: list[np.ndarray],
    sample_rate: int,
    pause_seconds: float = 0.12,
    fade_ms: float = 20,
) -> np.ndarray:
    """Join mono audio chunks with short fades and silence between segments."""
    if not chunks:
        return np.array([], dtype=np.float32)

    pause_samples = max(0, int(round(sample_rate * pause_seconds))) if sample_rate > 0 else 0
    fade_samples = max(0, int(round(sample_rate * fade_ms / 1000))) if sample_rate > 0 else 0
    silence = np.zeros(pause_samples, dtype=np.float32)
    pieces: list[np.ndarray] = []

    for chunk in chunks:
        segment = _to_mono_float(chunk)
        if segment.size == 0:
            continue

        segment = segment.copy()
        active_fade = min(fade_samples, segment.size // 2)
        if active_fade > 0:
            fade_in = np.linspace(0.0, 1.0, active_fade, endpoint=True, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, active_fade, endpoint=True, dtype=np.float32)
            segment[:active_fade] *= fade_in
            segment[-active_fade:] *= fade_out

        if pieces and pause_samples > 0:
            pieces.append(silence.copy())
        pieces.append(segment)

    if not pieces:
        return np.array([], dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def check_audio_file_health(path: str, *, label: str = "audio file") -> list[str]:
    """Read an audio file and return health warnings; failures become warnings."""
    try:
        audio, sample_rate = sf.read(path)
    except Exception as exc:
        return [f"{label}: could not read audio file ({exc})"]
    return check_audio_health(audio, int(sample_rate), label=label)


def _safe_remove(path: Optional[str]):
    """Delete a file silently if it exists."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> str:
    """Convert a WAV file to MP3 with ffmpeg and return the MP3 path."""

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"MP3 conversion failed: {exc.stderr.strip()}") from exc

    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise RuntimeError("MP3 conversion produced an empty file.")
    return mp3_path


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_text(text: str, max_chars: int = 80) -> list[str]:
    """Split text into chunks suitable for TTS generation.

    Major Chinese/English sentence punctuation is used first. Commas and
    semicolons are only used inside sentences that exceed *max_chars*.
    English words are never split.
    """
    if max_chars <= 0:
        max_chars = 80

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences: list[str] = []
    for line in text.split("\n"):
        sentences.extend(_split_on_major_punctuation(line))

    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) > max_chars:
            pieces.extend(_split_long_sentence(sentence, max_chars))
        else:
            pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for sent in pieces:
        if not current:
            current = sent
            continue

        separator = " " if _needs_join_space(current, sent) else ""
        candidate = current + separator + sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


def _split_on_major_punctuation(text: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    major_punctuation = set("。！？.!?")

    for char in text.strip():
        current.append(char)
        if char in major_punctuation:
            piece = "".join(current).strip()
            if piece:
                pieces.append(piece)
            current = []

    tail = "".join(current).strip()
    if tail:
        pieces.append(tail)
    return pieces


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    minor_parts = _split_on_minor_punctuation(sentence)
    if len(minor_parts) == 1:
        return _split_without_breaking_words(sentence, max_chars)

    pieces: list[str] = []
    current = ""
    for part in minor_parts:
        if not current:
            current = part
            continue

        separator = " " if _needs_join_space(current, part) else ""
        candidate = current + separator + part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.extend(_split_without_breaking_words(current, max_chars))
            current = part

    if current:
        pieces.extend(_split_without_breaking_words(current, max_chars))
    return pieces


def _split_on_minor_punctuation(text: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    minor_punctuation = set("，,；;")

    for char in text.strip():
        current.append(char)
        if char in minor_punctuation:
            piece = "".join(current).strip()
            if piece:
                pieces.append(piece)
            current = []

    tail = "".join(current).strip()
    if tail:
        pieces.append(tail)
    return pieces


def _split_without_breaking_words(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    if len(words) <= 1:
        return [text]

    pieces: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word

    if current:
        pieces.append(current)
    return pieces


def _needs_join_space(left: str, right: str) -> bool:
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return False
    if left[-1].isascii() and right[0].isascii():
        return left[-1].isalnum() or left[-1] in ".!?,;:"
    return False


# ---------------------------------------------------------------------------
# Loudness normalisation (pyloudnorm, optional)
# ---------------------------------------------------------------------------

def normalise_loudness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -16.0,
    peak_ceiling: float = 0.95,
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
        return limit_audio_peak(normalised, ceiling=peak_ceiling)
    except ImportError:
        return audio
    except Exception as e:
        print(f"[loudness] Normalisation skipped: {e}")
        return audio


def limit_audio_peak(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Scale audio down when needed so peaks stay below *ceiling* without clipping."""
    arr = np.asarray(audio)
    if arr.size == 0:
        return arr

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    peak = float(np.max(np.abs(finite)))
    if peak <= 0 or peak <= ceiling:
        return arr

    return arr * (ceiling / peak)


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

_CHECKPOINT_VERSION = 2


def save_checkpoint(
    checkpoint_path: str,
    chunks: list[str],
    completed_indices: list[int],
    audio_segments: list[str],
    metadata: Optional[dict] = None,
):
    """Persist generation progress to disk.

    *audio_segments* is a list of absolute paths to per-chunk WAV files.
    """
    data = {
        "version": _CHECKPOINT_VERSION,
        "saved_at": datetime.now().isoformat(),
        "chunks": chunks,
        "completed": completed_indices,
        "segments": audio_segments,
        "metadata": metadata or {},
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
