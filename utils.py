"""
Shared utility functions for Qwen3-TTS.
Centralises helpers used by the Streamlit app and CLI generator.
"""

import json
import logging
import os
import re
import math
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
FILENAME_MAX_LEN = 20
APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = APP_DIR / "runtime"


class ReferenceAudioError(RuntimeError):
    """Raised when a reference audio file cannot produce a usable voice clip."""


@dataclass(frozen=True)
class ReferencePair:
    clean_audio_path: Path
    clean_text_path: Path
    quality_path: Path
    payload: dict[str, Any]


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

    Returns the original path only if it is already a valid 24 kHz mono PCM
    WAV, or a temporary WAV path that the caller should delete via
    cleanup_reference_audio.

    Raises nothing; returns ``None`` on failure.
    """
    if not os.path.exists(input_path):
        _log.warning("Input not found: %s", input_path)
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
    tmp_path: Optional[str] = None
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
            _log.warning("ffmpeg ran but output is empty.")
            _safe_remove(tmp_path)
            return None
        return tmp_path
    except subprocess.CalledProcessError as e:
        _log.warning("ffmpeg error: %s", e.stderr.strip())
        _safe_remove(tmp_path)
        return None
    except FileNotFoundError:
        # Match the loud-failure behaviour of ``convert_wav_to_mp3`` so the
        # two audio-format code paths surface ffmpeg-missing errors the same
        # way. Returning None silently here meant users uploading a non-WAV
        # reference audio got a generic "参考音频转换失败" with no hint that
        # ffmpeg was the actual cause.
        _safe_remove(tmp_path)
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg "
            "(reference audio conversion requires ffmpeg for any non-24kHz-mono "
            "WAV inputs)."
        )


def _convert_audio_to_wav(input_path: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(input_path).suffix.lower() == ".wav":
        try:
            audio, sample_rate = sf.read(input_path, always_2d=False)
            arr = np.asarray(audio, dtype=np.float32)
            if int(sample_rate) == SAMPLE_RATE and arr.size > 0:
                if arr.ndim == 2:
                    if arr.shape[1] == 1:
                        arr = arr[:, 0]
                    else:
                        arr = np.array([], dtype=np.float32)
                if arr.size > 0:
                    sf.write(output_path, arr, SAMPLE_RATE, subtype="PCM_16")
                    return output_path
        except ReferenceAudioError:
            raise
        except Exception:
            pass

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", input_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ReferenceAudioError(
            "ffmpeg not found. Install with: brew install ffmpeg "
            "(reference audio cleanup requires ffmpeg)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffmpeg error"
        raise ReferenceAudioError(f"reference audio conversion failed: {detail}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ReferenceAudioError("reference audio conversion produced an empty WAV file")
    return output_path


def _mono_float_audio(path: Path) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(path, always_2d=False)
    except Exception as exc:
        raise ReferenceAudioError(f"reference audio could not be read: {exc}") from exc

    if int(sample_rate) != SAMPLE_RATE:
        raise ReferenceAudioError(
            f"reference audio conversion produced {sample_rate} Hz, expected {SAMPLE_RATE} Hz"
        )

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1, dtype=np.float32)
    if arr.size == 0:
        raise ReferenceAudioError("reference audio is empty")
    if not np.all(np.isfinite(arr)):
        raise ReferenceAudioError("reference audio contains NaN or Inf samples")
    return arr


def _frame_rms(audio: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    if len(audio) < frame_samples:
        return np.array([], dtype=np.float32)
    frame_count = 1 + (len(audio) - frame_samples) // hop_samples
    rms = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_samples
        frame = audio[start:start + frame_samples]
        rms[index] = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    return rms


def _longest_false_run_seconds(mask: np.ndarray, hop_seconds: float) -> float:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return float(longest * hop_seconds)


def _window_quality(
    audio: np.ndarray,
    frame_rms: np.ndarray,
    active: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    segment = audio[start:end]
    hop_samples = int(0.01 * SAMPLE_RATE)
    frame_start = max(0, start // hop_samples)
    frame_end = min(len(active), max(frame_start + 1, end // hop_samples))
    active_window = active[frame_start:frame_end]

    rms = float(np.sqrt(np.mean(np.square(segment), dtype=np.float64)))
    peak = float(np.max(np.abs(segment)))
    active_ratio = float(np.mean(active_window)) if active_window.size else 0.0
    pause_seconds = _longest_false_run_seconds(active_window, 0.01)
    clipped_ratio = float(np.mean(np.abs(segment) >= 0.995))
    duration = float(len(segment) / SAMPLE_RATE)
    return {
        "start_seconds": round(start / SAMPLE_RATE, 3),
        "end_seconds": round(end / SAMPLE_RATE, 3),
        "duration_seconds": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "active_ratio": round(active_ratio, 4),
        "longest_pause_seconds": round(pause_seconds, 3),
        "clipped_ratio": round(clipped_ratio, 6),
        "frame_rms_median": round(float(np.median(frame_rms)) if frame_rms.size else 0.0, 6),
    }


def _reference_rejection_reason(quality: dict[str, Any]) -> str | None:
    if quality["duration_seconds"] < 3.0:
        return "no continuous voice segment at least 3 seconds long"
    if quality["duration_seconds"] > 8.0:
        return "candidate voice segment is longer than 8 seconds"
    if quality["rms"] < 0.01:
        return f"candidate RMS is too low ({quality['rms']:.6f})"
    if quality["rms"] > 0.35:
        return f"candidate RMS is too high ({quality['rms']:.6f})"
    if quality["peak"] < 0.05:
        return f"candidate peak is too low ({quality['peak']:.6f})"
    if quality["peak"] > 0.98:
        return f"candidate peak is too high ({quality['peak']:.6f})"
    if quality["clipped_ratio"] > 0.001:
        return f"candidate appears clipped ({quality['clipped_ratio']:.4%} clipped samples)"
    if quality["active_ratio"] < 0.65:
        return f"candidate has too much silence ({quality['active_ratio']:.1%} active)"
    if quality["longest_pause_seconds"] > 0.75:
        return f"candidate has a long pause ({quality['longest_pause_seconds']:.2f}s)"
    return None


def _read_audio_for_audit(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, always_2d=False)
    except Exception as first_exc:
        converted_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                converted_path = Path(tmp.name)
            _convert_audio_to_wav(str(path), converted_path)
            audio, sample_rate = sf.read(converted_path, always_2d=False)
        except Exception as exc:
            raise ReferenceAudioError(f"reference audio could not be read: {first_exc}; ffmpeg conversion failed: {exc}") from exc
        finally:
            if converted_path is not None:
                safe_remove(str(converted_path))

    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    return np.asarray(arr, dtype=np.float32), int(sample_rate)


def _reference_active_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if sample_rate <= 0 or audio.size == 0:
        return {
            "active_ratio": 0.0,
            "longest_low_energy_run": 0.0,
            "frame_rms_median": 0.0,
            "low_energy_threshold": 0.0,
        }

    frame_samples = max(1, int(0.025 * sample_rate))
    hop_samples = max(1, int(0.01 * sample_rate))
    rms_values = _frame_rms(audio, frame_samples, hop_samples)
    if rms_values.size == 0:
        return {
            "active_ratio": 0.0,
            "longest_low_energy_run": float(len(audio) / sample_rate),
            "frame_rms_median": 0.0,
            "low_energy_threshold": 0.0,
        }

    median_rms = float(np.median(rms_values))
    threshold = max(0.006, min(0.03, median_rms * 0.2))
    active = rms_values >= threshold
    return {
        "active_ratio": float(np.mean(active)),
        "longest_low_energy_run": _longest_false_run_seconds(active, hop_samples / sample_rate),
        "frame_rms_median": median_rms,
        "low_energy_threshold": threshold,
    }


def _recommendations_for_issues(issues: list[str]) -> list[str]:
    recommendations: list[str] = []
    if any("duration" in issue for issue in issues):
        recommendations.append("Provide a clean spoken reference between 3 and 20 seconds when possible; 3-30 seconds is accepted.")
    if any("RMS" in issue or "peak" in issue or "low-energy" in issue or "pause" in issue for issue in issues):
        recommendations.append("Record steady speech without long silence, clipping, or very low volume.")
    if any("NaN" in issue or "Inf" in issue or "sample jumps" in issue for issue in issues):
        recommendations.append("Re-export or re-record the file; the waveform contains invalid or discontinuous samples.")
    if not recommendations:
        recommendations.append("Reference audio is ready for voice profile creation.")
    return recommendations


def _audit_reference_array(
    audio: np.ndarray,
    sample_rate: int,
    *,
    source_path: Path,
    ref_text: str | None = None,
) -> dict[str, Any]:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1, dtype=np.float32)

    metrics = compute_audio_quality_metrics(
        arr,
        sample_rate,
        low_energy_threshold=0.006,
    )
    active_metrics = _reference_active_metrics(arr[np.isfinite(arr)] if arr.size else arr, sample_rate)
    metrics.update(active_metrics)

    duration = float(metrics.get("duration", 0.0) or 0.0)
    rms = float(metrics.get("rms", 0.0) or 0.0)
    peak = float(metrics.get("peak", 0.0) or 0.0)
    jump_05 = int(metrics.get("jump_count_gt_0_5", 0) or 0)
    max_jump = float(metrics.get("max_jump", 0.0) or 0.0)
    low_run = float(metrics.get("longest_low_energy_run", 0.0) or 0.0)
    active_ratio = float(metrics.get("active_ratio", 0.0) or 0.0)

    issues: list[str] = []
    warnings: list[str] = []
    if not bool(metrics.get("finite", True)):
        issues.append("reference audio contains NaN or Inf samples")
    if duration < 3.0 or duration > 30.0:
        issues.append(f"reference duration must be 3-30 seconds ({duration:.2f}s)")
    elif duration > 20.0:
        warnings.append(f"reference is longer than ideal 3-20 seconds ({duration:.2f}s)")
    if rms < 0.01:
        issues.append(f"reference RMS is too low ({rms:.6f})")
    if rms > 0.35:
        issues.append(f"reference RMS is too high ({rms:.6f})")
    if peak < 0.05:
        issues.append(f"reference peak is too low ({peak:.6f})")
    if peak >= 0.98:
        issues.append(f"reference peak is near clipping ({peak:.6f})")
    elif peak >= 0.95:
        warnings.append(f"reference peak is high ({peak:.6f})")
    if jump_05 > 12 or max_jump >= 0.8:
        issues.append(f"reference has severe sample jumps ({jump_05} > 0.5, max {max_jump:.3f})")
    elif jump_05 > 0:
        warnings.append(f"reference has {jump_05} sample jumps > 0.5")
    if duration >= 3.0 and active_ratio < 0.45:
        issues.append(f"reference has too much low-energy audio ({active_ratio:.1%} active)")
    elif duration >= 3.0 and active_ratio < 0.65:
        warnings.append(f"reference has some low-energy audio ({active_ratio:.1%} active)")
    if low_run > 1.5:
        issues.append(f"reference has a long low-energy pause ({low_run:.2f}s)")
    elif low_run > 0.75:
        warnings.append(f"reference has a long low-energy pause ({low_run:.2f}s)")

    payload = {
        "ok": not issues,
        "source_path": str(source_path),
        "sample_rate": sample_rate,
        "duration": duration,
        "duration_seconds": duration,
        "issues": issues,
        "warnings": warnings,
        "recommendations": _recommendations_for_issues(issues + warnings),
        "metrics": metrics,
    }
    if ref_text is not None:
        payload["ref_text_chars"] = len(ref_text.strip())
    return payload


def audit_reference_audio(input_path: str | Path, ref_text: str | None = None) -> dict[str, Any]:
    """Return a deterministic quality audit for a candidate reference audio.

    This function is intentionally read-only: it does not trim, normalise, or
    write files. It only inspects waveform metrics so callers can reject bad
    references before creating profile artifacts.
    """
    source = Path(input_path)
    if not source.exists():
        return {
            "ok": False,
            "source_path": str(source),
            "sample_rate": 0,
            "duration": 0.0,
            "duration_seconds": 0.0,
            "issues": [f"reference audio not found: {source}"],
            "recommendations": ["Provide an existing audio file between 3 and 20 seconds."],
            "metrics": {},
        }

    try:
        audio, sample_rate = _read_audio_for_audit(source)
    except ReferenceAudioError as exc:
        return {
            "ok": False,
            "source_path": str(source),
            "sample_rate": 0,
            "duration": 0.0,
            "duration_seconds": 0.0,
            "issues": [str(exc)],
            "recommendations": ["Provide a readable WAV/MP3/M4A/FLAC/OGG reference file."],
            "metrics": {},
        }
    return _audit_reference_array(audio, sample_rate, source_path=source, ref_text=ref_text)


def prepare_reference_pair(
    input_path: str | Path,
    ref_text: str,
    output_dir: str | Path,
) -> ReferencePair:
    """Create schema-v2 reference artifacts only after a full-file audit passes."""
    source = Path(input_path)
    if not source.exists():
        raise ReferenceAudioError(f"reference audio not found: {source}")
    if not ref_text.strip():
        raise ReferenceAudioError("reference text is required for a voice profile")

    output_root = Path(output_dir)
    converted_path = output_root / "reference_converted.wav"
    clean_path = output_root / "reference_clean.wav"
    text_path = output_root / "reference_clean.txt"
    quality_path = output_root / "reference_quality.json"

    try:
        _convert_audio_to_wav(str(source), converted_path)
        clean_audio = remove_dc_offset(_mono_float_audio(converted_path))
        payload = _audit_reference_array(clean_audio, SAMPLE_RATE, source_path=source, ref_text=ref_text)
        if not payload["ok"]:
            raise ReferenceAudioError("; ".join(payload["issues"]))

        output_root.mkdir(parents=True, exist_ok=True)
        sf.write(clean_path, clean_audio, SAMPLE_RATE, subtype="PCM_16")
        text_path.write_text(ref_text.strip(), encoding="utf-8")
        payload.update(
            {
                "clean_path": str(clean_path),
                "text_path": str(text_path),
                "channels": 1,
            }
        )
        quality_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ReferencePair(clean_path, text_path, quality_path, payload)
    except ReferenceAudioError:
        safe_remove(str(clean_path))
        safe_remove(str(text_path))
        safe_remove(str(quality_path))
        raise
    finally:
        safe_remove(str(converted_path))


def _select_reference_segment(audio: np.ndarray) -> tuple[int, int, dict[str, Any]]:
    min_samples = int(3.0 * SAMPLE_RATE)
    max_samples = int(8.0 * SAMPLE_RATE)
    if len(audio) < min_samples:
        raise ReferenceAudioError("reference audio is shorter than 3 seconds after conversion")

    frame_samples = int(0.025 * SAMPLE_RATE)
    hop_samples = int(0.01 * SAMPLE_RATE)
    rms_values = _frame_rms(audio, frame_samples, hop_samples)
    if rms_values.size == 0:
        raise ReferenceAudioError("reference audio is too short to analyse")

    rms_floor = max(0.006, float(np.percentile(rms_values, 20)) * 2.5)
    rms_floor = min(rms_floor, 0.03)
    active = rms_values >= rms_floor
    if float(np.mean(active)) < 0.05:
        raise ReferenceAudioError("reference audio contains no detectable voiced segment")

    best: tuple[float, int, int, dict[str, Any]] | None = None
    total = len(audio)
    step = int(0.25 * SAMPLE_RATE)
    candidate_lengths = [max_samples, int(6.0 * SAMPLE_RATE), int(4.0 * SAMPLE_RATE), min_samples]
    for window_len in candidate_lengths:
        if total < window_len:
            continue
        for start in range(0, total - window_len + 1, step):
            end = start + window_len
            quality = _window_quality(audio, rms_values, active, start, end)
            reason = _reference_rejection_reason(quality)
            if reason:
                continue
            score = (
                quality["active_ratio"] * 4.0
                - quality["longest_pause_seconds"] * 1.5
                + min(quality["duration_seconds"], 6.0) * 0.1
                - abs(quality["rms"] - 0.08)
            )
            if best is None or score > best[0]:
                best = (score, start, end, quality)

    if best is not None:
        _, start, end, quality = best
        return start, end, quality

    active_indices = np.flatnonzero(active)
    if active_indices.size:
        start = max(0, int(active_indices[0] * hop_samples))
        end = min(total, int(active_indices[-1] * hop_samples + frame_samples))
        if end - start >= min_samples:
            end = min(end, start + max_samples)
            quality = _window_quality(audio, rms_values, active, start, end)
            reason = _reference_rejection_reason(quality)
            raise ReferenceAudioError(reason or "reference audio did not pass quality checks")

    full_quality = _window_quality(audio, rms_values, active, 0, min(total, max_samples))
    reason = _reference_rejection_reason(full_quality)
    raise ReferenceAudioError(reason or "reference audio did not contain a usable 3-8 second voice segment")


def prepare_reference_audio_clip(input_path: str | Path, output_dir: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    """Create ``reference_clean.wav`` and ``reference_quality.json`` for cloning.

    The selected clip is always 24 kHz mono PCM WAV and must be 3-8 seconds,
    mostly voiced, without long pauses, very low level, or clipping.
    """
    source = Path(input_path)
    if not source.exists():
        raise ReferenceAudioError(f"reference audio not found: {source}")

    output_root = Path(output_dir)
    converted_path = output_root / "reference_converted.wav"
    clean_path = output_root / "reference_clean.wav"
    quality_path = output_root / "reference_quality.json"

    try:
        _convert_audio_to_wav(str(source), converted_path)
        audio = remove_dc_offset(_mono_float_audio(converted_path))
        start, end, quality = _select_reference_segment(audio)
        clean_audio = audio[start:end]
        sf.write(clean_path, clean_audio, SAMPLE_RATE, subtype="PCM_16")

        payload = {
            "ok": True,
            "source_path": str(source),
            "clean_path": str(clean_path),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            **quality,
        }
        quality_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return clean_path, quality_path, payload
    except ReferenceAudioError as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "source_path": str(source),
                    "failure_reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        safe_remove(str(clean_path))
        raise
    finally:
        safe_remove(str(converted_path))


def check_audio_health(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    label: str = "audio",
    low_rms_threshold: float = 1e-4,
    low_peak_threshold: float = 1e-4,
    high_peak_threshold: float = 0.95,
) -> list[str]:
    """Return lightweight audio health warnings without raising errors.

    The default ``high_peak_threshold`` is aligned with the production
    limiter ceiling (``limit_audio_peak(ceiling=0.95)``). Any peak >= 0.95
    means the soft-knee limiter *failed* to bring the signal below the
    ceiling — flag it so the operator can investigate. A higher threshold
    (e.g. 0.98) would let limiter failures slip through silently and only
    catch true full-scale clipping, which is too late.
    """
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


def check_audio_file_health(path: str, *, label: str = "audio file") -> list[str]:
    """Read an audio file and return health warnings; failures become warnings."""
    try:
        audio, sample_rate = sf.read(path)
    except Exception as exc:
        return [f"{label}: could not read audio file ({exc})"]
    return check_audio_health(audio, int(sample_rate), label=label)


def compute_audio_quality_metrics(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    low_energy_threshold: float = 1e-4,
    pause_rms_threshold: float = 0.003,
) -> dict[str, float | int | bool]:
    """Compute deterministic audio-quality metrics for chunks and final audio."""
    arr = np.asarray(audio)
    duration = float(arr.shape[0] / sample_rate) if sample_rate > 0 else 0.0
    metrics: dict[str, float | int | bool] = {
        "duration": duration,
        "rms": 0.0,
        "peak": 0.0,
        "low_energy_ratio": 1.0 if arr.size else 0.0,
        "longest_low_energy_run": duration if arr.size else 0.0,
        "rms_low_energy_ratio": 1.0 if arr.size else 0.0,
        "longest_rms_low_energy_run": duration if arr.size else 0.0,
        "rms_low_energy_threshold": float(pause_rms_threshold),
        "jump_count_gt_0_3": 0,
        "jump_count_gt_0_5": 0,
        "max_jump": 0.0,
        "finite": bool(arr.size == 0),
    }

    if sample_rate <= 0 or arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return metrics

    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    arr = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    metrics["finite"] = bool(np.all(finite_mask))
    if not np.any(finite_mask):
        return metrics

    finite = arr[finite_mask]
    abs_audio = np.abs(finite)
    metrics["rms"] = float(math.sqrt(np.mean(np.square(finite))))
    metrics["peak"] = float(np.max(abs_audio))

    low_energy = abs_audio < low_energy_threshold
    metrics["low_energy_ratio"] = float(np.mean(low_energy)) if low_energy.size else 0.0
    if low_energy.size:
        longest = 0
        current = 0
        for is_low in low_energy:
            if bool(is_low):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        metrics["longest_low_energy_run"] = float(longest / sample_rate)

    # Sample-level low-energy checks catch digital silence, but TTS dropouts are
    # often low RMS over 1-4 second windows. Track that separately for listening
    # quality gates.
    win = max(1, int(sample_rate * 0.05))
    hop = max(1, int(sample_rate * 0.025))
    if finite.size >= win:
        n_frames = (finite.size - win) // hop + 1
        frame_rms = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop
            block = finite[s : s + win]
            frame_rms[i] = math.sqrt(float(np.mean(np.square(block))))
        rms_low = frame_rms < pause_rms_threshold
        metrics["rms_low_energy_ratio"] = float(np.mean(rms_low)) if rms_low.size else 0.0
        if rms_low.size:
            longest_frames = 0
            current_frames = 0
            for is_low in rms_low:
                if bool(is_low):
                    current_frames += 1
                    longest_frames = max(longest_frames, current_frames)
                else:
                    current_frames = 0
            if longest_frames:
                metrics["longest_rms_low_energy_run"] = float(
                    ((longest_frames - 1) * hop + win) / sample_rate
                )
            else:
                metrics["longest_rms_low_energy_run"] = 0.0

    if finite.size > 1:
        jumps = np.abs(np.diff(finite))
        metrics["jump_count_gt_0_3"] = int(np.count_nonzero(jumps > 0.3))
        metrics["jump_count_gt_0_5"] = int(np.count_nonzero(jumps > 0.5))
        metrics["max_jump"] = float(np.max(jumps))

    return metrics


def audio_quality_issues(
    metrics: dict[str, Any],
    *,
    label: str,
    low_rms_threshold: float = 1e-4,
    low_peak_threshold: float = 1e-4,
    high_peak_threshold: float = 0.95,
    max_low_energy_ratio: float = 0.98,
    max_low_energy_run_seconds: float = 2.0,
    max_rms_low_energy_ratio: float = 0.45,
    max_rms_low_energy_run_seconds: float = 1.2,
) -> list[str]:
    """Return hard quality-gate issues for a metrics payload."""
    issues: list[str] = []
    duration = float(metrics.get("duration", 0.0) or 0.0)
    rms = float(metrics.get("rms", 0.0) or 0.0)
    peak = float(metrics.get("peak", 0.0) or 0.0)
    low_ratio = float(metrics.get("low_energy_ratio", 0.0) or 0.0)
    low_run = float(metrics.get("longest_low_energy_run", 0.0) or 0.0)
    rms_low_ratio = float(metrics.get("rms_low_energy_ratio", 0.0) or 0.0)
    rms_low_run = float(metrics.get("longest_rms_low_energy_run", 0.0) or 0.0)
    jump_05 = int(metrics.get("jump_count_gt_0_5", 0) or 0)
    max_jump = float(metrics.get("max_jump", 0.0) or 0.0)

    if not bool(metrics.get("finite", True)):
        issues.append(f"{label}: contains NaN or Inf samples")
    if duration <= 0:
        issues.append(f"{label}: empty audio")
    if rms < low_rms_threshold:
        issues.append(f"{label}: very low RMS ({rms:.2e})")
    if peak < low_peak_threshold:
        issues.append(f"{label}: very low peak ({peak:.2e})")
    if peak > 1.0:
        issues.append(f"{label}: peak exceeds full scale ({peak:.3f})")
    elif peak >= high_peak_threshold:
        issues.append(f"{label}: peak is near clipping ({peak:.3f})")
    if duration >= 0.5 and low_ratio >= max_low_energy_ratio:
        issues.append(f"{label}: mostly low-energy audio ({low_ratio:.1%})")
    if low_run >= max_low_energy_run_seconds:
        issues.append(f"{label}: long low-energy run ({low_run:.1f}s)")
    if duration >= 1.0 and rms_low_run >= max_rms_low_energy_run_seconds:
        issues.append(f"{label}: long audible pause/dropout ({rms_low_run:.1f}s)")
    if duration >= 3.0 and rms_low_ratio >= max_rms_low_energy_ratio:
        issues.append(f"{label}: too much audible pause/dropout ({rms_low_ratio:.1%})")
    if jump_05 > 0:
        issues.append(f"{label}: {jump_05} sample jumps > 0.5 (max {max_jump:.3f})")
    return issues


def write_quality_report(path: str | os.PathLike[str], report: dict[str, Any]) -> str:
    """Write a quality report JSON and return its path."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return str(report_path)


def safe_remove(path: Optional[str]) -> bool:
    """Delete a file silently if it exists.

    Returns True when a file was removed, False if there was nothing to do
    or the path could not be deleted.
    """
    if not path or not os.path.exists(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


# Backwards-compatible alias for older imports.
_safe_remove = safe_remove


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

def sanitise_tts_text(text: str) -> str:
    """Remove invisible/control characters that can destabilise TTS tokenization."""
    cleaned_chars: list[str] = []
    for char in text.replace("\r\n", "\n").replace("\r", "\n"):
        if char == "\u00a0":
            cleaned_chars.append(" ")
            continue
        category = unicodedata.category(char)
        if category == "Cf":
            continue
        if category.startswith("C") and char not in {"\n", "\t"}:
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


def split_text(text: str, max_chars: int = 80) -> list[str]:
    """Split text into chunks suitable for TTS generation.

    Prefer major sentence boundaries. Commas are only used as fallback when a
    single sentence is too long, because comma-ending TTS chunks make cloned
    voices restart mid-thought and sound discontinuous.
    """
    text = sanitise_tts_text(text)
    max_chars = max(1, int(max_chars or 1))
    if len(text.strip()) <= max_chars:
        return [text.strip()] if text.strip() else []

    # Insert split markers after major sentence terminators only.
    text = re.sub(r"([。！？!?；;])\s*", r"\1\n", text)

    raw_sentences: list[str] = []
    for sentence in (s.strip() for s in text.split("\n") if s.strip()):
        if len(sentence) <= max_chars:
            raw_sentences.append(sentence)
            continue

        # Fallback for a very long single sentence: split on comma-sized pauses
        # while packing phrases up to max_chars.
        phrases = re.split(r"([，,])", sentence)
        current_phrase = ""
        for index in range(0, len(phrases), 2):
            phrase = phrases[index]
            comma = phrases[index + 1] if index + 1 < len(phrases) else ""
            token = phrase + comma
            if not token:
                continue
            if current_phrase and len(current_phrase) + len(token) > max_chars:
                raw_sentences.append(current_phrase.strip())
                current_phrase = token
            else:
                current_phrase += token
        if current_phrase.strip():
            raw_sentences.append(current_phrase.strip())

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
    target_lufs: float = -23.0,
    peak_ceiling: float = 0.95,
) -> np.ndarray:
    """Apply EBU R128 loudness normalisation to *audio*.

    Default target: **-23 LUFS** (broadcast-style conservative speech level).
    Falls back silently to the original audio if pyloudnorm is not installed.

    Why -23 instead of -16 or -20: the underlying Qwen3-TTS model often emits
    clean but low-level speech. Aggressive loudness gain can turn ordinary
    consonant slopes and small model roughness into audible noise, then the
    post-loudness de-click stage may over-edit the speech. -23 LUFS keeps the
    output usable while preserving more headroom for voice cloning artefacts.
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
        _log.warning("Loudness normalisation skipped: %s", e)
        return audio


def limit_audio_peak(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Ensure peak stays below *ceiling* using **soft** (tanh) limiting.

    Hard scaling (the previous behaviour) preserves the relative dynamic
    range of model glitches — a 0.5-amplitude token-level artefact stays at
    0.5 even after the whole file is scaled to peak=0.95. Soft limiting
    asymptotically compresses large transients toward *ceiling* without the
    hard step that produces audible clicks.

    For samples already under the ceiling the response is essentially linear
    (tanh(x) ≈ x for |x| < 0.5), so the perceived loudness is unchanged.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return arr

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    peak = float(np.max(np.abs(finite)))
    if peak <= 0 or peak <= ceiling * 0.95:
        # Under the soft-knee threshold — pass through unchanged.
        return arr

    # Soft clip: tanh-saturate, then rescale so the peak maps to *ceiling*.
    saturated = np.tanh(arr / ceiling) * ceiling
    # Re-apply the ceiling for very large values (tanh → ±ceiling, never over).
    return saturated


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Subtract the mean so the waveform is centred around 0.

    Tiny in practice (DC offset is typically < 1e-3) but cumulative across
    long concatenated audio it can drift a few percent — this prevents that.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return arr
    return arr - float(np.mean(arr))


def smooth_sample_jumps(
    audio: np.ndarray,
    *,
    threshold: float = 0.25,
    radius: int = 2,
    passes: int = 5,
    max_repairs: int = 5000,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Interpolate over isolated sample-level jumps.

    TTS models can emit one-sample discontinuities that become audible clicks
    after loudness gain. This repair is intentionally local: each detected
    boundary is replaced by a short linear ramp between nearby stable samples.
    It does not smooth ordinary speech contours or long noisy regions.
    """
    arr = np.asarray(audio, dtype=np.float32)
    stats: dict[str, int | float] = {
        "threshold": float(threshold),
        "radius": int(radius),
        "passes": int(passes),
        "detected_jumps": 0,
        "repaired_jumps": 0,
    }
    if arr.size < 3 or threshold <= 0 or radius < 1 or passes < 1:
        return arr, stats

    if arr.ndim > 1:
        repaired = arr.copy()
        detected = 0
        repaired_count = 0
        for channel in range(repaired.shape[1]):
            repaired_channel, channel_stats = smooth_sample_jumps(
                repaired[:, channel],
                threshold=threshold,
                radius=radius,
                passes=passes,
                max_repairs=max_repairs,
            )
            repaired[:, channel] = repaired_channel
            detected += int(channel_stats.get("detected_jumps", 0) or 0)
            repaired_count += int(channel_stats.get("repaired_jumps", 0) or 0)
        stats["detected_jumps"] = detected
        stats["repaired_jumps"] = repaired_count
        return repaired, stats

    repaired = arr.copy()
    detected_total = 0
    repaired_count = 0
    for _ in range(passes):
        jumps = np.flatnonzero(np.abs(np.diff(repaired)) > threshold)
        if jumps.size == 0:
            break
        detected_total += int(jumps.size)
        if repaired_count >= max_repairs:
            break

        groups: list[tuple[int, int]] = []
        group_start = int(jumps[0])
        group_end = int(jumps[0])
        merge_distance = radius * 2 + 2
        for raw_index in jumps[1:]:
            jump_index = int(raw_index)
            if jump_index <= group_end + merge_distance:
                group_end = jump_index
            else:
                groups.append((group_start, group_end))
                group_start = group_end = jump_index
        groups.append((group_start, group_end))

        for first_jump, last_jump in groups:
            if repaired_count >= max_repairs:
                break
            start = max(0, first_jump - radius)
            end = min(repaired.shape[0] - 1, last_jump + radius + 1)
            if end <= start + 1:
                continue
            left = repaired[start]
            right = repaired[end]
            repaired[start : end + 1] = np.linspace(left, right, end - start + 1, dtype=np.float32)
            repaired_count += 1

    stats["detected_jumps"] = detected_total
    stats["repaired_jumps"] = repaired_count
    return repaired, stats


def _effective_silence_threshold(rms: np.ndarray, threshold: float) -> float:
    """Return a conservative silence threshold for one generated chunk."""
    finite = np.asarray(rms, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(threshold)

    active_rms = float(np.percentile(finite, 90))
    # A fixed 5e-3 floor can classify quiet TTS speech as silence. Cap the
    # threshold at 10% of the chunk's active energy, while keeping a tiny floor
    # so digital zero and near-zero padding are still removable.
    return max(1e-5, min(float(threshold), active_rms * 0.10))


def _region_is_silent(audio: np.ndarray, start: int, end: int, threshold: float) -> bool:
    """Return True if audio[start:end] has RMS below *threshold*.

    Supports negative *start* / *end* indices (Python-style) so callers
    can address tail/head regions without recomputing absolute offsets,
    e.g. ``_region_is_silent(buf, -N, len(buf), thr)`` means "the last N
    samples".  Empty / zero-length regions are considered silent.
    """
    n = len(audio)
    if n == 0 or start == end:
        return True
    a = start % n if start < 0 else max(0, min(start, n))
    b = end % n if end < 0 else max(0, min(end, n))
    if a >= b:
        return True
    region = np.asarray(audio[a:b], dtype=np.float32)
    if region.size == 0:
        return True
    rms = float(np.sqrt(np.mean(region ** 2)))
    return rms < threshold


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    threshold: float = 5e-3,
    min_silence_ms: float = 300.0,
    keep_silence_ms: float = 200.0,
    max_trim_ratio: float | None = 0.35,
    head_trim_ms: float = 0.0,
    tail_trim_ms: float = 0.0,
    pad_ms: float = 0.0,
    force_trim_speech: bool = False,
) -> np.ndarray:
    """Trim long silence padding from a TTS chunk.

    The Qwen3-TTS and VoxCPM2 backends both emit noticeable silence padding
    at the *start* of each chunk (model warm-up, ref-audio echo, etc.) and
    occasional long silence valleys *inside* long chunks.  When those chunks
    are concatenated the result sounds "broken" / "with obvious gaps".

    Strategy:

    1. Compute a short-time RMS envelope (50 ms window, 25 ms hop).
    2. Locate every silence run where RMS is below a conservative per-chunk
       threshold for >= *min_silence_ms*.
    3. Inside the run, keep only the first *keep_silence_ms* worth of
       silence so listeners still hear a natural pause instead of an
       immediate hard cut.
    4. If the requested silence removal would discard more than
       *max_trim_ratio* of the chunk, keep the original chunk instead. This
       protects low-energy TTS speech from being mistaken for silence.
    5. If *head_trim_ms* > 0, trim that many ms from the start. By
       default this only fires when the head is actually silence
       (RMS < *threshold*); set *force_trim_speech* to True to always
       cut regardless of content. The "skip when speech" default fixes
       the "一句话没读完就开始读下一句" regression where the old
       unconditional 80ms tail trim clipped sentence endings that had
       trailing decay/breath/consonants still above the silence floor.
    6. If *tail_trim_ms* > 0, same smart-trim rule as the head.
    7. If *pad_ms* > 0, prepend / append that much zero padding so
       downstream crossfade has clean zero-crossings to latch onto.

    Default values are tuned for Qwen3-TTS at 24 kHz with a -23 LUFS target.
    VoxCPM2 (48 kHz) works with the same defaults — the function is purely
    envelope-based and sample-rate agnostic.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0 or sample_rate <= 0:
        return arr

    sr = int(sample_rate)
    min_silence_samples = max(1, int(sr * min_silence_ms / 1000.0))
    keep_silence_samples = max(0, int(sr * keep_silence_ms / 1000.0))
    head_force = max(0, int(sr * head_trim_ms / 1000.0))
    tail_force = max(0, int(sr * tail_trim_ms / 1000.0))
    pad_samples = max(0, int(sr * pad_ms / 1000.0))

    # ---- 1. Envelope ---------------------------------------------------------
    win_ms = 50
    hop_ms = 25
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(arr) < win:
        # Too short to envelope — only apply head/tail smart trim.
        # Same rule as the main path: skip the cut if the candidate region
        # contains speech, so short chunks don't get clipped either.
        out = np.asarray(arr, dtype=np.float32)
        effective_threshold = _effective_silence_threshold(
            np.array([float(np.sqrt(np.mean(out ** 2)))], dtype=np.float32),
            threshold,
        )
        if tail_force and len(out) > tail_force:
            if force_trim_speech or _region_is_silent(out, -tail_force, len(out), effective_threshold):
                out = out[:-tail_force]
        if head_force and len(out) > head_force:
            if force_trim_speech or _region_is_silent(out, 0, head_force, effective_threshold):
                out = out[head_force:]
        if pad_samples:
            out = np.concatenate([
                np.zeros(pad_samples, dtype=np.float32),
                out,
                np.zeros(pad_samples, dtype=np.float32),
            ])
        return out

    n_frames = (len(arr) - win) // hop + 1
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        rms[i] = float(np.sqrt(np.mean(arr[s : s + win] ** 2)))

    effective_threshold = _effective_silence_threshold(rms, threshold)
    is_silent = rms < effective_threshold

    # ---- 2. Find silence runs (in *frame* units, then convert to samples) ----
    # For trimming, we work directly on sample indices: a frame at index i
    # covers samples [i*hop, i*hop + win).  Silence "starts" at i*hop and
    # "ends" at j*hop — i.e. *just before* the first non-silent frame's
    # window, NOT at the end of that window. The old j*hop+win formula ate
    # up to one window (50 ms) of trailing speech at the end of every chunk
    # and caused the "一句话没读完就开始读下一句" regression — a 100 ms
    # tail of breath/final-consonant at amplitude 0.5 was getting included
    # in the silence run, then dropped by keep_segments because cursor had
    # already advanced past the speech.
    runs: list[tuple[int, int]] = []
    i = 0
    n = n_frames
    while i < n:
        if is_silent[i]:
            j = i
            while j < n and is_silent[j]:
                j += 1
            sample_start = i * hop
            sample_end = j * hop  # was: min(len(arr), j * hop + win)
            if (sample_end - sample_start) >= min_silence_samples:
                runs.append((sample_start, sample_end))
            i = j
        else:
            i += 1

    # ---- 3. Build a list of "kept" regions ----------------------------------
    # Start with full audio, then for each long silence we keep only the
    # *keep_silence_samples* of silence at its start.
    keep_segments: list[tuple[int, int]] = []
    cursor = 0
    for s, e in runs:
        if s > cursor:
            keep_segments.append((cursor, s))
        # Keep a short natural pause
        keep_end = min(e, s + keep_silence_samples)
        keep_segments.append((s, keep_end))
        cursor = e
    if cursor < len(arr):
        keep_segments.append((cursor, len(arr)))

    if not keep_segments:
        # Entire audio is silent — return as-is (crossfade_concat will handle)
        out = arr
    else:
        out = np.concatenate([arr[a:b] for a, b in keep_segments])

    if (
        max_trim_ratio is not None
        and 0 <= max_trim_ratio < 1
        and arr.size > 0
        and out.size < arr.size * (1.0 - max_trim_ratio)
    ):
        _log.warning(
            "trim_silence: skipped aggressive trim that would remove %.1f%% "
            "of a chunk (limit %.1f%%).",
            100.0 * (1.0 - (out.size / arr.size)),
            100.0 * max_trim_ratio,
        )
        out = arr

    # ---- 4. Force head/tail trim (smart: only trim if tail/head is silence) -----
    # The old behaviour unconditionally cut head_force/tail_force from the ends,
    # which silently clipped sentence endings whenever the TTS chunk did not have
    # a clean silent tail (i.e. most chunks — the model often trails off with a
    # 50–150 ms decay/breath/consonant that is well above the silence floor).
    # The new default checks the RMS of the candidate region and bails out (with
    # a warning) if it contains actual speech, so "一句话没读完" stops happening.
    if head_force or tail_force:
        # Tails first so the head-trim index stays valid after we shorten.
        if tail_force and len(out) > tail_force:
            if force_trim_speech or _region_is_silent(out, -tail_force, len(out), effective_threshold):
                out = out[:-tail_force]
            else:
                _log.warning(
                    "trim_silence: skipped tail trim of %.0f ms — tail RMS "
                    "exceeds threshold %.4f (sentence-ending speech detected, "
                    "preserving the last %.0f ms of audio).",
                    tail_trim_ms, effective_threshold, tail_trim_ms,
                )
        if head_force and len(out) > head_force:
            if force_trim_speech or _region_is_silent(out, 0, head_force, effective_threshold):
                out = out[head_force:]
            else:
                _log.warning(
                    "trim_silence: skipped head trim of %.0f ms — head RMS "
                    "exceeds threshold %.4f (cold-start speech detected, "
                    "preserving the first %.0f ms of audio).",
                    head_trim_ms, effective_threshold, head_trim_ms,
                )
        if out.size == 0:
            # The whole chunk was trimmed to silence by the smart cuts. This is
            # only possible if the entire chunk is silence — keep it as-is so
            # downstream crossfade still sees a (silent) chunk.
            _log.warning(
                "trim_silence: chunk became empty after smart head/tail trim; "
                "restoring the original audio."
            )
            out = np.asarray(audio, dtype=np.float32)

    # ---- 5. Optional pad with silence so crossfade has a clean zero -------
    if pad_samples and out.size:
        out = np.concatenate([
            np.zeros(pad_samples, dtype=np.float32),
            out,
            np.zeros(pad_samples, dtype=np.float32),
        ])

    return out


def crossfade_concat(
    segments: list,
    fade_ms: float = 30.0,
    sample_rate: int = SAMPLE_RATE,
    min_fade_samples: int = 8,
) -> np.ndarray:
    """Concatenate *segments* with a linear crossfade of *fade_ms* between each.

    A hard ``np.concatenate`` leaves a sample-to-sample step wherever the
    first sample of segment N+1 does not equal the last sample of segment N.
    A short linear crossfade smooths that boundary. Default 30 ms is below
    the human perceptual threshold for speech (~40 ms) and avoids overlap
    that would smear consonants.

    When a segment is shorter than the requested fade window, the window is
    shrunk to ``min(len(out), len(s), min_fade_samples)`` so we always
    perform *some* crossfade. A hard fallback is reserved for the case
    where either side is below ``min_fade_samples`` (e.g. empty / sub-ms
    segments) — at that point no linear ramp can fit and a click is
    unavoidable, but it would have happened anyway at ``np.concatenate``.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return np.asarray(segments[0], dtype=np.float32)

    requested_fade = max(1, int(sample_rate * fade_ms / 1000.0))
    out = np.asarray(segments[0], dtype=np.float32).copy()

    for seg in segments[1:]:
        s = np.asarray(seg, dtype=np.float32)
        fade = min(requested_fade, len(out), len(s))
        if fade < min_fade_samples:
            # Truly too short to ramp — fall back to hard concat. This
            # preserves audio but may click; callers should pre-pad to
            # avoid hitting this path.
            out = np.concatenate([out, s])
            continue
        # Linear ramp: 1→0 over fade samples
        ramp_down = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        ramp_up = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        head = out[:-fade]
        tail = out[-fade:]
        s_head = s[:fade]
        s_rest = s[fade:]
        mixed = tail * ramp_down + s_head * ramp_up
        out = np.concatenate([head, mixed, s_rest])

    return out


# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------

def make_output_filename(text_snippet: str, ext: str = "wav") -> str:
    """Return a timestamped, filesystem-safe filename.

    Truncation is codepoint-safe (Python 3 str slices on code points, not bytes).
    """
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
) -> bool:
    """Persist generation progress to disk.

    *audio_segments* is a list of absolute paths to per-chunk WAV files.
    Returns True on success, False on failure (already logged).
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
        parent = os.path.dirname(checkpoint_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, checkpoint_path)
        return True
    except OSError as e:
        _log.warning("Checkpoint save failed: %s", e)
        safe_remove(tmp)
        return False


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
