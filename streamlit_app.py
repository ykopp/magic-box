import hashlib
import html
import importlib
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    os.execv(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]],
    )

from article_extractor import extract_article_from_url
from tts_backends import get_backend_model_choices, get_default_model_ref
from utils import ReferenceAudioError, make_output_filename, safe_remove, split_text
from voice_controls import get_preset, optimize_podcast_rhythm, preset_names
from voice_profiles import delete_profile, list_profiles, save_profile, slugify_profile_id, text_fingerprint

# Configure logging once; guard against Streamlit bare-mode module reloads.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
RUNTIME_DIR = APP_DIR / "runtime" / "streamlit"
MAX_PREVIEW_CHUNKS = 8
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_RUNTIME_BYTES = 500 * 1024 * 1024  # 500 MB of kept uploads
PROFILE_AUDIO_TYPES = ["wav", "mp3", "m4a", "aac", "flac", "ogg"]
REFERENCE_RETRY_PROMPT = "请重新提供 3-20 秒干净单人声，并填写逐字匹配文本。"
QWEN_DISPLAY_TO_GENERATION_SPEED = 0.90
DEFAULT_SEGMENT_SECONDS = 120.0


st.set_page_config(
    page_title="Magic Box",
    page_icon="🎛️",
    layout="wide",
)


_CSS_PATH = APP_DIR / "app_styles.css"
if _CSS_PATH.exists():
    st.markdown(
        f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
else:  # pragma: no cover - safety net for frozen bundles
    st.markdown(
        """
        <style>
    :root {
        --bg: #08090b;
        --panel: #111318;
        --panel-2: #181b22;
        --panel-3: #20242c;
        --line: #343944;
        --line-hot: #ff8a2a;
        --text: #f4f1ea;
        --muted: #b8afa2;
        --dim: #7e7a73;
        --accent: #ff8a2a;
        --accent-2: #ffb45f;
        --danger: #ff5f4f;
    }
    .stApp {
        background:
            linear-gradient(rgba(255, 138, 42, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 138, 42, 0.025) 1px, transparent 1px),
            radial-gradient(circle at 80% 0%, rgba(255, 138, 42, 0.10), transparent 28rem),
            var(--bg);
        background-size: 28px 28px, 28px 28px, auto, auto;
        color: var(--text);
    }
    [data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu,
    footer,
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(255, 138, 42, 0.08), rgba(8, 9, 11, 0) 18rem),
            #0d0f13;
        border-right: 1px solid #3c3328;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: #f4ede3 !important;
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4, p, span, label {
        letter-spacing: 0;
        color: var(--text);
    }
    h2, h3 {
        color: var(--text);
        text-shadow: 0 0 16px rgba(255, 138, 42, 0.22);
    }
    label, [data-testid="stWidgetLabel"] p {
        color: #d8d0c4 !important;
        font-weight: 620;
    }
    .stCaptionContainer, .stCaptionContainer p, div[data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #171a20, #101216);
        border: 1px solid #3f3428;
        border-left: 3px solid var(--accent);
        border-radius: 2px;
        padding: 0.75rem 0.85rem;
        box-shadow: inset 0 0 0 1px rgba(255, 138, 42, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #fff6e8;
    }
    div[data-testid="stExpander"], div[data-testid="stForm"], div[data-testid="stTabs"] {
        border-color: var(--line);
    }
    div[data-testid="stExpander"] {
        background: rgba(17, 19, 24, 0.72);
        border-radius: 2px;
    }
    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] details > summary,
    div[data-testid="stExpander"] div[role="button"] {
        background: #15181f !important;
        color: #fff0da !important;
        border-color: #5a3b23 !important;
        border-radius: 2px !important;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p {
        color: #fff0da !important;
    }
    div[data-testid="stExpander"] svg {
        color: #ff9a3d !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea,
    input {
        background: #111318 !important;
        color: #fff3df !important;
        border-color: #44372a !important;
        border-radius: 2px !important;
    }
    textarea::placeholder,
    input::placeholder {
        color: #8e8377 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] svg {
        color: var(--accent);
    }
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background: #111318 !important;
        border: 1px solid #70451f !important;
        border-radius: 2px !important;
        color: #fff3df !important;
    }
    ul[role="listbox"] li,
    div[role="option"] {
        background: #111318 !important;
        color: #fff3df !important;
    }
    ul[role="listbox"] li:hover,
    div[role="option"]:hover {
        background: #23180f !important;
        color: #ffffff !important;
    }
    div[data-baseweb="tab-list"] {
        border-bottom: 1px solid #3b3127;
        gap: 0.25rem;
    }
    button[data-baseweb="tab"] {
        color: #8f877d !important;
        background: #0d0f13 !important;
        border: 1px solid transparent !important;
        border-radius: 0 !important;
        padding: 0.65rem 0.95rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ff9a3d !important;
        border-color: #6a4526 !important;
        border-bottom-color: #ff8a2a !important;
        background: linear-gradient(180deg, rgba(255, 138, 42, 0.14), rgba(255, 138, 42, 0.03)) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: #ff8a2a !important;
        border-color: #ffbd75 !important;
        box-shadow: 0 0 12px rgba(255, 138, 42, 0.45);
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] div {
        color: #ffb45f !important;
    }
    .console-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        background:
            linear-gradient(90deg, rgba(255, 138, 42, 0.16), rgba(255, 138, 42, 0.03) 22rem, transparent),
            linear-gradient(180deg, #171a20 0%, #111318 100%);
        border: 1px solid #4a3422;
        border-left: 4px solid var(--accent);
        border-radius: 2px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28), inset 0 0 0 1px rgba(255, 138, 42, 0.04);
    }
    .console-title {
        font-size: 1.5rem;
        font-weight: 760;
        line-height: 1.2;
        margin: 0 0 0.25rem 0;
        color: #fff6e8;
        text-transform: uppercase;
    }
    .console-subtitle {
        color: #cbbdad;
        font-size: 0.92rem;
        margin: 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        min-height: 30px;
        padding: 0 0.7rem;
        border: 1px solid rgba(255, 138, 42, 0.62);
        border-radius: 2px;
        color: #ffd5a3;
        background: rgba(255, 138, 42, 0.10);
        font-size: 0.86rem;
        white-space: nowrap;
    }
    .section-note {
        color: #c3b9aa;
        font-size: 0.88rem;
        margin-top: -0.35rem;
        margin-bottom: 0.55rem;
    }
    .file-path {
        color: #ffcf95;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.86rem;
    }
    .stButton > button,
    .stDownloadButton > button {
        background: #171a20 !important;
        color: #ffe8ca !important;
        border: 1px solid #70451f !important;
        border-radius: 2px !important;
        box-shadow: inset 0 -2px 0 rgba(255, 138, 42, 0.16);
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #ff8a2a !important;
        color: #ffffff !important;
        background: #23180f !important;
    }
    button[kind="primary"] {
        background: linear-gradient(180deg, #ff9b3d, #d86112) !important;
        border: 1px solid #ffbd75 !important;
        color: #120c06 !important;
        font-weight: 760 !important;
    }
    [data-testid="stAlert"] {
        background: #14171d;
        color: #fff3df;
        border: 1px solid #5a3b23;
        border-radius: 2px;
    }
    [data-testid="stFileUploader"] section {
        background: #111318 !important;
        border: 1px dashed #7b542f !important;
        border-radius: 2px !important;
    }
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section span {
        color: #d8cbb9 !important;
    }
    @media (max-width: 720px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 0.85rem;
        }
        .console-header {
            flex-direction: column;
            gap: 0.65rem;
            padding: 0.9rem 1rem;
        }
        .console-title {
            font-size: 1.18rem;
            line-height: 1.25;
        }
        .console-subtitle {
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .status-pill {
            align-self: flex-start;
            min-height: 28px;
        }
        button[data-baseweb="tab"] {
            padding: 0.55rem 0.75rem !important;
        }
        div[data-testid="stMetric"] {
            padding: 0.65rem 0.7rem;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (APP_DIR / "voices" / "profiles").mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str, fallback: str) -> str:
    stem = Path(name or fallback).name
    stem = re.sub(r"[^\w.\- ]+", "_", stem, flags=re.UNICODE).strip()
    return stem or fallback


def _enforce_runtime_quota() -> None:
    """Remove oldest uploads in ``RUNTIME_DIR`` when total size exceeds quota."""
    if not RUNTIME_DIR.exists():
        return
    files = [path for path in RUNTIME_DIR.iterdir() if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    if total <= MAX_RUNTIME_BYTES:
        return

    files.sort(key=lambda path: path.stat().st_mtime)
    for path in files:
        if total <= MAX_RUNTIME_BYTES:
            break
        try:
            total -= path.stat().st_size
            path.unlink()
        except OSError:
            continue


def _persist_upload(uploaded_file, prefix: str) -> Optional[Path]:
    if uploaded_file is None:
        return None
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error(f"文件过大（{uploaded_file.size / 1024 / 1024:.0f} MB），上限为 50 MB。")
        return None
    _ensure_dirs()
    _enforce_runtime_quota()
    filename = _safe_filename(uploaded_file.name, f"{prefix}.bin")
    target = RUNTIME_DIR / f"{prefix}_{uuid4().hex}_{filename}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _persist_upload_once(uploaded_file, prefix: str, state_key: str) -> Optional[Path]:
    if uploaded_file is None:
        st.session_state.pop(state_key, None)
        return None
    raw_bytes = uploaded_file.getvalue()
    upload_id = f"{uploaded_file.name}:{uploaded_file.size}:{hashlib.md5(raw_bytes).hexdigest()}"
    cached = st.session_state.get(state_key)
    if isinstance(cached, dict) and cached.get("upload_id") == upload_id and cached.get("path"):
        cached_path = Path(cached["path"])
        if cached_path.exists():
            return cached_path

    path = _persist_upload(uploaded_file, prefix)
    if path is not None:
        st.session_state[state_key] = {"upload_id": upload_id, "path": str(path)}
    return path


def _read_uploaded_text(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def _normalise_output_path(text: str, requested_name: str, output_format: str = "mp3") -> Path:
    requested_name = requested_name.strip()
    ext = "mp3" if output_format == "mp3" else "wav"
    name = _safe_filename(requested_name, f"podcast.{ext}")
    if Path(name).suffix.lower() != f".{ext}":
        name = f"{Path(name).stem}.{ext}"
    return OUTPUT_DIR / name


def _auto_output_path(text: str, output_format: str = "mp3") -> Path:
    text_key = text.strip()
    ext = "mp3" if output_format == "mp3" else "wav"
    auto_state = st.session_state.get("auto_output")
    if (
        auto_state
        and auto_state.get("text") == text_key
        and auto_state.get("format") == output_format
        and auto_state.get("path")
    ):
        return Path(auto_state["path"])

    path = OUTPUT_DIR / make_output_filename(text_key[:20], ext=ext)
    st.session_state["auto_output"] = {
        "text": text_key,
        "format": output_format,
        "path": str(path),
    }
    return path


def _resolve_output_path(text: str, requested_name: str, output_format: str = "mp3") -> Path:
    if requested_name.strip():
        return _normalise_output_path(text, requested_name, output_format)
    return _auto_output_path(text, output_format)


def _format_model_choice(choice) -> str:
    return getattr(choice, "label", str(choice))


def _filter_model_choices_for_clone(backend: str, choices: list) -> list:
    if backend != "qwen":
        return choices
    return [
        choice
        for choice in choices
        if "base" in f"{getattr(choice, 'label', '')} {getattr(choice, 'value', '')}".lower()
    ]


def _get_default_choice_index(choices: list, backend: str) -> int:
    if not choices:
        return 0
    default_ref = get_default_model_ref(backend)
    for index, choice in enumerate(choices):
        if choice.value == default_ref:
            return index
    return 0


def _effective_generation_speed(backend: str, display_speed: float) -> float:
    if backend == "qwen":
        return round(float(display_speed) * QWEN_DISPLAY_TO_GENERATION_SPEED, 3)
    return float(display_speed)


def _build_generation_command(
    *,
    text_path: Path,
    ref_audio_path: Path,
    ref_text: str,
    output_path: Path,
    backend: str,
    model_ref: str,
    speed: float,
    temperature: float,
    chunk_max_chars: int,
    checkpoint_path: Path,
    metadata_path: Path,
    output_format: str,
    normalise: bool,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(APP_DIR / "podcast_generator.py"),
        "--file",
        str(text_path),
        "--ref-audio",
        str(ref_audio_path),
        "--ref-text",
        ref_text,
        "--output",
        str(output_path),
        "--backend",
        backend,
        "--model",
        model_ref,
        "--speed",
        str(speed),
        "--temperature",
        str(temperature),
        "--chunk-max-chars",
        str(chunk_max_chars),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-metadata-file",
        str(metadata_path),
        "--format",
        output_format,
    ]
    if not normalise:
        command.append("--no-normalise")
    if resume:
        command.append("--resume")
    return command


def _run_generation_subprocess(
    *,
    target_text: str,
    ref_audio_path: Path,
    ref_text: str,
    output_path: Path,
    backend: str,
    model_ref: str,
    speed: float,
    temperature: float,
    chunk_max_chars: int,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    output_format: str,
    normalise: bool,
    resume: bool,
    log_callback=None,
    process_callback=None,
    heartbeat_callback=None,
    heartbeat_interval: float = 1.0,
) -> Path:
    """Run MLX generation in a child process so native crashes cannot kill Streamlit."""

    _ensure_dirs()
    _enforce_runtime_quota()
    request_id = uuid4().hex
    text_path = RUNTIME_DIR / f"generation_{request_id}.txt"
    metadata_path = RUNTIME_DIR / f"generation_{request_id}.metadata.json"
    text_path.write_text(target_text, encoding="utf-8")
    metadata_path.write_text(json.dumps(checkpoint_metadata or {}, ensure_ascii=False, indent=2), encoding="utf-8")

    command = _build_generation_command(
        text_path=text_path,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        output_path=output_path,
        backend=backend,
        model_ref=model_ref,
        speed=speed,
        temperature=temperature,
        chunk_max_chars=chunk_max_chars,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        output_format=output_format,
        normalise=normalise,
        resume=resume,
    )
    output_lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process_callback is not None:
        process_callback(process.pid)
    assert process.stdout is not None
    line_queue: queue.Queue[str | None] = queue.Queue()

    def read_stdout() -> None:
        try:
            for line in process.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stdout_thread.start()

    start_time = time.monotonic()
    last_heartbeat = 0.0
    stdout_done = False
    returncode: int | None = None
    while True:
        try:
            line = line_queue.get(timeout=0.2)
        except queue.Empty:
            line = None
        else:
            if line is None:
                stdout_done = True
            else:
                clean_line = line.rstrip()
                if clean_line:
                    output_lines.append(clean_line)
                    if log_callback is not None:
                        log_callback(clean_line)

        now = time.monotonic()
        if heartbeat_callback is not None and now - last_heartbeat >= heartbeat_interval:
            heartbeat_callback(now - start_time, output_lines[-1] if output_lines else "")
            last_heartbeat = now

        returncode = process.poll()
        if returncode is not None and stdout_done:
            break

    stdout_thread.join(timeout=1)
    if returncode is None:
        returncode = process.wait()
    if returncode != 0:
        tail = "\n".join(output_lines[-12:])
        message = _generation_failure_message(output_path, output_path)
        if "未找到可解析的质量报告" in message and tail:
            message = f"生成子进程异常退出（退出码 {returncode}）。最近日志:\n{tail}"
        else:
            message = f"{message}（子进程退出码 {returncode}）"
        raise RuntimeError(message)

    if output_path.exists():
        return output_path
    report_path, report = _load_quality_report(output_path, output_path)
    if isinstance(report, dict) and isinstance(report.get("result_path"), str):
        result_path = Path(report["result_path"])
        if result_path.exists():
            return result_path
    raise RuntimeError(_generation_failure_message(output_path, output_path))


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "估算中"
    if seconds < 60:
        return f"约 {max(1, int(seconds))} 秒"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"约 {minutes} 分钟"
    hours, minutes = divmod(minutes, 60)
    return f"约 {hours} 小时 {minutes} 分钟"


def _estimated_generation_progress(
    *,
    completed: int,
    total: int,
    elapsed_seconds: float,
    current_segment_elapsed: float,
    completed_segment_durations: list[float],
) -> dict[str, float | None]:
    if total <= 0:
        return {"ratio": 0.0, "remaining_seconds": None}
    if completed >= total:
        return {"ratio": 1.0, "remaining_seconds": 0.0}

    if completed_segment_durations:
        segment_seconds = max(10.0, sum(completed_segment_durations) / len(completed_segment_durations))
    elif completed > 0:
        segment_seconds = max(10.0, elapsed_seconds / completed)
    else:
        segment_seconds = DEFAULT_SEGMENT_SECONDS

    current_fraction = min(0.95, max(0.0, current_segment_elapsed / segment_seconds))
    estimated_units = completed + current_fraction
    ratio = min(0.99, max(0.0, estimated_units / total))
    remaining_units = max(0.0, total - estimated_units)
    return {
        "ratio": ratio,
        "remaining_seconds": remaining_units * segment_seconds,
    }


def _generation_progress_snapshot(output_path: Path, checkpoint_path: Path, total_chunks: int) -> dict[str, Any]:
    completed: set[int] = set()
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checkpoint = {}
        for item in checkpoint.get("completed", []):
            if isinstance(item, int):
                completed.add(item)

    segment_dir = output_path.parent / f".{output_path.stem}_segments"
    segment_count = 0
    if segment_dir.exists():
        for path in segment_dir.glob("_seg_*.wav"):
            segment_count += 1
            match = re.search(r"_seg_(\d+)\.wav$", path.name)
            if match:
                completed.add(int(match.group(1)))

    completed_count = min(len(completed), total_chunks)
    return {
        "completed": completed_count,
        "total": total_chunks,
        "current": min(completed_count + 1, total_chunks) if total_chunks else 0,
        "segment_count": segment_count,
        "segment_dir": segment_dir,
    }


def _parse_process_etime(etime: str) -> float | None:
    try:
        days = 0
        value = etime.strip()
        if "-" in value:
            day_text, value = value.split("-", 1)
            days = int(day_text)
        parts = [int(part) for part in value.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            hours = 0
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            return None
        return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)
    except (TypeError, ValueError):
        return None


def _generation_process_snapshot(output_path: Path) -> dict[str, Any] | None:
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,stat=,pcpu=,pmem=,etime=,command="],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output_token = str(output_path)
    for line in output.splitlines():
        if "podcast_generator.py" not in line or output_token not in line:
            continue
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, stat, cpu, mem, etime, command = parts
        try:
            pid_value = int(pid)
        except ValueError:
            continue
        return {
            "pid": pid_value,
            "ppid": ppid,
            "stat": stat,
            "cpu": cpu,
            "mem": mem,
            "etime": etime,
            "elapsed_seconds": _parse_process_etime(etime),
            "command": command,
        }
    return None


def _checkpoint_saved_at(checkpoint_path: Path) -> float | None:
    try:
        return checkpoint_path.stat().st_mtime
    except OSError:
        return None


@st.fragment(run_every=2)
def _show_generation_monitor(output_path: Path, checkpoint_path: Path, total_chunks: int) -> None:
    if total_chunks <= 0:
        return

    snapshot = _generation_progress_snapshot(output_path, checkpoint_path, total_chunks)
    process = _generation_process_snapshot(output_path)
    completed = int(snapshot["completed"])
    total = int(snapshot["total"])
    has_artifacts = checkpoint_path.exists() or Path(snapshot["segment_dir"]).exists()
    if process is None and not has_artifacts:
        return
    if process is None and output_path.exists() and completed >= total:
        return

    process_elapsed = process.get("elapsed_seconds") if process else None
    elapsed_seconds = float(process_elapsed or 0.0)
    now = time.time()
    saved_at = _checkpoint_saved_at(checkpoint_path)
    if process_elapsed is not None and saved_at is not None:
        current_segment_elapsed = max(0.0, now - saved_at)
    elif process_elapsed is not None:
        current_segment_elapsed = elapsed_seconds
    else:
        current_segment_elapsed = 0.0

    if completed > 0 and elapsed_seconds > 0:
        average_segment_seconds = max(10.0, elapsed_seconds / completed)
        completed_durations = [average_segment_seconds] * completed
    else:
        completed_durations = []

    estimate = _estimated_generation_progress(
        completed=completed,
        total=total,
        elapsed_seconds=elapsed_seconds,
        current_segment_elapsed=current_segment_elapsed,
        completed_segment_durations=completed_durations,
    )
    ratio = float(estimate["ratio"] or 0.0)

    st.subheader("当前生成进度")
    if process is not None:
        st.info(
            " | ".join(
                [
                    f"生成子进程 PID: {process['pid']}",
                    f"运行时间: {process['etime']}",
                    f"CPU: {process['cpu']}%",
                    f"预计剩余: {_format_eta(estimate['remaining_seconds'])}",
                ]
            )
        )
    else:
        st.warning("没有检测到正在运行的生成子进程；已保留断点，重新点击生成会从断点继续。")

    if completed >= total:
        label = f"已完成 {completed}/{total} 段，正在收尾或拼接输出"
    else:
        label = f"估算进度 {ratio * 100:.0f}% | 正在第 {snapshot['current']}/{total} 段，已完成 {completed}/{total} 段"
    st.progress(ratio, text=label)
    st.caption(
        " | ".join(
            [
                f"segment 文件: {snapshot['segment_count']}",
                f"断点: {checkpoint_path.relative_to(APP_DIR) if checkpoint_path.exists() else '暂无'}",
                f"输出: {output_path.relative_to(APP_DIR)}",
            ]
        )
    )


def _recent_outputs(limit: int = 10) -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    wavs = sorted(
        [*OUTPUT_DIR.glob("*.wav"), *OUTPUT_DIR.glob("*.mp3")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return wavs[:limit]


def _download_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _quality_report_candidates(final_path: Path, requested_output_path: Path) -> list[Path]:
    """Return likely quality-report locations in priority order."""

    candidates = [
        final_path.parent / "quality_report.json",
        requested_output_path.parent / "quality_report.json",
        final_path.with_suffix(".quality_report.json"),
        requested_output_path.with_suffix(".quality_report.json"),
        final_path.with_name(f"{final_path.stem}_quality_report.json"),
        requested_output_path.with_name(f"{requested_output_path.stem}_quality_report.json"),
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def _load_quality_report(final_path: Path, requested_output_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for candidate in _quality_report_candidates(final_path, requested_output_path):
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return candidate, None
        if isinstance(data, dict):
            return candidate, data
        return candidate, None
    return None, None


def _quality_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    if isinstance(summary, dict):
        source = summary
    else:
        source = report
    chunks = report.get("chunks")
    chunk_reports = chunks if isinstance(chunks, list) else []
    failed_chunk_count = sum(
        1
        for chunk in chunk_reports
        if isinstance(chunk, dict) and bool(chunk.get("issues") or chunk.get("errors") or chunk.get("warnings"))
    )

    return {
        "total_segments": source.get("total_segments")
        or source.get("segment_count")
        or source.get("total")
        or source.get("chunks_total")
        or (len(chunk_reports) if chunk_reports else None),
        "passed_segments": source.get("passed_segments")
        or source.get("ok_segments")
        or source.get("passed")
        or source.get("chunks_passed")
        or (len(chunk_reports) - failed_chunk_count if chunk_reports else None),
        "failed_segments": source.get("failed_segments_count")
        or source.get("failed_count")
        or source.get("chunks_failed")
        or (failed_chunk_count if chunk_reports else None),
        "status": source.get("status") or report.get("status"),
    }


def _quality_report_failed_segments(report: dict[str, Any]) -> list[int]:
    explicit_segments = report.get("failed_segments")
    if isinstance(explicit_segments, list):
        return sorted(
            {number for item in explicit_segments if (number := _coerce_segment_number(item)) is not None}
        )

    explicit_indices = report.get("failed_segment_indices") or report.get("failed_indices")
    if isinstance(explicit_indices, list):
        return sorted(
            {number for item in explicit_indices if (number := _coerce_segment_number(item, zero_based=True)) is not None}
        )

    failed: set[int] = set()
    segments = report.get("segments") or report.get("segment_reports") or report.get("chunks")
    if isinstance(segments, list):
        for index, segment in enumerate(segments, start=1):
            if not isinstance(segment, dict):
                continue
            status = str(segment.get("status") or segment.get("result") or "").lower()
            passed = segment.get("passed")
            warnings = segment.get("warnings") or segment.get("errors") or segment.get("issues")
            if status in {"failed", "fail", "error"} or passed is False or warnings:
                segment_number = (
                    _coerce_segment_number(segment.get("segment"))
                    or _coerce_segment_number(segment.get("segment_index"), zero_based=True)
                    or _coerce_segment_number(segment.get("index"), zero_based=True)
                    or index
                )
                failed.add(segment_number)
    return sorted(failed)


def _quality_report_issue_lines(report: dict[str, Any], limit: int = 5) -> list[str]:
    issues = report.get("issues")
    lines: list[str] = []
    if isinstance(issues, list):
        lines.extend(str(issue) for issue in issues if str(issue).strip())

    final = report.get("final")
    if isinstance(final, dict):
        for key in (
            "output_diagnostics",
            "written_wav_diagnostics",
            "decoded_mp3_diagnostics",
            "output_warnings",
            "written_wav_warnings",
            "decoded_mp3_warnings",
        ):
            values = final.get(key)
            if isinstance(values, list):
                lines.extend(str(value) for value in values if str(value).strip())

    deduped = list(dict.fromkeys(lines))
    return deduped[:limit]


def _coerce_segment_number(value: Any, *, zero_based: bool = False) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value + 1 if zero_based else value
    if isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
        return number + 1 if zero_based else number
    return None


def _segments_dir_from_report(report: dict[str, Any] | None, final_path: Path, requested_output_path: Path) -> Path:
    if report:
        for key in ("segments_dir", "segments_folder", "segment_dir", "segment_folder"):
            raw_path = report.get(key)
            if isinstance(raw_path, str) and raw_path.strip():
                path = Path(raw_path).expanduser()
                return path if path.is_absolute() else final_path.parent / path
    return requested_output_path.parent / f".{requested_output_path.stem}_segments"


def _show_quality_report(final_path: Path, requested_output_path: Path) -> None:
    report_path, report = _load_quality_report(final_path, requested_output_path)
    segments_dir = _segments_dir_from_report(report, final_path, requested_output_path)

    st.subheader("质量报告")
    if report_path is None:
        st.info("未找到 quality_report.json；生成器可能未写出质量报告。")
    elif report is None:
        st.warning(f"质量报告无法解析: {report_path.relative_to(APP_DIR) if report_path.is_relative_to(APP_DIR) else report_path}")
    else:
        summary = _quality_report_summary(report)
        summary_cols = st.columns(4)
        summary_cols[0].metric("总段数", summary["total_segments"] or "-")
        summary_cols[1].metric("通过段", summary["passed_segments"] or "-")
        summary_cols[2].metric("失败段", summary["failed_segments"] or len(_quality_report_failed_segments(report)) or 0)
        summary_cols[3].metric("状态", summary["status"] or "-")

        failed_segments = _quality_report_failed_segments(report)
        if failed_segments:
            st.error("失败段编号: " + ", ".join(str(index) for index in failed_segments))
        else:
            st.success("质量报告未标记失败段。")
        issue_lines = _quality_report_issue_lines(report)
        if issue_lines:
            st.warning("质量诊断: " + "；".join(issue_lines))
        st.markdown(
            f'<span class="file-path">报告: {report_path.relative_to(APP_DIR) if report_path.is_relative_to(APP_DIR) else report_path}</span>',
            unsafe_allow_html=True,
        )

    if segments_dir.exists():
        segment_count = len([path for path in segments_dir.iterdir() if path.is_file()])
        st.markdown(
            f'<span class="file-path">segments 文件夹: {segments_dir.relative_to(APP_DIR) if segments_dir.is_relative_to(APP_DIR) else segments_dir} ({segment_count} 个文件)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"segments 文件夹未找到: {segments_dir}")


def _has_quality_artifacts(final_path: Path, requested_output_path: Path) -> bool:
    report_path, report = _load_quality_report(final_path, requested_output_path)
    segments_dir = _segments_dir_from_report(report, final_path, requested_output_path)
    return report_path is not None or segments_dir.exists()


def _generation_failure_message(final_path: Path, requested_output_path: Path) -> str:
    report_path, report = _load_quality_report(final_path, requested_output_path)
    if not report:
        return "生成器未返回输出路径，且未找到可解析的质量报告。"
    issues = _quality_report_issue_lines(report, limit=3)
    if issues:
        return "生成器未返回输出路径: " + "；".join(issues)
    status = report.get("status") or "unknown"
    if report_path is not None:
        path_label = report_path.relative_to(APP_DIR) if report_path.is_relative_to(APP_DIR) else report_path
        return f"生成器未返回输出路径，质量报告状态为 {status}: {path_label}"
    return f"生成器未返回输出路径，质量报告状态为 {status}。"


def _estimate_duration_label(characters: int) -> str:
    if characters <= 0:
        return "未生成估算"
    minutes = max(1, round(characters / 260))
    if minutes < 60:
        return f"约 {minutes} 分钟"
    hours, remain = divmod(minutes, 60)
    return f"约 {hours} 小时 {remain} 分钟"


def _set_voice_defaults_from_preset(preset_name: str) -> None:
    preset = get_preset(preset_name)
    st.session_state["voice_speed"] = preset.speed
    st.session_state["voice_temperature"] = preset.temperature
    st.session_state["voice_chunk_max_chars"] = preset.chunk_max_chars
    st.session_state["active_voice_preset"] = preset.name


def _ensure_voice_state() -> None:
    default_name = "自然播客"
    if "selected_voice_preset" not in st.session_state:
        st.session_state["selected_voice_preset"] = default_name
    if "active_voice_preset" not in st.session_state:
        _set_voice_defaults_from_preset(st.session_state["selected_voice_preset"])


def _apply_rhythm_preview() -> None:
    preview = st.session_state.get("rhythm_preview", "")
    if preview:
        st.session_state["target_text"] = preview
        st.session_state["rhythm_preview"] = ""


def _show_output_library() -> None:
    st.subheader("输出库")
    recent_files = _recent_outputs()
    if not recent_files:
        st.caption("outputs/ 里还没有音频文件。")
        return

    for path in recent_files:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{path.name} - {modified} - {stat.st_size / 1024 / 1024:.1f} MB"):
            mime = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
            st.audio(str(path), format=mime)
            st.download_button(
                f"下载 {path.suffix[1:].upper()}",
                data=_download_bytes(path),
                file_name=path.name,
                mime=mime,
                key=f"download_{path.name}_{stat.st_mtime}",
            )


def _format_profile(profile) -> str:
    suffix = " / 需要文本" if profile.needs_transcript else ""
    source = "内置" if profile.built_in else "已保存"
    return f"{profile.display_name} ({source}{suffix})"


def _profile_ref_text_key(profile_id: str) -> str:
    return f"voice_profile_ref_text_{profile_id}"


def _profile_has_saved_reference(profile: Any) -> bool:
    return bool(
        profile is not None
        and not getattr(profile, "built_in", False)
        and getattr(profile, "can_generate", False)
        and str(getattr(profile, "ref_text", "") or "").strip()
    )


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _profile_root(profile: Any) -> Path | None:
    ref_audio_path = getattr(profile, "ref_audio_path", None)
    if ref_audio_path is None:
        return None
    return Path(ref_audio_path).parent


def _read_profile_metadata(profile: Any) -> dict[str, Any]:
    root = _profile_root(profile)
    if root is None:
        return {}
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _is_legacy_profile(profile: Any) -> bool:
    if profile is None:
        return False
    root = _profile_root(profile)
    if root is None:
        return True

    metadata = _read_profile_metadata(profile)
    schema = metadata.get("schema_version") or metadata.get("profile_schema_version")
    if str(schema).strip().lower() in {"1", "1.0", "v1"}:
        return True

    clean_audio = root / "reference_clean.wav"
    clean_text = root / "reference_clean.txt"
    return not (clean_audio.exists() and clean_text.exists())


def _normalise_reference_quality_report(raw_report: Any) -> dict[str, Any]:
    report = raw_report if isinstance(raw_report, dict) else {}
    status = str(report.get("status") or "").strip().lower()
    passed = report.get("passed")
    if passed is None:
        passed = report.get("ok")
    if passed is None and status:
        passed = status in {"pass", "passed", "ok", "success"}
    passed = bool(passed)

    metrics_source = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    metric_keys = (
        "duration_seconds",
        "sample_rate",
        "rms",
        "peak",
        "clipped_ratio",
        "active_ratio",
        "snr_db",
        "noise_floor_db",
    )
    metrics = {key: metrics_source[key] for key in metric_keys if key in metrics_source}
    issues = (
        _as_text_list(report.get("issues"))
        + _as_text_list(report.get("errors"))
        + _as_text_list(report.get("warnings"))
        + _as_text_list(report.get("reason"))
        + _as_text_list(report.get("rejection_reason"))
    )
    recommendations = _as_text_list(report.get("recommendations"))
    if not passed and not recommendations:
        recommendations = [REFERENCE_RETRY_PROMPT]

    return {
        "status": "passed" if passed else (status or "failed"),
        "passed": passed,
        "metrics": metrics,
        "issues": issues,
        "recommendations": recommendations,
        "raw": report,
    }


def _load_profile_reference_report(profile: Any) -> dict[str, Any] | None:
    root = _profile_root(profile)
    if root is None:
        return None
    for path in (root / "reference_quality.json", root / "reference_audit.json"):
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict):
            return report
    return None


def _call_optional_reference_audit(audio_path: Path, ref_text: str, profile: Any = None) -> dict[str, Any] | None:
    candidates = (
        ("voice_profiles", "audit_profile_reference"),
        ("voice_profiles", "audit_reference_profile"),
        ("utils", "audit_reference_audio"),
        ("utils", "audit_reference_audio_file"),
        ("utils", "validate_reference_audio_file"),
        ("utils", "audit_reference_audio_quality"),
    )
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        audit_func = getattr(module, function_name, None)
        if audit_func is None:
            continue
        call_attempts = []
        if profile is not None:
            call_attempts.extend(
                [
                    lambda: audit_func(profile=profile, ref_text=ref_text, app_dir=APP_DIR),
                    lambda: audit_func(profile, ref_text),
                ]
            )
        call_attempts.extend(
            [
                lambda: audit_func(audio_path=audio_path, ref_text=ref_text),
                lambda: audit_func(str(audio_path), ref_text),
                lambda: audit_func(audio_path),
            ]
        )
        for attempt in call_attempts:
            try:
                result = attempt()
            except TypeError:
                continue
            except Exception as exc:
                return {"ok": False, "status": "failed", "issues": [str(exc)], "recommendations": [REFERENCE_RETRY_PROMPT]}
            if isinstance(result, tuple) and result:
                result = result[-1]
            if isinstance(result, dict):
                return result
    return None


def _reference_quality_status(
    *,
    ref_audio_path: Path | None,
    ref_text: str,
    profile: Any = None,
    profile_mode: str = "",
) -> dict[str, Any]:
    if profile is not None and _is_legacy_profile(profile):
        return _normalise_reference_quality_report(
            {
                "ok": False,
                "status": "failed",
                "issues": ["旧 Profile 缺少 reference_clean.wav / reference_clean.txt，或仍是 schema v1。"],
                "recommendations": ["请在“管理声音 Profile”中用干净单人声重新保存这个 Profile。", REFERENCE_RETRY_PROMPT],
            }
        )
    if not ref_text.strip():
        return _normalise_reference_quality_report(
            {
                "ok": False,
                "status": "failed",
                "issues": ["参考文本缺失。"],
                "recommendations": [REFERENCE_RETRY_PROMPT],
            }
        )
    if ref_audio_path is None:
        label = "请上传临时源声音。" if profile_mode == "临时上传源声音" else "请选择可用 Profile。"
        return _normalise_reference_quality_report(
            {"ok": False, "status": "missing", "issues": [label], "recommendations": [REFERENCE_RETRY_PROMPT]}
        )
    if not ref_audio_path.exists():
        return _normalise_reference_quality_report(
            {
                "ok": False,
                "status": "failed",
                "issues": ["参考音频文件不存在。"],
                "recommendations": [REFERENCE_RETRY_PROMPT],
            }
        )

    profile_report = _load_profile_reference_report(profile) if profile is not None else None
    if profile_report is not None:
        return _normalise_reference_quality_report(profile_report)

    audit_report = _call_optional_reference_audit(ref_audio_path, ref_text, profile=profile)
    if audit_report is not None:
        return _normalise_reference_quality_report(audit_report)

    return _normalise_reference_quality_report(
        {
            "ok": False,
            "status": "pending",
            "issues": ["参考音频还没有可用审核结果。"],
            "recommendations": [REFERENCE_RETRY_PROMPT],
        }
    )


def _can_generate_with_reference_quality(status: dict[str, Any]) -> bool:
    return bool(status.get("passed"))


def _show_reference_quality_card(status: dict[str, Any]) -> None:
    passed = _can_generate_with_reference_quality(status)
    st.subheader("参考音频质量")
    if passed:
        st.success("通过：参考音频可用于克隆。")
    else:
        st.error("失败：参考音频暂不可用于生成。")

    metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
    if metrics:
        metric_items = list(metrics.items())[:4]
        cols = st.columns(len(metric_items))
        for col, (key, value) in zip(cols, metric_items):
            col.metric(key, value)
    else:
        st.caption("暂无可展示的审核指标。")

    issues = _as_text_list(status.get("issues"))
    recommendations = _as_text_list(status.get("recommendations"))
    if issues:
        st.markdown("**Issues**")
        for item in issues:
            st.write(f"- {item}")
    if recommendations:
        st.markdown("**Recommendations**")
        for item in recommendations:
            st.write(f"- {item}")


def _sync_uploaded_text(uploaded_file) -> None:
    if uploaded_file is None:
        return
    # Use content hash instead of name:size to detect genuinely changed uploads
    raw_bytes = uploaded_file.getvalue()
    upload_id = f"{uploaded_file.name}:{hashlib.md5(raw_bytes).hexdigest()}"
    if st.session_state.get("last_text_upload_id") == upload_id:
        return
    persisted_path = _persist_upload(uploaded_file, "script_text")
    st.session_state["target_text"] = _read_uploaded_text(uploaded_file)
    st.session_state["target_text_path"] = str(persisted_path) if persisted_path else ""
    st.session_state["last_text_upload_id"] = upload_id
    st.rerun()


def main() -> None:
    _ensure_dirs()
    _ensure_voice_state()
    if "target_text" not in st.session_state:
        st.session_state["target_text"] = ""
    if "extracted_article" not in st.session_state:
        st.session_state["extracted_article"] = None
    if "rhythm_preview" not in st.session_state:
        st.session_state["rhythm_preview"] = ""

    with st.sidebar:
        st.header("制作参数")
        backend = st.selectbox("TTS 后端", ["qwen", "voxcpm"], index=0)
        model_choices = _filter_model_choices_for_clone(backend, get_backend_model_choices(backend))

        if model_choices:
            model_choice = st.selectbox(
                "模型",
                model_choices,
                index=_get_default_choice_index(model_choices, backend),
                format_func=_format_model_choice,
            )
            model_ref = model_choice.value
        else:
            model_ref = ""
            st.error(f"No local/available model choices found for backend '{backend}'.")

        names = preset_names()
        preset_index = names.index(st.session_state["selected_voice_preset"]) if st.session_state["selected_voice_preset"] in names else 0
        selected_preset = st.selectbox(
            "表达预设",
            names,
            index=preset_index,
        )
        if selected_preset != st.session_state.get("selected_voice_preset"):
            st.session_state["selected_voice_preset"] = selected_preset
        if selected_preset != st.session_state.get("active_voice_preset"):
            _set_voice_defaults_from_preset(selected_preset)

        preset = get_preset(selected_preset)
        st.caption(preset.explanation)

        with st.expander("高级微调", expanded=True):
            speed = st.slider(
                "语速",
                min_value=0.50,
                max_value=2.00,
                step=0.05,
                key="voice_speed",
                help="控制相对参考音频的朗读速度。Qwen clone 下 1.00 会先按参考音频语速折算成模型 speed；过高更容易吞字、断裂或音色漂移。",
            )
            effective_speed = _effective_generation_speed(backend, speed)
            if backend == "qwen":
                st.caption(f"显示语速 {speed:.2f}；实际传入生成器 {effective_speed:.2f}。")
            if backend == "qwen" and speed > 1.05:
                st.warning("Qwen clone 语速高于 1.05 时坏段风险会上升。")
            temperature = st.slider(
                "表达随机性",
                min_value=0.10,
                max_value=2.00,
                step=0.05,
                key="voice_temperature",
                help="控制声音表现的变化幅度。Qwen clone 建议不超过 0.85，过高更容易出现杂音、跑调或不稳定段。",
            )
            if backend == "qwen" and temperature > 0.85:
                st.warning("Qwen clone 表达随机性高于 0.85 时音质不稳定风险会上升。")
            st.session_state["voice_chunk_max_chars"] = min(
                600,
                max(80, int(st.session_state.get("voice_chunk_max_chars", preset.chunk_max_chars))),
            )
            chunk_max_chars = st.slider(
                "单段字符上限",
                min_value=80,
                max_value=600,
                step=20,
                key="voice_chunk_max_chars",
                help="控制切分粒度。短文建议 400-600 保持连贯；长文可调低方便定位坏段。",
            )
            if backend == "qwen" and chunk_max_chars < 200:
                st.warning("单段字符上限过低会频繁重新起声，短文容易听起来断断续续。")

        normalise = st.toggle("响度标准化", value=True)
        resume = st.toggle(
            "从断点继续",
            value=True,
            help=(
                "长文生成会按 chunk 保存进度；中断后保持同一个输出文件名，"
                "打开这里会跳过已完成片段继续生成。成功完成后断点会自动清理。"
            ),
        )
        output_name = st.text_input(
            "输出文件名",
            value="",
            placeholder="留空自动命名，默认 .mp3；也可输入 my_episode.wav",
        )
        output_format_label = st.selectbox(
            "输出格式",
            ["WAV", "MP3", "WAV + MP3"],
            index=1,
        )
        output_format = {
            "WAV": "wav",
            "MP3": "mp3",
            "WAV + MP3": "both",
        }[output_format_label]

        if backend == "voxcpm":
            st.info("VoxCPM2 MLX: 48kHz 高保真, 30 语言, Voice Design + Clone。语速/表达随机性将被忽略。")

    target_text_for_header = st.session_state.get("target_text", "").strip()
    header_chunks = split_text(target_text_for_header, max_chars=chunk_max_chars) if target_text_for_header else []
    st.markdown(
        f"""
        <div class="console-header">
            <div>
                <div class="console-title">播客制作控制台</div>
                <p class="console-subtitle">长文抽取、节奏整理、声音微调与断点生成</p>
            </div>
            <div class="status-pill">{"稿件已就绪" if target_text_for_header else "等待稿件"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if target_text_for_header:
        metrics = st.columns(4)
        metrics[0].metric("字符数", f"{len(target_text_for_header):,}")
        metrics[1].metric("切分段数", len(header_chunks))
        metrics[2].metric("预估成片", _estimate_duration_label(len(target_text_for_header)))
        metrics[3].metric("单段上限", chunk_max_chars)

    st.subheader("稿件来源")
    source_tabs = st.tabs(["手动输入", "TXT 导入", "URL 抽取"])

    with source_tabs[1]:
        st.markdown('<p class="section-note">上传 UTF-8 或 GB18030 编码的 .txt 文件。</p>', unsafe_allow_html=True)
        text_file = st.file_uploader("导入 .txt 稿件", type=["txt"])
        _sync_uploaded_text(text_file)
        if st.session_state.get("target_text_path"):
            st.markdown(
                f'<span class="file-path">已保存: {Path(st.session_state["target_text_path"]).relative_to(APP_DIR)}</span>',
                unsafe_allow_html=True,
            )

    with source_tabs[2]:
        st.markdown('<p class="section-note">抽取公开文章页面正文，自动去除广告、订阅、分享和相关阅读等网页杂质。</p>', unsafe_allow_html=True)
        article_url = st.text_input("文章 URL", placeholder="https://example.com/article")
        col_extract, col_timeout = st.columns([1, 3])
        with col_extract:
            extract_clicked = st.button("抽取正文", use_container_width=True)
        with col_timeout:
            st.caption("抽取后会整理成更适合播客朗读的稿件。")

        if extract_clicked:
            try:
                with st.spinner("正在抽取正文..."):
                    st.session_state["extracted_article"] = extract_article_from_url(article_url.strip(), timeout=15)
                st.success("正文抽取完成。")
            except Exception as exc:
                st.session_state["extracted_article"] = None
                st.error(f"抽取失败：{exc}")

        article = st.session_state.get("extracted_article")
        if article is not None:
            article_preview = article.podcast_text[:1200]
            st.markdown(f"**标题：** {article.title}")
            st.markdown(f"**来源：** {article.source_url}")
            article_metrics = st.columns(2)
            article_metrics[0].metric("原文正文", f"{len(article.text):,}")
            article_metrics[1].metric("播客稿", f"{len(article.podcast_text):,}")
            st.text_area("播客稿预览", article_preview, height=240, disabled=True)
            if st.button("使用播客稿", type="primary", use_container_width=True):
                st.session_state["target_text"] = article.podcast_text
                st.session_state["target_text_path"] = ""
                st.session_state["rhythm_preview"] = ""
                st.rerun()

    with source_tabs[0]:
        st.markdown('<p class="section-note">粘贴或直接编辑正式播客稿件。</p>', unsafe_allow_html=True)
        st.text_area(
            "稿件正文",
            key="target_text",
            height=300,
            placeholder="在这里粘贴长文播客稿件，或从 TXT / URL 抽取后使用为稿件。",
        )

    target_text = st.session_state.get("target_text", "").strip()

    st.subheader("播客节奏优化")
    st.markdown('<p class="section-note">生成一个只调整断句和段落节奏的预览，确认后再应用。</p>', unsafe_allow_html=True)
    rhythm_cols = st.columns([1, 3])
    with rhythm_cols[0]:
        if st.button("生成优化预览", disabled=not bool(target_text), use_container_width=True):
            st.session_state["rhythm_preview"] = optimize_podcast_rhythm(target_text)
    with rhythm_cols[1]:
        st.caption("不会自动覆盖稿件；适合在生成前改善长句、逗号串和段落呼吸。")

    if st.session_state.get("rhythm_preview"):
        st.text_area("优化预览", st.session_state["rhythm_preview"], height=220, disabled=True)
        st.button("应用到稿件", use_container_width=True, on_click=_apply_rhythm_preview)

    st.subheader("克隆声音 Profile")
    profiles = list_profiles(APP_DIR)
    profile_mode = st.radio(
        "源声音来源",
        ["选择已保存/内置 Profile", "临时上传源声音"],
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_profile = None
    ref_audio_upload = None
    ref_audio_path: Optional[Path] = None
    ref_text = ""
    profile_metadata = {}

    if profile_mode == "选择已保存/内置 Profile":
        if not profiles:
            st.warning("还没有可用声音 Profile。请临时上传源声音，或在“管理声音 Profile”里保存自己的声音样本。")
        else:
            selected_profile_id = st.session_state.get("selected_voice_profile_id", profiles[0].id)
            profile_ids = [profile.id for profile in profiles]
            profile_index = profile_ids.index(selected_profile_id) if selected_profile_id in profile_ids else 0
            selected_profile = st.selectbox(
                "选择克隆声音",
                profiles,
                index=profile_index,
                format_func=_format_profile,
            )
            if selected_profile.id != st.session_state.get("selected_voice_profile_id"):
                st.session_state["selected_voice_profile_id"] = selected_profile.id
                st.rerun()

            ref_audio_path = selected_profile.ref_audio_path
            ref_text_key = _profile_ref_text_key(selected_profile.id)
            saved_ref_text = (selected_profile.ref_text or "").strip()
            if saved_ref_text and not str(st.session_state.get(ref_text_key, "")).strip():
                st.session_state[ref_text_key] = saved_ref_text

            if _profile_has_saved_reference(selected_profile):
                ref_text = saved_ref_text
                st.success("已使用 Profile 中保存的参考文本；本次生成不需要重新上传或填写。")
                with st.expander("查看 / 临时覆盖参考文本", expanded=False):
                    override_ref_text = st.text_area(
                        "参考音频文本",
                        key=ref_text_key,
                        height=140,
                        help="默认使用 Profile 保存的文本。只有在你明确修改这里时，本次生成才会临时覆盖；不会改写 Profile。",
                    ).strip()
                    if override_ref_text and override_ref_text != saved_ref_text:
                        ref_text = override_ref_text
                        st.warning("本次生成使用临时覆盖文本；Profile 本身不会被修改。")
                    elif not override_ref_text:
                        st.warning("覆盖文本为空，已继续使用 Profile 保存的参考文本。")
            else:
                ref_text = st.text_area(
                    "参考音频文本",
                    key=ref_text_key,
                    height=140,
                    help="必须与源声音里实际朗读的文字一致；缺失时不能生成。",
                ).strip()
            profile_metadata = {
                "profile": {
                    "voice_profile_id": selected_profile.id,
                    "voice_profile_fingerprint": f"{selected_profile.fingerprint}:{text_fingerprint(ref_text)}",
                },
            }

            info_cols = st.columns([1, 1])
            with info_cols[0]:
                st.markdown(
                    f'<span class="file-path">源声音: {ref_audio_path.relative_to(APP_DIR)}</span>',
                    unsafe_allow_html=True,
                )
                if selected_profile.description:
                    st.caption(selected_profile.description)
            with info_cols[1]:
                if ref_audio_path.exists():
                    st.audio(str(ref_audio_path))
                else:
                    st.error("源声音文件不存在。")

    else:
        left, right = st.columns([1, 1])
        with left:
            ref_audio_upload = st.file_uploader("临时源声音", type=PROFILE_AUDIO_TYPES)
            ref_audio_path = _persist_upload_once(ref_audio_upload, "ref_audio", "temporary_ref_audio_upload")
            if ref_audio_path is not None:
                st.audio(str(ref_audio_path))
        with right:
            ref_text = st.text_area(
                "参考音频文本",
                value="",
                height=140,
                help="需要与临时源声音实际朗读内容一致。",
            ).strip()
        profile_metadata = {"profile": {"voice_profile_id": "temporary_upload"}}

    reference_quality_status = _reference_quality_status(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        profile=selected_profile,
        profile_mode=profile_mode,
    )
    _show_reference_quality_card(reference_quality_status)
    if not _can_generate_with_reference_quality(reference_quality_status):
        st.warning(REFERENCE_RETRY_PROMPT)

    with st.expander("管理声音 Profile", expanded=False):
        st.caption("保存后的 profile 会放在 voices/profiles/，默认不会提交到 Git。")
        profile_name = st.text_input("Profile 名称", placeholder="例如 Thomas / Vivian / 旁白男声")
        profile_description = st.text_input("说明", placeholder="例如：Thomas 开场声音，情绪更热情")
        profile_preset = st.selectbox("默认表达预设", preset_names(), index=preset_names().index("自然播客"))
        profile_audio = st.file_uploader("保存源声音", type=PROFILE_AUDIO_TYPES, key="save_profile_audio")
        profile_transcript = st.text_area(
            "源声音对应文本",
            height=110,
            placeholder="逐字填写源声音里说的话。",
        ).strip()
        manage_cols = st.columns([1, 1])
        with manage_cols[0]:
            if st.button("保存为 Profile", use_container_width=True):
                if not profile_name.strip():
                    st.error("请填写 Profile 名称。")
                elif profile_audio is None:
                    st.error("请上传源声音。")
                elif not profile_transcript:
                    st.error("请填写源声音对应文本。")
                else:
                    saved_audio = _persist_upload(profile_audio, "profile_audio")
                    if saved_audio is None:
                        st.error("源声音保存失败。")
                    else:
                        try:
                            saved_profile = save_profile(
                                APP_DIR,
                                display_name=profile_name,
                                audio_source=saved_audio,
                                transcript=profile_transcript,
                                description=profile_description,
                                default_preset=profile_preset,
                                profile_id=slugify_profile_id(profile_name),
                            )
                        except ReferenceAudioError as exc:
                            st.error(f"参考音频质量不合格，未保存 Profile: {exc}")
                            st.warning(REFERENCE_RETRY_PROMPT)
                        except Exception as exc:
                            st.error(f"保存 Profile 失败: {exc}")
                        else:
                            st.session_state["selected_voice_profile_id"] = saved_profile.id
                            st.success(f"已保存 Profile: {saved_profile.display_name}")
                            st.rerun()
        with manage_cols[1]:
            user_profiles = [profile for profile in profiles if not profile.built_in]
            if user_profiles:
                delete_target = st.selectbox("删除已保存 Profile", user_profiles, format_func=_format_profile)
                if st.button("删除 Profile", use_container_width=True):
                    delete_profile(APP_DIR, delete_target.id)
                    st.session_state.pop("selected_voice_profile_id", None)
                    st.success(f"已删除 Profile: {delete_target.display_name}")
                    st.rerun()
            else:
                st.caption("还没有用户保存的 profile。")

    chunks = split_text(target_text, max_chars=chunk_max_chars) if target_text else []
    output_path = _resolve_output_path(target_text, output_name, output_format)
    checkpoint_path = output_path.with_suffix(".ckpt")

    st.subheader("切分与输出")
    stats = st.columns(4)
    stats[0].metric("字符数", f"{len(target_text):,}")
    stats[1].metric("切分段数", len(chunks))
    stats[2].metric("预估成片", _estimate_duration_label(len(target_text)))
    stats[3].metric("断点", "继续" if resume else "重建")
    st.markdown(f'<span class="file-path">输出路径: {output_path.relative_to(APP_DIR)}</span>', unsafe_allow_html=True)
    _show_generation_monitor(output_path, checkpoint_path, len(chunks))

    if chunks:
        preview_text = "\n\n".join(
            f"{index}. {chunk}" for index, chunk in enumerate(chunks[:MAX_PREVIEW_CHUNKS], start=1)
        )
        if len(chunks) > MAX_PREVIEW_CHUNKS:
            preview_text += f"\n\n... 还有 {len(chunks) - MAX_PREVIEW_CHUNKS} 段"
        st.text_area("切分预览", preview_text, height=180, disabled=True)
    else:
        st.info("输入、导入或抽取稿件后可预览切分结果。")

    reference_can_generate = _can_generate_with_reference_quality(reference_quality_status)
    active_generation = _generation_process_snapshot(output_path)
    generate = st.button(
        "生成播客音频",
        type="primary",
        use_container_width=True,
        disabled=not reference_can_generate or active_generation is not None,
    )
    if active_generation is not None:
        st.caption(f"当前输出正在生成中，已禁用重复启动。PID: {active_generation['pid']}")
    if generate:
        errors = []
        if not target_text:
            errors.append("请先准备稿件正文。")
        if not model_ref:
            errors.append("请先选择可用模型。")
        if not ref_text:
            errors.append(REFERENCE_RETRY_PROMPT)

        if ref_audio_path is None:
            errors.append(REFERENCE_RETRY_PROMPT)
        elif not ref_audio_path.exists():
            errors.append("参考音频文件不存在。")
        if selected_profile is not None and selected_profile.needs_transcript and not ref_text:
            errors.append(REFERENCE_RETRY_PROMPT)
        if not reference_can_generate:
            errors.append(REFERENCE_RETRY_PROMPT)

        if profile_mode == "临时上传源声音" and ref_audio_upload is not None and ref_audio_path is not None:
            profile_metadata = {
                "profile": {
                    "voice_profile_id": "temporary_upload",
                    "voice_profile_fingerprint": f"temporary_upload:{ref_audio_upload.name}:{ref_audio_upload.size}:{text_fingerprint(ref_text)}",
                },
            }

        if errors:
            for error in dict.fromkeys(errors):
                st.error(error)
        else:
            assert ref_audio_path is not None
            if not resume and checkpoint_path.exists():
                safe_remove(str(checkpoint_path))

            try:
                with st.status("在隔离子进程中生成音频...", expanded=True) as status:
                    st.write(f"后端: {backend}")
                    st.write(f"模型: {model_ref}")
                    if backend == "qwen":
                        st.write(f"语速: 显示 {speed:.2f} → 实际参数 {effective_speed:.2f}")
                    st.write(f"生成 {len(chunks)} 段到 {output_path.relative_to(APP_DIR)}")
                    st.caption("模型推理运行在独立 Python 子进程中；如果底层 MLX 崩溃，Web 服务会保留运行并显示断点/报告。")
                    progress_bar = st.progress(0, text=f"准备开始 0/{len(chunks)}")
                    process_box = st.empty()
                    heartbeat_box = st.empty()
                    log_box = st.empty()
                    recent_logs: list[str] = []
                    last_segment_line = ""
                    generation_started_at = time.monotonic()
                    current_segment_started_at = generation_started_at
                    last_completed_count = 0
                    completed_segment_durations: list[float] = []

                    def render_progress(elapsed_seconds: float = 0.0, latest_line: str = "") -> None:
                        nonlocal last_segment_line, current_segment_started_at, last_completed_count
                        if re.match(r"^\[\d+/\d+\]", latest_line):
                            last_segment_line = latest_line
                            current_segment_started_at = time.monotonic()
                        snapshot = _generation_progress_snapshot(output_path, checkpoint_path, len(chunks))
                        completed = snapshot["completed"]
                        total = snapshot["total"]
                        current = snapshot["current"]
                        now = time.monotonic()
                        if completed > last_completed_count:
                            segment_elapsed = max(0.0, now - current_segment_started_at)
                            completed_segment_durations.extend([segment_elapsed] * (completed - last_completed_count))
                            current_segment_started_at = now
                            last_completed_count = completed
                        current_segment_elapsed = max(0.0, now - current_segment_started_at)
                        estimate = _estimated_generation_progress(
                            completed=completed,
                            total=total,
                            elapsed_seconds=elapsed_seconds,
                            current_segment_elapsed=current_segment_elapsed,
                            completed_segment_durations=completed_segment_durations,
                        )
                        ratio = float(estimate["ratio"] or 0.0)
                        if completed >= total and total:
                            label = f"已完成 {completed}/{total} 段"
                        else:
                            label = f"估算进度 {ratio * 100:.0f}% | 正在生成第 {current}/{total} 段，已完成 {completed}/{total} 段"
                        progress_bar.progress(ratio, text=label)
                        heartbeat_parts = [
                            f"运行中: {_format_elapsed(elapsed_seconds)}",
                            f"预计剩余: {_format_eta(estimate['remaining_seconds'])}",
                            label,
                        ]
                        if snapshot["segment_count"]:
                            heartbeat_parts.append(f"已写入 segment 文件: {snapshot['segment_count']}")
                        if last_segment_line:
                            heartbeat_parts.append(f"最近段落: {last_segment_line}")
                        heartbeat_box.info(" | ".join(heartbeat_parts))

                    def on_process(pid: int) -> None:
                        process_box.info(f"生成子进程 PID: {pid}")

                    def on_log(line: str) -> None:
                        recent_logs.append(line)
                        render_progress(elapsed_seconds=time.monotonic() - generation_started_at, latest_line=line)
                        log_text = html.escape("\n".join(recent_logs[-40:]))
                        log_box.markdown(f'<pre class="generation-log">{log_text}</pre>', unsafe_allow_html=True)

                    def on_heartbeat(elapsed_seconds: float, latest_line: str) -> None:
                        render_progress(elapsed_seconds=elapsed_seconds, latest_line=latest_line)

                    result = _run_generation_subprocess(
                        target_text=target_text,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                        output_path=output_path,
                        backend=backend,
                        model_ref=model_ref,
                        speed=effective_speed,
                        temperature=temperature,
                        chunk_max_chars=chunk_max_chars,
                        checkpoint_path=checkpoint_path,
                        checkpoint_metadata=profile_metadata,
                        output_format=output_format,
                        normalise=normalise,
                        resume=resume,
                        log_callback=on_log,
                        process_callback=on_process,
                        heartbeat_callback=on_heartbeat,
                    )
                    status.update(label="生成完成", state="complete", expanded=False)

                final_path = result
                final_mime = "audio/mpeg" if final_path.suffix.lower() == ".mp3" else "audio/wav"
                st.success(f"已保存音频: {final_path}")
                st.audio(str(final_path), format=final_mime)
                st.download_button(
                    "下载生成结果",
                    data=_download_bytes(final_path),
                    file_name=final_path.name,
                    mime=final_mime,
                    key=f"download_generated_{final_path.name}_{final_path.stat().st_mtime}",
                )
                if output_format == "both":
                    mp3_path = final_path.with_suffix(".mp3")
                    if mp3_path.exists():
                        st.download_button(
                            "下载 MP3",
                            data=_download_bytes(mp3_path),
                            file_name=mp3_path.name,
                            mime="audio/mpeg",
                            key=f"download_generated_{mp3_path.name}_{mp3_path.stat().st_mtime}",
                        )
                _show_quality_report(final_path, output_path)
            except Exception as exc:
                st.error(f"生成失败: {exc}")
                if _has_quality_artifacts(output_path, output_path):
                    _show_quality_report(output_path, output_path)
                if checkpoint_path.exists():
                    st.warning(f"已保留断点，可继续生成: {checkpoint_path.relative_to(APP_DIR)}")

    st.divider()
    _show_output_library()


if __name__ == "__main__":
    main()
