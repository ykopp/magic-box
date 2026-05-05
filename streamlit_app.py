import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import streamlit as st

from podcast_generator import generate_podcast
from tts_backends import get_backend_model_choices, get_default_model_ref, load_backend_model
from utils import _safe_remove, make_output_filename, split_text


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
RUNTIME_DIR = APP_DIR / "runtime" / "streamlit"
DEFAULT_REF_AUDIO = APP_DIR / "audio_samples" / "龙湖安置小区 2.m4a"
DEFAULT_REF_TEXT = (
    "大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。"
    "让我们开始吧。大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。让我们开始吧。"
)
MAX_PREVIEW_CHUNKS = 8


st.set_page_config(
    page_title="Podcast TTS Production Console",
    page_icon="🎙️",
    layout="wide",
)


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str, fallback: str) -> str:
    stem = Path(name or fallback).name
    stem = re.sub(r"[^\w.\- ]+", "_", stem, flags=re.UNICODE).strip()
    return stem or fallback


def _persist_upload(uploaded_file, prefix: str) -> Optional[Path]:
    if uploaded_file is None:
        return None
    _ensure_dirs()
    filename = _safe_filename(uploaded_file.name, f"{prefix}.bin")
    target = RUNTIME_DIR / f"{prefix}_{uuid4().hex}_{filename}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _read_uploaded_text(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def _normalise_output_path(text: str, requested_name: str) -> Path:
    requested_name = requested_name.strip()
    name = _safe_filename(requested_name, "podcast.wav")
    if not name.lower().endswith(".wav"):
        name = f"{Path(name).stem}.wav"
    return OUTPUT_DIR / name


def _auto_output_path(text: str) -> Path:
    text_key = text.strip()
    auto_state = st.session_state.get("auto_output")
    if auto_state and auto_state.get("text") == text_key and auto_state.get("path"):
        return Path(auto_state["path"])

    path = OUTPUT_DIR / make_output_filename(text_key[:20])
    st.session_state["auto_output"] = {
        "text": text_key,
        "path": str(path),
    }
    return path


def _resolve_output_path(text: str, requested_name: str) -> Path:
    if requested_name.strip():
        return _normalise_output_path(text, requested_name)
    return _auto_output_path(text)


def _format_model_choice(choice) -> str:
    return getattr(choice, "label", str(choice))


def _get_default_choice_index(choices: list, backend: str) -> int:
    if not choices:
        return 0
    default_ref = get_default_model_ref(backend)
    for index, choice in enumerate(choices):
        if choice.value == default_ref:
            return index
    return 0


@st.cache_resource(show_spinner=False)
def _load_model_cached(backend: str, model_ref: str):
    model, resolved_ref = load_backend_model(backend, model_ref)
    return model, resolved_ref


def _recent_outputs(limit: int = 10) -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    wavs = sorted(
        OUTPUT_DIR.glob("*.wav"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return wavs[:limit]


def _download_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _show_output_library() -> None:
    st.subheader("Output Library")
    recent_files = _recent_outputs()
    if not recent_files:
        st.caption("No WAV files in outputs/ yet.")
        return

    for path in recent_files:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{path.name} - {modified} - {stat.st_size / 1024 / 1024:.1f} MB"):
            st.audio(str(path), format="audio/wav")
            st.download_button(
                "Download WAV",
                data=_download_bytes(path),
                file_name=path.name,
                mime="audio/wav",
                key=f"download_{path.name}_{stat.st_mtime}",
            )


def _sync_uploaded_text(uploaded_file) -> None:
    if uploaded_file is None:
        return
    upload_id = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("last_text_upload_id") == upload_id:
        return
    persisted_path = _persist_upload(uploaded_file, "script_text")
    st.session_state["target_text"] = _read_uploaded_text(uploaded_file)
    st.session_state["target_text_path"] = str(persisted_path) if persisted_path else ""
    st.session_state["last_text_upload_id"] = upload_id


def main() -> None:
    _ensure_dirs()

    st.title("Podcast TTS Production Console")

    with st.sidebar:
        st.header("Generation Controls")
        backend = st.selectbox("Backend", ["qwen", "voxcpm"], index=0)
        model_choices = get_backend_model_choices(backend)

        if model_choices:
            model_choice = st.selectbox(
                "Model",
                model_choices,
                index=_get_default_choice_index(model_choices, backend),
                format_func=_format_model_choice,
            )
            model_ref = model_choice.value
        else:
            model_ref = ""
            st.error(f"No local/available model choices found for backend '{backend}'.")

        speed = st.slider("Speed", min_value=0.50, max_value=2.00, value=1.15, step=0.05)
        temperature = st.slider("Temperature", min_value=0.10, max_value=2.00, value=1.00, step=0.05)
        normalise = st.toggle("Normalise loudness", value=True)
        resume = st.toggle("Resume from checkpoint", value=False)
        output_name = st.text_input(
            "Output filename or stem",
            value="",
            placeholder="Auto-generate from text, or enter my_episode.wav",
        )

        if backend == "voxcpm":
            st.info("VoxCPM is experimental. Speed and temperature are ignored by the backend layer.")

    text_file = st.file_uploader("Load script from .txt", type=["txt"])
    _sync_uploaded_text(text_file)
    if st.session_state.get("target_text_path"):
        st.caption(f"Uploaded text saved to {Path(st.session_state['target_text_path']).relative_to(APP_DIR)}")

    if "target_text" not in st.session_state:
        st.session_state["target_text"] = ""

    target_text = st.text_area(
        "Podcast script",
        key="target_text",
        height=260,
        placeholder="Paste or upload the long-form podcast script here.",
    ).strip()

    left, right = st.columns([1, 1])
    with left:
        ref_audio_upload = st.file_uploader(
            "Reference audio",
            type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        )
        use_default_ref = False
        if DEFAULT_REF_AUDIO.exists():
            use_default_ref = st.checkbox(
                f"Use default reference audio ({DEFAULT_REF_AUDIO.relative_to(APP_DIR)})",
                value=ref_audio_upload is None,
            )
        else:
            st.warning("Default reference audio is not present at audio_samples/龙湖安置小区 2.m4a.")

    with right:
        ref_text = st.text_area(
            "Reference transcript",
            value=DEFAULT_REF_TEXT if DEFAULT_REF_AUDIO.exists() else "",
            height=140,
            help="This must match the spoken content in the reference audio.",
        ).strip()

    chunks = split_text(target_text, max_chars=80) if target_text else []
    output_path = _resolve_output_path(target_text, output_name)
    checkpoint_path = output_path.with_suffix(".ckpt")

    st.subheader("Chunk Preview")
    stats = st.columns(3)
    stats[0].metric("Characters", len(target_text))
    stats[1].metric("Chunks", len(chunks))
    stats[2].metric("Estimated output", str(output_path.relative_to(APP_DIR)))

    if chunks:
        preview_text = "\n\n".join(
            f"{index}. {chunk}" for index, chunk in enumerate(chunks[:MAX_PREVIEW_CHUNKS], start=1)
        )
        if len(chunks) > MAX_PREVIEW_CHUNKS:
            preview_text += f"\n\n... {len(chunks) - MAX_PREVIEW_CHUNKS} more chunk(s)"
        st.text_area("Preview", preview_text, height=180, disabled=True)
    else:
        st.info("Enter or upload script text to preview chunks.")

    generate = st.button("Generate Podcast WAV", type="primary", use_container_width=True)
    if generate:
        errors = []
        if not target_text:
            errors.append("Script text is required.")
        if not model_ref:
            errors.append("Select a model before generation.")
        if not ref_text:
            errors.append("Reference transcript is required for voice clone generation.")

        ref_audio_path: Optional[Path]
        if ref_audio_upload is not None:
            ref_audio_path = _persist_upload(ref_audio_upload, "ref_audio")
        elif use_default_ref and DEFAULT_REF_AUDIO.exists():
            ref_audio_path = DEFAULT_REF_AUDIO
        else:
            ref_audio_path = None

        if ref_audio_path is None:
            errors.append("Reference audio is required for voice clone generation.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            assert ref_audio_path is not None
            if not resume and checkpoint_path.exists():
                _safe_remove(str(checkpoint_path))

            try:
                with st.status("Loading model and generating audio...", expanded=True) as status:
                    st.write(f"Backend: {backend}")
                    st.write(f"Model: {model_ref}")
                    model, resolved_ref = _load_model_cached(backend, model_ref)
                    st.write(f"Loaded model: {resolved_ref}")
                    st.write(f"Generating {len(chunks)} chunk(s) to {output_path.relative_to(APP_DIR)}")
                    result = generate_podcast(
                        model=model,
                        backend=backend,
                        ref_audio_path=str(ref_audio_path),
                        ref_text=ref_text,
                        target_text=target_text,
                        output_path=str(output_path),
                        speed=speed,
                        temperature=temperature,
                        checkpoint_path=str(checkpoint_path),
                        normalise=normalise,
                    )
                    if not result:
                        raise RuntimeError("Generation returned no output path.")
                    status.update(label="Generation complete", state="complete", expanded=False)

                final_path = Path(result)
                st.success(f"Saved WAV: {final_path}")
                st.audio(str(final_path), format="audio/wav")
                st.download_button(
                    "Download generated WAV",
                    data=_download_bytes(final_path),
                    file_name=final_path.name,
                    mime="audio/wav",
                    key=f"download_generated_{final_path.name}_{final_path.stat().st_mtime}",
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                if checkpoint_path.exists():
                    st.warning(f"Checkpoint saved for resume: {checkpoint_path.relative_to(APP_DIR)}")

    st.divider()
    _show_output_library()


if __name__ == "__main__":
    main()
