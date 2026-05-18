import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import streamlit as st

from article_extractor import extract_article_from_url
from podcast_generator import generate_podcast
from tts_backends import get_backend_model_choices, get_default_model_ref, load_backend_model
from utils import _safe_remove, make_output_filename, split_text
from voice_controls import get_preset, optimize_podcast_rhythm, preset_names
from voice_profiles import delete_profile, list_profiles, save_profile, slugify_profile_id, text_fingerprint


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
RUNTIME_DIR = APP_DIR / "runtime" / "streamlit"
MAX_PREVIEW_CHUNKS = 8
PROFILE_AUDIO_TYPES = ["wav", "mp3", "m4a", "aac", "flac", "ogg"]


st.set_page_config(
    page_title="Podcast TTS Production Console",
    page_icon="🎛️",
    layout="wide",
)


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


@st.cache_resource(show_spinner=False)
def _load_model_cached(backend: str, model_ref: str):
    model, resolved_ref = load_backend_model(backend, model_ref)
    return model, resolved_ref


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
                help="控制整体朗读速度，数值越高越快。",
            )
            temperature = st.slider(
                "表达随机性",
                min_value=0.10,
                max_value=2.00,
                step=0.05,
                key="voice_temperature",
                help="控制声音表现的变化幅度，过高可能更不稳定。",
            )
            chunk_max_chars = st.slider(
                "单段字符上限",
                min_value=40,
                max_value=140,
                step=5,
                key="voice_chunk_max_chars",
                help="控制切分粒度，短段更稳，长段更连贯。",
            )

        normalise = st.toggle("响度标准化", value=True)
        resume = st.toggle(
            "从断点继续",
            value=False,
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
            st.info("VoxCPM 仍是实验后端；语速和表达随机性可能会被后端忽略。")

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
            ref_text = st.text_area(
                "参考音频文本",
                value=selected_profile.ref_text,
                height=140,
                help="必须与源声音里实际朗读的文字一致；缺失时不能生成。",
            ).strip()
            profile_metadata = {
                "voice_profile_id": selected_profile.id,
                "voice_profile_fingerprint": f"{selected_profile.fingerprint}:{text_fingerprint(ref_text)}",
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
        with right:
            ref_text = st.text_area(
                "参考音频文本",
                value="",
                height=140,
                help="需要与临时源声音实际朗读内容一致。",
            ).strip()
        profile_metadata = {"voice_profile_id": "temporary_upload"}

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
                        saved_profile = save_profile(
                            APP_DIR,
                            display_name=profile_name,
                            audio_source=saved_audio,
                            transcript=profile_transcript,
                            description=profile_description,
                            default_preset=profile_preset,
                            profile_id=slugify_profile_id(profile_name),
                        )
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

    if chunks:
        preview_text = "\n\n".join(
            f"{index}. {chunk}" for index, chunk in enumerate(chunks[:MAX_PREVIEW_CHUNKS], start=1)
        )
        if len(chunks) > MAX_PREVIEW_CHUNKS:
            preview_text += f"\n\n... 还有 {len(chunks) - MAX_PREVIEW_CHUNKS} 段"
        st.text_area("切分预览", preview_text, height=180, disabled=True)
    else:
        st.info("输入、导入或抽取稿件后可预览切分结果。")

    generate = st.button("生成播客音频", type="primary", use_container_width=True)
    if generate:
        errors = []
        if not target_text:
            errors.append("请先准备稿件正文。")
        if not model_ref:
            errors.append("请先选择可用模型。")
        if not ref_text:
            errors.append("请填写参考音频文本。")

        if profile_mode == "临时上传源声音" and ref_audio_upload is not None:
            ref_audio_path = _persist_upload(ref_audio_upload, "ref_audio")
            if ref_audio_path is not None:
                profile_metadata = {
                    "voice_profile_id": "temporary_upload",
                    "voice_profile_fingerprint": f"temporary_upload:{ref_audio_upload.name}:{ref_audio_upload.size}:{text_fingerprint(ref_text)}",
                }

        if ref_audio_path is None:
            errors.append("请提供参考音频。")
        elif not ref_audio_path.exists():
            errors.append("参考音频文件不存在。")
        if selected_profile is not None and selected_profile.needs_transcript and not ref_text:
            errors.append("这个声音 Profile 缺少参考文本，请先补全后再生成。")

        if errors:
            for error in errors:
                st.error(error)
        else:
            assert ref_audio_path is not None
            if not resume and checkpoint_path.exists():
                _safe_remove(str(checkpoint_path))

            try:
                with st.status("加载模型并生成音频...", expanded=True) as status:
                    st.write(f"后端: {backend}")
                    st.write(f"模型: {model_ref}")
                    model, resolved_ref = _load_model_cached(backend, model_ref)
                    st.write(f"已加载模型: {resolved_ref}")
                    st.write(f"生成 {len(chunks)} 段到 {output_path.relative_to(APP_DIR)}")
                    result = generate_podcast(
                        model=model,
                        backend=backend,
                        ref_audio_path=str(ref_audio_path),
                        ref_text=ref_text,
                        target_text=target_text,
                        output_path=str(output_path),
                        speed=speed,
                        temperature=temperature,
                        chunk_max_chars=chunk_max_chars,
                        checkpoint_path=str(checkpoint_path),
                        checkpoint_metadata=profile_metadata,
                        output_format=output_format,
                        normalise=normalise,
                    )
                    if not result:
                        raise RuntimeError("Generation returned no output path.")
                    status.update(label="生成完成", state="complete", expanded=False)

                final_path = Path(result)
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
            except Exception as exc:
                st.error(f"生成失败: {exc}")
                if checkpoint_path.exists():
                    st.warning(f"已保留断点，可继续生成: {checkpoint_path.relative_to(APP_DIR)}")

    st.divider()
    _show_output_library()


if __name__ == "__main__":
    main()
