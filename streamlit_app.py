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
    page_icon="🎛️",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #0f1115;
        --panel: #171a20;
        --panel-2: #1d222b;
        --line: #2b313c;
        --text: #e7eaf0;
        --muted: #9aa3b2;
        --accent: #4fa3ff;
        --accent-2: #49c6a7;
        --danger: #e46f6f;
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: #11141a;
        border-right: 1px solid var(--line);
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }
    div[data-testid="stMetric"] label {
        color: var(--muted);
    }
    div[data-testid="stExpander"], div[data-testid="stForm"] {
        border-color: var(--line);
    }
    .console-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, #171a20 0%, #14171d 100%);
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    .console-title {
        font-size: 1.45rem;
        font-weight: 680;
        line-height: 1.2;
        margin: 0 0 0.25rem 0;
    }
    .console-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        min-height: 30px;
        padding: 0 0.7rem;
        border: 1px solid rgba(79, 163, 255, 0.45);
        border-radius: 999px;
        color: #cfe6ff;
        background: rgba(79, 163, 255, 0.10);
        font-size: 0.86rem;
        white-space: nowrap;
    }
    .section-note {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: -0.35rem;
        margin-bottom: 0.55rem;
    }
    .file-path {
        color: #c8d2df;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.86rem;
    }
    button[kind="primary"] {
        border: 1px solid rgba(79, 163, 255, 0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
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
        st.caption("outputs/ 里还没有 WAV 文件。")
        return

    for path in recent_files:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{path.name} - {modified} - {stat.st_size / 1024 / 1024:.1f} MB"):
            st.audio(str(path), format="audio/wav")
            st.download_button(
                "下载 WAV",
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
        model_choices = get_backend_model_choices(backend)

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
            "声音预设",
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
            placeholder="留空自动命名，或输入 my_episode.wav",
        )

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
        st.markdown('<p class="section-note">抽取公开文章页面的标题与正文，确认后再写入稿件。</p>', unsafe_allow_html=True)
        article_url = st.text_input("文章 URL", placeholder="https://example.com/article")
        col_extract, col_timeout = st.columns([1, 3])
        with col_extract:
            extract_clicked = st.button("抽取正文", use_container_width=True)
        with col_timeout:
            st.caption("抽取失败时会显示来自抽取器的中文错误信息。")

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
            article_preview = article.text[:1200]
            st.markdown(f"**标题：** {article.title}")
            st.markdown(f"**来源：** {article.source_url}")
            st.metric("正文字符数", f"{len(article.text):,}")
            st.text_area("正文预览", article_preview, height=220, disabled=True)
            if st.button("使用为稿件", type="primary", use_container_width=True):
                st.session_state["target_text"] = article.text
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

    st.subheader("参考声音")
    left, right = st.columns([1, 1])
    with left:
        ref_audio_upload = st.file_uploader(
            "参考音频",
            type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        )
        use_default_ref = False
        if DEFAULT_REF_AUDIO.exists():
            use_default_ref = st.checkbox(
                f"使用默认参考音频 ({DEFAULT_REF_AUDIO.relative_to(APP_DIR)})",
                value=ref_audio_upload is None,
            )
        else:
            st.warning("默认参考音频不存在: audio_samples/龙湖安置小区 2.m4a")

    with right:
        ref_text = st.text_area(
            "参考音频文本",
            value=DEFAULT_REF_TEXT if DEFAULT_REF_AUDIO.exists() else "",
            height=140,
            help="需要与参考音频实际朗读内容一致。",
        ).strip()

    chunks = split_text(target_text, max_chars=chunk_max_chars) if target_text else []
    output_path = _resolve_output_path(target_text, output_name)
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

    generate = st.button("生成播客 WAV", type="primary", use_container_width=True)
    if generate:
        errors = []
        if not target_text:
            errors.append("请先准备稿件正文。")
        if not model_ref:
            errors.append("请先选择可用模型。")
        if not ref_text:
            errors.append("请填写参考音频文本。")

        ref_audio_path: Optional[Path]
        if ref_audio_upload is not None:
            ref_audio_path = _persist_upload(ref_audio_upload, "ref_audio")
        elif use_default_ref and DEFAULT_REF_AUDIO.exists():
            ref_audio_path = DEFAULT_REF_AUDIO
        else:
            ref_audio_path = None

        if ref_audio_path is None:
            errors.append("请提供参考音频。")

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
                        normalise=normalise,
                    )
                    if not result:
                        raise RuntimeError("Generation returned no output path.")
                    status.update(label="生成完成", state="complete", expanded=False)

                final_path = Path(result)
                st.success(f"已保存 WAV: {final_path}")
                st.audio(str(final_path), format="audio/wav")
                st.download_button(
                    "下载生成结果",
                    data=_download_bytes(final_path),
                    file_name=final_path.name,
                    mime="audio/wav",
                    key=f"download_generated_{final_path.name}_{final_path.stat().st_mtime}",
                )
            except Exception as exc:
                st.error(f"生成失败: {exc}")
                if checkpoint_path.exists():
                    st.warning(f"已保留断点，可继续生成: {checkpoint_path.relative_to(APP_DIR)}")

    st.divider()
    _show_output_library()


if __name__ == "__main__":
    main()
