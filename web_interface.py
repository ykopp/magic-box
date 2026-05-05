import os
import sys
import warnings
import time
import numpy as np
import gradio as gr
from pathlib import Path

try:
    from model_manager import model_manager
except Exception:
    model_manager = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from utils import (
    convert_audio_if_needed,
    get_smart_path as _get_smart_path_util,
    make_output_filename,
    model_cache,
    normalise_loudness,
    SAMPLE_RATE,
    split_text,
    _safe_remove,
)
from tts_backends import (
    QWEN_OBJECTIVE_CHOICES,
    backend_note,
    generate_backend_chunk,
    get_backend_model_choices,
    get_default_model_ref,
    load_backend_model,
    supported_task_choices,
)

try:
    import soundfile as sf
except ImportError as e:
    print(f"Error: 无法导入必要模块 - {e}")
    print("请确保依赖已正确安装")
    sys.exit(1)

MODEL_PATH = "models/Qwen3-TTS-12Hz-1.7B-Base-8bit"
BACKEND_CHOICES = [
    ("Qwen Mainline", "qwen"),
    ("VoxCPM Experimental", "voxcpm"),
]
DEFAULT_REF_AUDIO = "audio_samples/龙湖安置小区 2.m4a"
DEFAULT_REF_TEXT = "大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。让我们开始吧。大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。让我们开始吧。"
TASK_CHOICES = supported_task_choices("qwen")
OBJECTIVE_CHOICES = QWEN_OBJECTIVE_CHOICES
SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}
SPEAKER_CHOICES = [name for names in SPEAKER_MAP.values() for name in names]
DEFAULT_CUSTOM_SPEAKER = "Vivian"
DEFAULT_CUSTOM_INSTRUCT = "Normal tone"
DEFAULT_DESIGN_INSTRUCT = "calm podcast narrator with clear articulation"

USER_MODELS_DIR = Path.home() / "podcast_generator_models"

# model_cache (thread-safe) is imported from utils — no bare globals needed
available_models = []


def has_default_reference_audio():
    return Path(DEFAULT_REF_AUDIO).exists()


def format_route_preview_html(route_result):
    if route_result is None:
        return '<div style="color: #888; font-size: 0.9rem;">模型路由不可用</div>'

    lines = [f"路由任务: <b>{route_result.task}</b> | 目标: <b>{route_result.objective}</b>"]
    lines.append(route_result.message)
    if route_result.route:
        lines.append("候选链路:")
        for index, model in enumerate(route_result.route, start=1):
            status = "已下载" if model.downloaded else "未下载"
            lines.append(
                f"{index}. {model.name} ({status}, Q{model.quality_score}/S{model.speed_score})"
            )
    else:
        lines.append("无候选模型")

    return (
        '<div style="color: #888; font-size: 0.9rem;">'
        + "<br>".join(lines)
        + "</div>"
    )


def wrap_route_preview_html(content_html):
    return f'<div class="route-preview">{content_html}</div>'

def get_available_models(backend="qwen"):
    global available_models
    models = [{"label": choice.label, "path": choice.value} for choice in get_backend_model_choices(backend)]
    available_models = models
    return models

def get_default_model_path(backend="qwen"):
    if backend == "qwen" and model_manager is not None:
        routed = model_manager.route_model_for_task(
            task="clone",
            objective="quality",
            downloaded_only=True,
        )
        best_model = routed.selected
        if not best_model:
            best_model = model_manager.recommend_best_model(objective="quality", downloaded_only=True)
        if best_model and best_model.local_path:
            return str(best_model.local_path)

    return get_default_model_ref(backend)


def build_model_choices(models):
    if not models:
        return [("Default", MODEL_PATH)]

    choices = []
    for m in models:
        choices.append((m["label"], m["path"]))
    return choices


def get_model_status_html(backend="qwen", force=False, extra_note=""):
    if backend == "voxcpm":
        note_text = f"<br>{extra_note}" if extra_note else ""
        return (
            '<div style="color: #888; font-size: 0.9rem;">'
            "实验后端: VoxCPM 不参与 Qwen 路由；建议仅用于隔离 A/B 试听。"
            f"{note_text}</div>"
        )

    if model_manager is None:
        models = get_available_models(backend)
        suffix = f"<br>{extra_note}" if extra_note else ""
        return (
            f'<div style="color: #888; font-size: 0.9rem;">'
            f"Found {len(models)} model(s){suffix}</div>"
        )

    try:
        summary = model_manager.get_model_status_summary(objective="quality", force=force)
        best_name = summary.get("best_downloaded") or "无"
        updates = len(summary.get("updates", []))
        auto_update = "开" if summary.get("auto_update_enabled") else "关"
        check_text = summary.get("update_summary", "未检查")
        note_text = f"<br>{extra_note}" if extra_note else ""
        return (
            '<div style="color: #888; font-size: 0.9rem;">'
            f"最佳模型: <b>{best_name}</b><br>"
            f"已下载: {summary.get('downloaded_models', 0)}/{summary.get('total_models', 0)} | "
            f"可更新: {updates} | 自动更新: {auto_update}<br>"
            f"{check_text}{note_text}</div>"
        )
    except Exception as e:
        return f'<div style="color: #ff6666; font-size: 0.9rem;">模型状态读取失败: {e}</div>'

def get_smart_path(folder_name):
    """Delegate to utils — supports snapshot dir sorting and user model dir."""
    return _get_smart_path_util(folder_name)

def load_model_once(backend="qwen", model_path=None):
    """Load a model, reusing the thread-safe cache when the same path is requested."""
    if model_path is None:
        model_path = get_default_model_path(backend)

    try:
        _, cache_key = load_backend_model(backend, model_path)
        return True, f"模型加载成功: {model_path}", cache_key
    except Exception as e:
        return False, f"模型加载失败: {e}", None

def generate_podcast_gradio(
    backend,
    model_path,
    task_mode,
    objective_mode,
    use_task_routing,
    auto_download_routing,
    text_input,
    text_file,
    ref_audio,
    ref_text_input,
    use_default_ref,
    custom_speaker,
    custom_instruction,
    design_instruction,
    speed,
    temperature,
    progress=gr.Progress(),
):
    selected_model_path = model_path
    selected_model_name = None
    routing_note = ""
    if backend == "qwen" and use_task_routing and model_manager is not None:
        routed = model_manager.ensure_model_for_task(
            task=task_mode,
            objective=objective_mode,
            auto_download=auto_download_routing,
        )
        if not routed.selected or not routed.selected.local_path:
            return None, f"模型路由失败: {routed.message}", None
        selected_model_path = str(routed.selected.local_path)
        selected_model_name = routed.selected.name
        routing_note = f" [路由模型: {selected_model_name}]"

    success, msg, cache_key = load_model_once(backend, selected_model_path)
    if not success:
        return None, msg, None
    
    # 处理文本输入
    if text_file:
        file_path = text_file.name if hasattr(text_file, 'name') else str(text_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            target_text = f.read().strip()
    elif text_input:
        target_text = text_input.strip()
    else:
        return None, "请输入文本内容或上传文本文件", None

    if not target_text:
        return None, "文本内容不能为空", None
    
    # 分段生成
    chunks = split_text(target_text, max_chars=80)
    if not chunks:
        return None, "文本分段失败，请检查输入内容", None
    progress(0, desc=f"准备生成，共 {len(chunks)} 段")
    
    all_audio = []
    total_chunks = len(chunks)
    
    clean_audio = None
    ref_audio_path = None
    ref_text = None
    if task_mode == "clone":
        if use_default_ref:
            if not has_default_reference_audio():
                return None, "默认参考音频不存在，请关闭 USE DEFAULT 后上传参考音频", None
            ref_audio_path = DEFAULT_REF_AUDIO
            ref_text = DEFAULT_REF_TEXT
        else:
            if not ref_audio:
                return None, "请上传参考音频", None
            if not ref_text_input:
                return None, "请输入参考音频文本", None
            ref_audio_path = ref_audio.name if hasattr(ref_audio, 'name') else str(ref_audio)
            ref_text = ref_text_input

        if not os.path.exists(ref_audio_path):
            return None, "参考音频不存在", None

        clean_audio = convert_audio_if_needed(ref_audio_path)
        if not clean_audio:
            return None, "参考音频转换失败", None

    effective_speaker = (custom_speaker or DEFAULT_CUSTOM_SPEAKER).strip()
    effective_custom_instruction = (
        custom_instruction.strip() if custom_instruction else DEFAULT_CUSTOM_INSTRUCT
    )
    effective_design_instruction = (
        design_instruction.strip() if design_instruction else DEFAULT_DESIGN_INSTRUCT
    )

    # Retrieve the loaded model from thread-safe cache
    _model = model_cache.get(cache_key)
    if _model is None:
        return None, f"模型未加载: {selected_model_path}", None

    failed_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        progress((i - 1) / total_chunks, desc=f"生成第 {i}/{total_chunks} 段")
        try:
            audio_chunk = generate_backend_chunk(
                backend=backend,
                model=_model,
                task_mode=task_mode,
                chunk=chunk,
                ref_audio_path=clean_audio,
                ref_text=ref_text,
                custom_speaker=effective_speaker,
                custom_instruction=effective_custom_instruction,
                design_instruction=effective_design_instruction,
                speed=speed,
                temperature=temperature,
            )
            all_audio.append(np.array(audio_chunk))
        except Exception as e:
            failed_chunks.append(i)
            print(f"[web] 第 {i} 段生成失败: {e}")

    # 清理临时参考音频文件
    if clean_audio and clean_audio != ref_audio_path:
        _safe_remove(clean_audio)

    if not all_audio:
        return None, "生成失败，所有片段均无输出", None

    final_audio = np.concatenate(all_audio)

    # 响度标准化 -16 LUFS (播客行业标准)
    progress(0.95, desc="响度标准化 (-16 LUFS)…")
    final_audio = normalise_loudness(final_audio, SAMPLE_RATE)

    os.makedirs("outputs", exist_ok=True)
    snippet = chunks[0][:20] if chunks else "podcast"
    output_path = os.path.join("outputs", make_output_filename(snippet))
    sf.write(output_path, final_audio, SAMPLE_RATE)

    duration = len(final_audio) / SAMPLE_RATE
    warn = f"  ⚠ {len(failed_chunks)} 段失败: {failed_chunks}" if failed_chunks else ""
    backend_suffix = "" if backend == "qwen" else " [VoxCPM experimental]"
    return output_path, f"生成成功！总时长: {duration:.1f}s{routing_note}{backend_suffix}{warn}", output_path

def create_interface():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    
    * {
        image-rendering: pixelated !important;
    }
    
    .gradio-container {
        background: #0a0a0a !important;
        min-height: 100vh;
        font-family: 'VT323', monospace !important;
    }
    
    .main-container {
        background: repeating-linear-gradient(
            0deg,
            rgba(255, 200, 100, 0.03) 0px,
            rgba(255, 200, 100, 0.03) 1px,
            transparent 1px,
            transparent 2px
        );
        padding: 20px;
    }
    
    .pixel-title {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .pixel-title h1 {
        font-family: 'VT323', monospace !important;
        font-size: 2.8rem !important;
        color: #ffd700 !important;
        text-shadow: 
            0 0 15px rgba(255, 215, 0, 1),
            0 0 30px rgba(255, 215, 0, 0.7),
            0 0 50px rgba(255, 215, 0, 0.5);
        letter-spacing: 4px;
        animation: flicker 0.15s infinite;
    }
    
    @keyframes flicker {
        0% { opacity: 0.97; }
        50% { opacity: 1; }
        100% { opacity: 0.98; }
    }
    
    .pixel-subtitle {
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
        color: #ffcc66 !important;
        margin-top: 10px;
        letter-spacing: 2px;
    }
    
    .pixel-card {
        background: #0d0d0d !important;
        border: 2px solid #ffcc00 !important;
        border-radius: 0 !important;
        box-shadow: 
            inset 0 0 50px rgba(255, 204, 0, 0.08),
            0 0 0 1px #1a1a1a;
        padding: 20px !important;
        margin-bottom: 20px !important;
        position: relative;
    }
    
    .pixel-card::before {
        content: '';
        position: absolute;
        top: 4px;
        left: 4px;
        right: 4px;
        bottom: 4px;
        border: 1px solid rgba(255, 204, 0, 0.3);
        pointer-events: none;
    }

    .step-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 0 0 18px 0;
    }

    .step-chip {
        border: 1px solid #6f5a1b;
        background: linear-gradient(180deg, rgba(255, 204, 0, 0.16), rgba(0, 0, 0, 0.1));
        color: #ffe08a;
        padding: 8px 10px;
        letter-spacing: 1px;
        font-size: 0.95rem;
    }

    .flow-note {
        border-left: 2px solid #ffcc00;
        padding: 8px 10px;
        background: rgba(255, 204, 0, 0.06);
        color: #ffdb7a;
        font-size: 0.92rem;
        margin-top: 8px;
    }

    .route-preview {
        border: 1px dashed #8d6f1f;
        background: rgba(255, 204, 0, 0.05);
        padding: 10px;
        margin-top: 8px;
    }
    
    .pixel-section-title {
        font-family: 'VT323', monospace !important;
        font-size: 1.3rem !important;
        color: #ffe066 !important;
        margin-bottom: 15px !important;
        text-shadow: 0 0 15px rgba(255, 224, 102, 0.8);
        padding-bottom: 8px;
        border-bottom: 1px solid #444;
        letter-spacing: 2px;
    }
    
    .pixel-input textarea,
    .pixel-input input {
        background: #0a0a0a !important;
        border: 1px solid #888 !important;
        border-radius: 0 !important;
        color: #ffdd66 !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.2rem !important;
        padding: 12px !important;
        box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.8);
        transition: all 0.2s !important;
    }
    
    .pixel-input textarea:focus,
    .pixel-input input:focus {
        border-color: #ffcc00 !important;
        box-shadow: 
            inset 0 0 30px rgba(0, 0, 0, 0.8),
            0 0 20px rgba(255, 204, 0, 0.3);
        outline: none !important;
    }
    
    .pixel-input textarea::placeholder {
        color: #776644 !important;
    }
    
    .pixel-input label {
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
        color: #ffcc33 !important;
    }
    
    .pixel-file {
        background: #0a0a0a !important;
        border: 1px dashed #888 !important;
        border-radius: 0 !important;
        padding: 20px !important;
    }
    
    .pixel-file label {
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
        color: #ffcc33 !important;
    }
    
    .pixel-checkbox {
        background: #0a0a0a !important;
        border: 1px solid #ffaa00 !important;
        border-radius: 0 !important;
        padding: 12px !important;
    }
    
    .pixel-checkbox label {
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
        color: #ffaa00 !important;
    }
    
    .pixel-slider {
        background: #0a0a0a !important;
        border: 1px solid #555 !important;
        border-radius: 0 !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    
    .pixel-slider label {
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
        color: #ffcc33 !important;
    }
    
    .pixel-slider input[type="range"] {
        -webkit-appearance: none;
        background: #1a1a1a !important;
        height: 6px !important;
        border: none !important;
        border-radius: 0 !important;
    }
    
    .pixel-slider input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 18px !important;
        height: 18px !important;
        background: #ffcc00 !important;
        border: none !important;
        border-radius: 0 !important;
        cursor: pointer;
        box-shadow: 0 0 15px rgba(255, 204, 0, 0.7);
    }
    
    .pixel-btn {
        font-family: 'VT323', monospace !important;
        font-size: 1.4rem !important;
        background: #1a1a1a !important;
        border: 2px solid #ffcc00 !important;
        border-radius: 0 !important;
        color: #ffcc00 !important;
        padding: 15px 40px !important;
        text-shadow: 0 0 15px rgba(255, 204, 0, 0.8);
        box-shadow: 
            0 0 25px rgba(255, 204, 0, 0.3),
            inset 0 0 20px rgba(255, 204, 0, 0.08);
        cursor: pointer;
        transition: all 0.2s !important;
        letter-spacing: 3px;
    }
    
    .pixel-btn:hover {
        background: #ffcc00 !important;
        color: #0a0a0a !important;
        box-shadow: 
            0 0 40px rgba(255, 204, 0, 0.6),
            inset 0 0 20px rgba(255, 255, 255, 0.15);
    }
    
    .pixel-btn:active {
        transform: scale(0.98);
    }
    
    .output-card {
        background: #0d0d0d !important;
        border: 2px solid #ffaa00 !important;
        box-shadow: 
            inset 0 0 50px rgba(255, 170, 0, 0.08),
            0 0 0 1px #1a1a1a;
    }
    
    .output-card::before {
        border-color: rgba(255, 170, 0, 0.3) !important;
    }
    
    .output-card .pixel-section-title {
        color: #ffcc66 !important;
        border-bottom-color: #ffaa00 !important;
    }
    
    .pixel-audio {
        background: #0a0a0a !important;
        border: 1px solid #ffaa00 !important;
        border-radius: 0 !important;
        padding: 15px !important;
    }
    
    .pixel-status {
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
        background: #0a0a0a !important;
        border: 1px solid #555 !important;
        border-radius: 0 !important;
        color: #ffcc66 !important;
        padding: 12px !important;
        text-align: center;
    }
    
    .pixel-download {
        background: #0a0a0a !important;
        border: 1px dashed #ffaa00 !important;
        border-radius: 0 !important;
        padding: 15px !important;
    }
    
    .pixel-download label {
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
        color: #ffaa00 !important;
    }
    
    .scanline {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        background: repeating-linear-gradient(
            0deg,
            rgba(0, 0, 0, 0.1) 0px,
            rgba(0, 0, 0, 0.1) 1px,
            transparent 1px,
            transparent 2px
        );
        z-index: 9999;
    }
    
    .crt-effect {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        background: radial-gradient(ellipse at center, transparent 0%, rgba(0,0,0,0.2) 100%);
        z-index: 9998;
    }
    
    .pixel-icon {
        font-size: 1rem;
        margin-right: 8px;
    }
    
    .gradio-container .prose {
        font-family: 'VT323', monospace !important;
    }
    
    .contain {
        max-width: 1400px !important;
    }
    
    .gradio-container .gr-check-label {
        color: #ffaa00 !important;
    }
    
    .gradio-container input[type="checkbox"]:checked {
        background: #ffcc00 !important;
        border-color: #ffcc00 !important;
    }
    
    .gradio-container select {
        background: #0a0a0a !important;
        border: 1px solid #888 !important;
        color: #ffdd66 !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
    }
    
    .gradio-container .gr-box {
        color: #ffdd66 !important;
    }
    
    .gradio-container .gr-input-label, 
    .gradio-container .gr-radio-label {
        color: #ffcc33 !important;
    }

    @media (max-width: 900px) {
        .pixel-title h1 {
            font-size: 2.1rem !important;
        }
        .pixel-btn {
            font-size: 1.1rem !important;
            padding: 11px 18px !important;
            letter-spacing: 1px;
        }
        .pixel-section-title {
            font-size: 1.1rem !important;
            letter-spacing: 1px;
        }
        .step-strip {
            grid-template-columns: 1fr 1fr;
        }
    }
    """

    default_ref_available = has_default_reference_audio()
    default_ref_checked = default_ref_available
    ref_inputs_interactive = not default_ref_checked
    ref_status_text = (
        f"默认参考音频可用: {DEFAULT_REF_AUDIO}"
        if default_ref_available
        else "未找到默认参考音频，请取消 USE DEFAULT 并上传自定义参考音频。"
    )
    if model_manager is not None:
        initial_route_preview = wrap_route_preview_html(
            format_route_preview_html(
                model_manager.route_model_for_task(
                    task="clone",
                    objective="quality",
                    downloaded_only=False,
                )
            )
        )
    else:
        initial_route_preview = wrap_route_preview_html(
            '<div style="color: #888; font-size: 0.9rem;">模型路由不可用</div>'
        )
    
    with gr.Blocks(
        title="PODCAST GENERATOR", 
        theme=gr.themes.Default(
            primary_hue="yellow",
            secondary_hue="orange",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("VT323")
        ),
        css=custom_css
    ) as demo:
        gr.HTML("""
            <div class="scanline"></div>
            <div class="crt-effect"></div>
            <div class="pixel-title">
                <h1>PODCAST GENERATOR</h1>
                <p class="pixel-subtitle">[ QWEN3-TTS VOICE SYNTHESIS SYSTEM v1.0 ]</p>
            </div>
        """)
        gr.HTML("""
            <div class="step-strip">
                <div class="step-chip">1. 选择 TASK MODE</div>
                <div class="step-chip">2. 应用模型路由</div>
                <div class="step-chip">3. 输入文本与参数</div>
                <div class="step-chip">4. 生成并试听</div>
            </div>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["pixel-card"]):
                    gr.HTML('<div class="pixel-section-title">► TEXT INPUT</div>')
                    text_input = gr.Textbox(
                        label="ENTER TEXT CONTENT",
                        placeholder="> INPUT TEXT HERE...\n> SUPPORTS LONG TEXT\n> AUTO SEGMENTATION ENABLED",
                        lines=6,
                        elem_classes=["pixel-input"]
                    )
                    text_file = gr.File(
                        label="OR UPLOAD .TXT FILE",
                        file_types=[".txt"],
                        elem_classes=["pixel-file"]
                    )
                
                with gr.Group(elem_classes=["pixel-card"], visible=True) as clone_ref_group:
                    gr.HTML('<div class="pixel-section-title">► REFERENCE AUDIO</div>')
                    with gr.Row():
                        with gr.Column(scale=1):
                            use_default_ref = gr.Checkbox(
                                label="USE DEFAULT",
                                value=default_ref_checked,
                                elem_classes=["pixel-checkbox"]
                            )
                        with gr.Column(scale=2):
                            ref_audio = gr.File(
                                label="UPLOAD CUSTOM AUDIO",
                                file_types=[".wav", ".mp3", ".m4a"],
                                interactive=ref_inputs_interactive,
                                elem_classes=["pixel-file"]
                            )
                    ref_text_input = gr.Textbox(
                        label="REFERENCE TEXT",
                        placeholder="> ENTER REFERENCE AUDIO TEXT...",
                        lines=2,
                        interactive=ref_inputs_interactive,
                        elem_classes=["pixel-input"]
                    )
                    gr.HTML(
                        f'<div style="color: #aa9966; font-size: 0.9rem;">{ref_status_text}</div>'
                    )
                
                with gr.Group(elem_classes=["pixel-card"]):
                    gr.HTML('<div class="pixel-section-title">► MODEL SELECTION</div>')
                    backend_selector = gr.Dropdown(
                        choices=BACKEND_CHOICES,
                        value="qwen",
                        label="BACKEND",
                        interactive=True,
                        elem_classes=["pixel-input"],
                    )
                    gr.HTML(
                        f'<div class="flow-note">{backend_note("qwen")}</div>'
                    )
                    models = get_available_models("qwen")
                    model_choices = build_model_choices(models)
                    default_model = get_default_model_path("qwen")
                    if not any(value == default_model for _, value in model_choices):
                        default_model = model_choices[0][1] if model_choices else MODEL_PATH
                    model_selector = gr.Dropdown(
                        choices=model_choices,
                        value=default_model,
                        label="SELECT MODEL",
                        interactive=True,
                        elem_classes=["pixel-input"]
                    )
                    with gr.Row():
                        refresh_models_btn = gr.Button("REFRESH MODELS", elem_classes=["pixel-btn"])
                        check_updates_btn = gr.Button("CHECK UPDATES", elem_classes=["pixel-btn"])
                    maintain_best_btn = gr.Button("UPDATE BEST MODEL", elem_classes=["pixel-btn"])
                    auto_update_toggle = gr.Checkbox(
                        label="AUTO UPDATE BEST MODEL",
                        value=model_manager.get_auto_update_enabled() if model_manager else False,
                        elem_classes=["pixel-checkbox"],
                    )
                    model_status = gr.HTML(
                        get_model_status_html("qwen", force=False)
                    )
                    task_mode = gr.Dropdown(
                        choices=TASK_CHOICES,
                        value="clone",
                        label="TASK MODE",
                        interactive=True,
                        elem_classes=["pixel-input"],
                    )
                    objective_mode = gr.Dropdown(
                        choices=OBJECTIVE_CHOICES,
                        value="quality",
                        label="ROUTING OBJECTIVE",
                        interactive=True,
                        elem_classes=["pixel-input"],
                    )
                    use_task_routing = gr.Checkbox(
                        label="AUTO ROUTE MODEL FOR TASK",
                        value=True,
                        elem_classes=["pixel-checkbox"],
                    )
                    auto_download_routing = gr.Checkbox(
                        label="AUTO DOWNLOAD IF MISSING",
                        value=False,
                        elem_classes=["pixel-checkbox"],
                    )
                    apply_route_btn = gr.Button("APPLY ROUTE TO SELECTOR", elem_classes=["pixel-btn"])
                    route_preview = gr.HTML(initial_route_preview)
                    gr.HTML('<div class="flow-note">流程: 先选 TASK MODE → 看候选链路 → APPLY ROUTE → GENERATE</div>')

                with gr.Group(elem_classes=["pixel-card"]):
                    gr.HTML('<div class="pixel-section-title">► TASK INPUTS</div>')
                    with gr.Group(visible=False) as custom_inputs_group:
                        custom_speaker = gr.Dropdown(
                            choices=SPEAKER_CHOICES,
                            value=DEFAULT_CUSTOM_SPEAKER,
                            label="CUSTOM SPEAKER",
                            interactive=True,
                            elem_classes=["pixel-input"],
                        )
                        custom_instruction = gr.Textbox(
                            label="CUSTOM INSTRUCTION",
                            value=DEFAULT_CUSTOM_INSTRUCT,
                            lines=2,
                            elem_classes=["pixel-input"],
                        )
                    with gr.Group(visible=False) as design_inputs_group:
                        design_instruction = gr.Textbox(
                            label="DESIGN INSTRUCTION",
                            value=DEFAULT_DESIGN_INSTRUCT,
                            lines=2,
                            interactive=True,
                            elem_classes=["pixel-input"],
                        )
                    gr.HTML(
                        '<div class="flow-note">TASK INPUTS 会根据 TASK MODE 自动切换。</div>'
                    )
                
                with gr.Group(elem_classes=["pixel-card"]):
                    gr.HTML('<div class="pixel-section-title">► PARAMETERS</div>')
                    with gr.Row():
                        with gr.Column():
                            speed = gr.Slider(
                                minimum=0.8, 
                                maximum=1.5, 
                                value=1.15, 
                                step=0.05, 
                                label="SPEED",
                                info="FASTER >",
                                elem_classes=["pixel-slider"]
                            )
                        with gr.Column():
                            temperature = gr.Slider(
                                minimum=0.8, 
                                maximum=1.3, 
                                value=1.0, 
                                step=0.05, 
                                label="TEMPERATURE",
                                info="NATURAL >",
                                elem_classes=["pixel-slider"]
                            )
                
                generate_btn = gr.Button(
                    "► GENERATE AUDIO ◄", 
                    variant="primary",
                    elem_classes=["pixel-btn"]
                )
            
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["pixel-card", "output-card"]):
                    gr.HTML('<div class="pixel-section-title">► OUTPUT</div>')
                    output_audio = gr.Audio(
                        label="AUDIO PREVIEW",
                        type="filepath",
                        show_label=False,
                        elem_classes=["pixel-audio"]
                    )
                    status_msg = gr.Textbox(
                        label="STATUS",
                        interactive=False,
                        show_label=False,
                        placeholder="> WAITING FOR INPUT...",
                        elem_classes=["pixel-status"]
                    )
                    download_btn = gr.File(
                        label="DOWNLOAD FILE",
                        interactive=False,
                        elem_classes=["pixel-download"]
                    )
        
        generate_btn.click(
            fn=generate_podcast_gradio,
            inputs=[
                backend_selector,
                model_selector,
                task_mode,
                objective_mode,
                use_task_routing,
                auto_download_routing,
                text_input,
                text_file,
                ref_audio,
                ref_text_input,
                use_default_ref,
                custom_speaker,
                custom_instruction,
                design_instruction,
                speed,
                temperature,
            ],
            outputs=[output_audio, status_msg, download_btn],
            api_name="generate"
        )

        def get_route_preview(backend_value, task_value, objective_value, downloaded_only=False):
            if backend_value != "qwen":
                return wrap_route_preview_html(
                    '<div style="color: #888; font-size: 0.9rem;">VoxCPM 实验后端不参与 Qwen 模型路由</div>'
                )
            if model_manager is None:
                return wrap_route_preview_html(
                    '<div style="color: #888; font-size: 0.9rem;">模型路由不可用</div>'
                )
            routed = model_manager.route_model_for_task(
                task=task_value,
                objective=objective_value,
                downloaded_only=downloaded_only,
            )
            return wrap_route_preview_html(format_route_preview_html(routed))

        def refresh_models_view(backend_value="qwen", task_value="clone", objective_value="quality", force_check=False, note=""):
            models_now = get_available_models(backend_value)
            choices_now = build_model_choices(models_now)
            default_now = get_default_model_path(backend_value)
            if not any(value == default_now for _, value in choices_now):
                default_now = choices_now[0][1] if choices_now else MODEL_PATH
            status_now = get_model_status_html(backend_value, force=force_check, extra_note=note)
            route_now = get_route_preview(backend_value, task_value, objective_value, downloaded_only=False)
            return gr.update(choices=choices_now, value=default_now), status_now, route_now

        def on_refresh_models(backend_value, task_value, objective_value):
            return refresh_models_view(
                backend_value=backend_value,
                task_value=task_value,
                objective_value=objective_value,
                force_check=False,
                note="模型列表已刷新",
            )

        def on_check_updates(backend_value, task_value, objective_value):
            if backend_value != "qwen" or model_manager is None:
                return refresh_models_view(
                    backend_value=backend_value,
                    task_value=task_value,
                    objective_value=objective_value,
                    force_check=False,
                    note="当前环境不支持远程更新检查",
                )
            result = model_manager.check_all_updates(force=True)
            return refresh_models_view(
                backend_value=backend_value,
                task_value=task_value,
                objective_value=objective_value,
                force_check=False,
                note=result.get("summary", "更新检查完成"),
            )

        def on_maintain_best_model(backend_value, task_value, objective_value, auto_download):
            if backend_value != "qwen" or model_manager is None:
                return refresh_models_view(
                    backend_value=backend_value,
                    task_value=task_value,
                    objective_value=objective_value,
                    force_check=False,
                    note="当前环境不支持自动维护最佳模型",
                )
            success, message = model_manager.auto_maintain_best_model(
                objective=objective_value,
                force_check=True,
                allow_download=auto_download,
            )
            prefix = "完成" if success else "失败"
            return refresh_models_view(
                backend_value=backend_value,
                task_value=task_value,
                objective_value=objective_value,
                force_check=False,
                note=f"{prefix}: {message}",
            )

        def on_toggle_auto_update(enabled, backend_value, task_value, objective_value):
            if backend_value != "qwen" or model_manager is None:
                status_now = get_model_status_html(backend_value, force=False, extra_note="当前环境不支持自动更新设置")
                route_now = get_route_preview(backend_value, task_value, objective_value, downloaded_only=False)
                return status_now, route_now
            model_manager.set_auto_update_enabled(bool(enabled))
            state = "已开启" if enabled else "已关闭"
            status_now = get_model_status_html(backend_value, force=False, extra_note=f"自动更新{state}")
            route_now = get_route_preview(backend_value, task_value, objective_value, downloaded_only=False)
            return status_now, route_now

        def on_apply_route(backend_value, task_value, objective_value, auto_download):
            if backend_value != "qwen" or model_manager is None:
                return (
                    gr.update(),
                    get_route_preview(backend_value, task_value, objective_value, downloaded_only=False),
                    get_model_status_html(backend_value, force=False, extra_note="当前后端不支持模型路由"),
                )

            routed = model_manager.ensure_model_for_task(
                task=task_value,
                objective=objective_value,
                auto_download=auto_download,
            )
            models_now = get_available_models(backend_value)
            choices_now = build_model_choices(models_now)
            selected_value = None
            if routed.selected and routed.selected.local_path:
                candidate = str(routed.selected.local_path)
                if any(value == candidate for _, value in choices_now):
                    selected_value = candidate

            if selected_value is None:
                selected_value = get_default_model_path(backend_value)
                if not any(value == selected_value for _, value in choices_now):
                    selected_value = choices_now[0][1] if choices_now else MODEL_PATH

            selector_update = gr.update(choices=choices_now, value=selected_value)
            route_now = wrap_route_preview_html(format_route_preview_html(routed))
            status_now = get_model_status_html(backend_value, force=False, extra_note=routed.message)
            return selector_update, route_now, status_now

        def on_task_mode_change(backend_value, task_value, use_default, objective_value):
            supported_tasks = [value for _, value in supported_task_choices(backend_value)]
            normalized_task = task_value if task_value in supported_tasks else supported_tasks[0]
            clone_task = normalized_task == "clone"
            custom_task = normalized_task == "custom"
            design_task = normalized_task == "design"

            ref_interactive = clone_task and (not use_default)
            if ref_interactive:
                ref_audio_update = gr.update(interactive=True)
                ref_text_update = gr.update(interactive=True)
            else:
                ref_audio_update = gr.update(interactive=False, value=None)
                ref_text_update = gr.update(interactive=False, value="")

            route_now = get_route_preview(backend_value, normalized_task, objective_value, downloaded_only=False)
            return (
                gr.update(choices=supported_task_choices(backend_value), value=normalized_task),
                gr.update(visible=clone_task),
                gr.update(interactive=clone_task),
                ref_audio_update,
                ref_text_update,
                gr.update(visible=custom_task),
                gr.update(interactive=custom_task),
                gr.update(interactive=custom_task),
                gr.update(visible=design_task),
                gr.update(interactive=design_task),
                route_now,
            )

        def on_objective_change(backend_value, task_value, objective_value):
            return get_route_preview(backend_value, task_value, objective_value, downloaded_only=False)

        def on_backend_change(backend_value, task_value, objective_value, use_default):
            model_update, status_now, route_now = refresh_models_view(
                backend_value=backend_value,
                task_value=task_value,
                objective_value=objective_value,
                force_check=False,
                note=backend_note(backend_value),
            )
            task_updates = on_task_mode_change(backend_value, task_value, use_default, objective_value)
            return (model_update, status_now) + task_updates

        refresh_models_btn.click(
            fn=on_refresh_models,
            inputs=[backend_selector, task_mode, objective_mode],
            outputs=[model_selector, model_status, route_preview]
        )

        check_updates_btn.click(
            fn=on_check_updates,
            inputs=[backend_selector, task_mode, objective_mode],
            outputs=[model_selector, model_status, route_preview]
        )

        maintain_best_btn.click(
            fn=on_maintain_best_model,
            inputs=[backend_selector, task_mode, objective_mode, auto_download_routing],
            outputs=[model_selector, model_status, route_preview]
        )

        apply_route_btn.click(
            fn=on_apply_route,
            inputs=[backend_selector, task_mode, objective_mode, auto_download_routing],
            outputs=[model_selector, route_preview, model_status]
        )

        auto_update_toggle.change(
            fn=on_toggle_auto_update,
            inputs=[auto_update_toggle, backend_selector, task_mode, objective_mode],
            outputs=[model_status, route_preview]
        )
        
        def toggle_ref_options(use_default, task_value):
            if use_default or task_value != "clone":
                return (
                    gr.update(interactive=False, value=None),
                    gr.update(interactive=False, value=""),
                )
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        use_default_ref.change(
            fn=toggle_ref_options,
            inputs=[use_default_ref, task_mode],
            outputs=[ref_audio, ref_text_input]
        )

        task_mode.change(
            fn=on_task_mode_change,
            inputs=[backend_selector, task_mode, use_default_ref, objective_mode],
            outputs=[
                task_mode,
                clone_ref_group,
                use_default_ref,
                ref_audio,
                ref_text_input,
                custom_inputs_group,
                custom_speaker,
                custom_instruction,
                design_inputs_group,
                design_instruction,
                route_preview,
            ],
        )

        objective_mode.change(
            fn=on_objective_change,
            inputs=[backend_selector, task_mode, objective_mode],
            outputs=[route_preview],
        )

        demo.load(
            fn=on_refresh_models,
            inputs=[backend_selector, task_mode, objective_mode],
            outputs=[model_selector, model_status, route_preview]
        )

        demo.load(
            fn=on_task_mode_change,
            inputs=[backend_selector, task_mode, use_default_ref, objective_mode],
            outputs=[
                task_mode,
                clone_ref_group,
                use_default_ref,
                ref_audio,
                ref_text_input,
                custom_inputs_group,
                custom_speaker,
                custom_instruction,
                design_inputs_group,
                design_instruction,
                route_preview,
            ],
        )

        backend_selector.change(
            fn=on_backend_change,
            inputs=[backend_selector, task_mode, objective_mode, use_default_ref],
            outputs=[
                model_selector,
                model_status,
                task_mode,
                clone_ref_group,
                use_default_ref,
                ref_audio,
                ref_text_input,
                custom_inputs_group,
                custom_speaker,
                custom_instruction,
                design_inputs_group,
                design_instruction,
                route_preview,
            ],
        )
    
    return demo

def main():
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    demo = create_interface()
    
    # 启动界面
    print("启动播客音频生成工具 Web 界面...")
    print("访问地址: http://127.0.0.1:7860")
    print("按 Ctrl+C 停止服务")
    
    demo.launch(share=False)

if __name__ == "__main__":
    main()
