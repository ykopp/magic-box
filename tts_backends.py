import os
import sys
import time
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from utils import convert_audio_if_needed, get_smart_path, model_cache

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
QWEN_DIR = Path(__file__).resolve().parent
VOXCPM_DIR = WORKSPACE_ROOT / "VoxCPM"
VOXCPM_SRC_DIR = VOXCPM_DIR / "src"

QWEN_DEFAULT_MODEL = "models/Qwen3-TTS-12Hz-1.7B-Base-8bit"
VOXCPM_DEFAULT_MODEL = "openbmb/VoxCPM2"

QWEN_TASK_CHOICES = [
    ("Voice Clone", "clone"),
    ("Custom Voice", "custom"),
    ("Voice Design", "design"),
]
VOXCPM_TASK_CHOICES = [
    ("Voice Clone", "clone"),
    ("Voice Design", "design"),
]

QWEN_OBJECTIVE_CHOICES = [
    ("Quality", "quality"),
    ("Balanced", "balanced"),
    ("Speed", "speed"),
]


@dataclass
class BackendModelChoice:
    label: str
    value: str


def supported_task_choices(backend: str):
    return QWEN_TASK_CHOICES if backend == "qwen" else VOXCPM_TASK_CHOICES


def backend_note(backend: str) -> str:
    if backend == "voxcpm":
        return "VoxCPM 实验后端: Custom Voice / Qwen 路由不可用，speed/temperature 将被忽略。"
    return "Qwen 主线后端: 支持模型路由、speed/temperature、长文主工作流。"


def _discover_voxcpm_local_models() -> List[BackendModelChoice]:
    choices: List[BackendModelChoice] = []
    models_root = VOXCPM_DIR / "models"
    if models_root.exists():
        for model_dir in sorted(models_root.iterdir()):
            if (model_dir / "config.json").exists():
                choices.append(
                    BackendModelChoice(
                        label=f"{model_dir.name} | project",
                        value=str(model_dir),
                    )
                )
    return choices


def get_voxcpm_model_choices() -> List[BackendModelChoice]:
    choices = _discover_voxcpm_local_models()
    remote = BackendModelChoice(
        label="openbmb/VoxCPM2 | remote(default)",
        value=VOXCPM_DEFAULT_MODEL,
    )
    if not any(choice.value == remote.value for choice in choices):
        choices.insert(0, remote)
    return choices


def get_qwen_model_choices():
    from model_manager import model_manager

    choices: List[BackendModelChoice] = []
    if model_manager is not None:
        for model in model_manager.get_available_models():
            if not model.downloaded or not model.local_path:
                continue
            source = "用户" if model.source == "user" else "项目"
            label = f"{model.name} | {source} | Q{model.quality_score} S{model.speed_score}"
            if model.update_available:
                label += " | 可更新"
            choices.append(BackendModelChoice(label=label, value=str(model.local_path)))

    if not choices:
        fallback = get_smart_path(QWEN_DEFAULT_MODEL)
        if fallback:
            choices.append(BackendModelChoice(label="Default", value=fallback))
    return choices


def get_backend_model_choices(backend: str) -> List[BackendModelChoice]:
    return get_qwen_model_choices() if backend == "qwen" else get_voxcpm_model_choices()


def get_default_model_ref(backend: str) -> str:
    choices = get_backend_model_choices(backend)
    if choices:
        if backend == "qwen":
            preferred = [
                "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                "Qwen3-TTS-12Hz-0.6B-Base-8bit",
            ]
            for pref in preferred:
                for choice in choices:
                    if pref in choice.value or pref in choice.label:
                        return choice.value
        return choices[0].value
    return QWEN_DEFAULT_MODEL if backend == "qwen" else VOXCPM_DEFAULT_MODEL


def _ensure_voxcpm_import():
    missing = [
        module
        for module in ("torch", "torchaudio", "einops")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "VoxCPM experimental backend is not installed in this environment. "
            f"Missing: {', '.join(missing)}. "
            "Keep using backend=qwen, or create an isolated VoxCPM environment with "
            "`pip install -r requirements-voxcpm.txt`."
        )

    if str(VOXCPM_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(VOXCPM_SRC_DIR))
    try:
        from voxcpm import VoxCPM
    except Exception as exc:
        raise RuntimeError(
            "VoxCPM experimental backend failed to import. "
            "Use backend=qwen for production, or install VoxCPM dependencies in an isolated environment."
        ) from exc

    return VoxCPM


def _load_qwen_model(model_ref: str):
    from mlx_audio.tts.utils import load_model
    from transformers.utils import logging as transformers_logging

    smart_path = get_smart_path(model_ref) if isinstance(model_ref, str) else model_ref
    if not smart_path or not Path(smart_path).exists():
        raise FileNotFoundError(f"模型未找到: {model_ref}")

    resolved = str(Path(smart_path).resolve())
    cached = model_cache.get(resolved)
    if cached is not None:
        return cached, resolved

    original_verbosity = transformers_logging.get_verbosity()

    try:
        transformers_logging.set_verbosity_error()
        model = load_model(str(smart_path))
    finally:
        transformers_logging.set_verbosity(original_verbosity)

    model_cache.set(resolved, model)
    return model, resolved


def _load_voxcpm_model(model_ref: str):
    VoxCPM = _ensure_voxcpm_import()
    cache_key = f"voxcpm::{model_ref}"
    cached = model_cache.get(cache_key)
    if cached is not None:
        return cached, cache_key

    model = VoxCPM.from_pretrained(
        hf_model_id=model_ref,
        load_denoiser=False,
        optimize=False,
    )
    model_cache.set(cache_key, model)
    return model, cache_key


def load_backend_model(backend: str, model_ref: str):
    if backend == "qwen":
        return _load_qwen_model(model_ref)
    if backend == "voxcpm":
        return _load_voxcpm_model(model_ref)
    raise ValueError(f"Unsupported backend: {backend}")


def generate_qwen_chunk(
    model,
    task_mode: str,
    chunk: str,
    ref_audio_path: Optional[str],
    ref_text: Optional[str],
    custom_speaker: str,
    custom_instruction: str,
    design_instruction: str,
    speed: float,
    temperature: float,
) -> np.ndarray:
    if task_mode == "clone":
        results = list(
            model.generate(
                text=chunk,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                speed=speed,
                temperature=temperature,
            )
        )
    elif task_mode == "custom":
        results = list(
            model.generate(
                text=chunk,
                voice=custom_speaker,
                instruct=custom_instruction,
                speed=speed,
                temperature=temperature,
            )
        )
    elif task_mode == "design":
        results = list(
            model.generate(
                text=chunk,
                instruct=design_instruction,
                speed=speed,
                temperature=temperature,
            )
        )
    else:
        raise ValueError(f"Qwen backend does not support task: {task_mode}")

    if not results:
        raise RuntimeError("Qwen backend returned no audio")
    return np.array(results[0].audio)


def _build_voxcpm_design_text(chunk: str, design_instruction: str) -> str:
    instruction = (design_instruction or "").strip()
    return f"({instruction}){chunk}" if instruction else chunk


def generate_voxcpm_chunk(
    model,
    task_mode: str,
    chunk: str,
    ref_audio_path: Optional[str],
    ref_text: Optional[str],
    design_instruction: str,
) -> np.ndarray:
    if task_mode == "custom":
        raise ValueError("VoxCPM backend does not support the custom speaker task.")

    if task_mode == "design":
        return model.generate(
            text=_build_voxcpm_design_text(chunk, design_instruction),
            cfg_value=2.0,
            inference_timesteps=10,
        )

    if task_mode == "clone":
        kwargs = {
            "text": chunk,
            "reference_wav_path": ref_audio_path,
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        }
        if ref_text and ref_text.strip() and ref_text.strip() != ".":
            kwargs["prompt_wav_path"] = ref_audio_path
            kwargs["prompt_text"] = ref_text
        return model.generate(**kwargs)

    raise ValueError(f"VoxCPM backend does not support task: {task_mode}")


def generate_backend_chunk(
    backend: str,
    model,
    task_mode: str,
    chunk: str,
    ref_audio_path: Optional[str],
    ref_text: Optional[str],
    custom_speaker: str,
    custom_instruction: str,
    design_instruction: str,
    speed: float,
    temperature: float,
) -> np.ndarray:
    if backend == "qwen":
        return generate_qwen_chunk(
            model,
            task_mode,
            chunk,
            ref_audio_path,
            ref_text,
            custom_speaker,
            custom_instruction,
            design_instruction,
            speed,
            temperature,
        )
    if backend == "voxcpm":
        return generate_voxcpm_chunk(
            model,
            task_mode,
            chunk,
            ref_audio_path,
            ref_text,
            design_instruction,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def prepare_reference_audio(task_mode: str, ref_audio_path: Optional[str]) -> Optional[str]:
    if task_mode != "clone" or not ref_audio_path:
        return None
    return convert_audio_if_needed(ref_audio_path)


def cleanup_reference_audio(original_path: Optional[str], cleaned_path: Optional[str]):
    if cleaned_path and original_path and cleaned_path != original_path and os.path.exists(cleaned_path):
        try:
            os.remove(cleaned_path)
        except OSError:
            pass


def benchmark_generate_case(
    backend: str,
    model_ref: str,
    task_mode: str,
    text: str,
    ref_audio_path: Optional[str],
    ref_text: Optional[str],
    custom_speaker: str = "Vivian",
    custom_instruction: str = "Normal tone",
    design_instruction: str = "calm podcast narrator with clear articulation",
    speed: float = 1.15,
    temperature: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    from utils import split_text

    start = time.perf_counter()
    model, _ = load_backend_model(backend, model_ref)
    load_elapsed = time.perf_counter() - start

    chunks = split_text(text, max_chars=80)
    original_ref = ref_audio_path
    clean_ref = prepare_reference_audio(task_mode, ref_audio_path)

    audio_chunks = []
    generate_start = time.perf_counter()
    try:
        for index, chunk in enumerate(chunks, start=1):
            if progress_callback is not None:
                progress_callback(f"{backend}:{task_mode}:{index}/{len(chunks)}")
            audio_chunks.append(
                generate_backend_chunk(
                    backend=backend,
                    model=model,
                    task_mode=task_mode,
                    chunk=chunk,
                    ref_audio_path=clean_ref,
                    ref_text=ref_text,
                    custom_speaker=custom_speaker,
                    custom_instruction=custom_instruction,
                    design_instruction=design_instruction,
                    speed=speed,
                    temperature=temperature,
                )
            )
    finally:
        cleanup_reference_audio(original_ref, clean_ref)

    audio = np.concatenate(audio_chunks)
    generate_elapsed = time.perf_counter() - generate_start
    return audio, {
        "backend": backend,
        "model_ref": model_ref,
        "task_mode": task_mode,
        "chunks": len(chunks),
        "load_seconds": round(load_elapsed, 3),
        "generate_seconds": round(generate_elapsed, 3),
    }
