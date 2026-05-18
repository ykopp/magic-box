import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MODELS_DIR_NAME = "podcast_generator_models"
UPDATE_CACHE_HOURS = 12
DEFAULT_AUTO_UPDATE = False
TASK_FALLBACK_CHAIN = {
    "clone": ["clone"],
    "custom": ["custom", "clone"],
    "design": ["design", "custom", "clone"],
}

HUGGINGFACE_MODELS = {
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit": {
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "size_gb": 3.08,
        "description": "CustomVoice 大模型 (1.7B) - 预置说话人和风格控制，不替代参考音频克隆",
        "recommended": True,
        "quality_score": 100,
        "speed_score": 61,
        "task": "custom",
    },
    "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit": {
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "size_gb": 1.97,
        "description": "CustomVoice 小模型 (0.6B) - 更快的预置说话人和风格控制备选",
        "recommended": True,
        "quality_score": 82,
        "speed_score": 94,
        "task": "custom",
    },
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit": {
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "size_gb": 3.08,
        "description": "VoiceDesign 大模型 (1.7B) - 推荐用于按描述设计声音",
        "recommended": True,
        "quality_score": 99,
        "speed_score": 61,
        "task": "design",
    },
    "Qwen3-TTS-12Hz-0.6B-Base-8bit": {
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "size_gb": 1.9,
        "description": "基础模型 (0.6B) - 速度快，适合日常生成",
        "recommended": True,
        "quality_score": 78,
        "speed_score": 95,
        "task": "clone",
    },
    "Qwen3-TTS-12Hz-1.7B-Base-8bit": {
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "size_gb": 2.9,
        "description": "大模型 (1.7B) - 音质最好，推荐高质量场景",
        "recommended": True,
        "quality_score": 98,
        "speed_score": 62,
        "task": "clone",
    },
}


@dataclass
class ModelInfo:
    name: str
    repo_id: str
    size_gb: float
    description: str
    recommended: bool
    downloaded: bool
    source: str
    quality_score: int
    speed_score: int
    local_path: Optional[Path] = None
    version: Optional[str] = None
    update_available: bool = False
    remote_revision: Optional[str] = None


@dataclass
class ModelRouteResult:
    task: str
    objective: str
    selected: Optional[ModelInfo]
    route: List[ModelInfo]
    message: str


def get_models_dir() -> Path:
    home_dir = Path.home()
    models_dir = home_dir / MODELS_DIR_NAME

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    except PermissionError:
        project_models = Path.cwd() / "models"
        project_models.mkdir(parents=True, exist_ok=True)
        return project_models


def get_project_models_dir() -> Path:
    return Path.cwd() / "models"


def get_config_file() -> Path:
    home_dir = Path.home()
    config_dir = home_dir / ".podcast_generator"

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.json"
    except PermissionError:
        return Path.cwd() / "podcast_config.json"


MODELS_DIR = get_models_dir()
PROJECT_MODELS_DIR = get_project_models_dir()
CONFIG_FILE = get_config_file()


class ModelManager:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.project_models_dir = PROJECT_MODELS_DIR
        self.config_file = CONFIG_FILE
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        return loaded
            except Exception:
                return {}
        return {}

    def _save_config(self):
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _is_stale(self, ts_str: Optional[str], ttl_hours: int) -> bool:
        if not ts_str:
            return True
        try:
            checked_at = time.mktime(time.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
            return (time.time() - checked_at) > ttl_hours * 3600
        except Exception:
            return True

    def _model_record(self, model_name: str) -> dict:
        return self.config.get("models", {}).get(model_name, {})

    def _set_model_record(self, model_name: str, record: dict):
        if "models" not in self.config:
            self.config["models"] = {}
        self.config["models"][model_name] = record

    def _remote_record(self, repo_id: str) -> dict:
        return self.config.get("remote_meta", {}).get(repo_id, {})

    def _set_remote_record(self, repo_id: str, record: dict):
        if "remote_meta" not in self.config:
            self.config["remote_meta"] = {}
        self.config["remote_meta"][repo_id] = record

    def _is_model_valid(self, model_path: Path) -> bool:
        required_files = [
            "model.safetensors",
            "config.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/config.json",
        ]
        return all((model_path / f).exists() for f in required_files)

    def _read_local_revision_from_cache(self, model_path: Optional[Path]) -> Optional[str]:
        if not model_path:
            return None

        candidates = [
            model_path / ".cache" / "huggingface" / "download" / "config.json.metadata",
            model_path / ".cache" / "huggingface" / "download" / "model.safetensors.metadata",
        ]
        for metadata_file in candidates:
            if not metadata_file.exists():
                continue
            try:
                first_line = metadata_file.read_text(encoding="utf-8").splitlines()[0].strip()
                if first_line:
                    return first_line
            except Exception:
                continue

        return None

    def _get_local_revision(self, model_name: str, local_path: Optional[Path]) -> Optional[str]:
        model_record = self._model_record(model_name)
        local_revision = model_record.get("revision")
        if local_revision:
            return local_revision

        return self._read_local_revision_from_cache(local_path)

    def _score_model(self, model: ModelInfo, objective: str = "quality") -> tuple:
        if objective == "speed":
            return (model.speed_score, model.quality_score, -model.size_gb, model.recommended)

        if objective == "balanced":
            return (
                0.7 * model.quality_score + 0.3 * model.speed_score,
                model.quality_score,
                -model.size_gb,
                model.recommended,
            )

        return (model.quality_score, model.speed_score, -model.size_gb, model.recommended)

    def _resolve_model_location(self, model_name: str) -> Tuple[Optional[Path], str]:
        user_path = self.models_dir / model_name
        if user_path.exists() and self._is_model_valid(user_path):
            return user_path, "user"

        project_path = self.project_models_dir / model_name
        if project_path.exists() and self._is_model_valid(project_path):
            return project_path, "project"

        return None, "none"

    def _fetch_remote_meta(
        self, repo_id: str, force: bool = False, ttl_hours: int = UPDATE_CACHE_HOURS
    ) -> Tuple[Optional[dict], Optional[str]]:
        cached = self._remote_record(repo_id)
        if cached and not force and not self._is_stale(cached.get("checked_at"), ttl_hours):
            return cached, None

        try:
            from huggingface_hub import model_info

            info = model_info(repo_id)
            remote = {
                "repo_id": repo_id,
                "revision": getattr(info, "sha", None),
                "last_modified": str(getattr(info, "last_modified", "")),
                "downloads": int(getattr(info, "downloads", 0) or 0),
                "likes": int(getattr(info, "likes", 0) or 0),
                "checked_at": self._now(),
            }
            self._set_remote_record(repo_id, remote)
            self._save_config()
            return remote, None
        except Exception as e:
            if cached:
                return cached, f"远程检查失败，使用缓存: {e}"
            return None, str(e)

    def get_available_models(self) -> List[ModelInfo]:
        models: List[ModelInfo] = []

        for name, info in HUGGINGFACE_MODELS.items():
            local_path, source = self._resolve_model_location(name)
            downloaded = local_path is not None

            model_record = self._model_record(name)
            remote_record = self._remote_record(info["repo_id"])

            local_revision = self._get_local_revision(name, local_path)
            remote_revision = remote_record.get("revision")
            update_available = bool(
                downloaded
                and local_revision
                and remote_revision
                and local_revision != remote_revision
            )

            models.append(
                ModelInfo(
                    name=name,
                    repo_id=info["repo_id"],
                    size_gb=info["size_gb"],
                    description=info["description"],
                    recommended=info.get("recommended", False),
                    downloaded=downloaded,
                    source=source,
                    quality_score=info.get("quality_score", 0),
                    speed_score=info.get("speed_score", 0),
                    local_path=local_path,
                    version=local_revision,
                    update_available=update_available,
                    remote_revision=remote_revision,
                )
            )

        return models

    def get_model_path(self, model_name: str) -> Optional[Path]:
        local_path, _ = self._resolve_model_location(model_name)
        return local_path

    def recommend_best_model(
        self, objective: str = "quality", downloaded_only: bool = True
    ) -> Optional[ModelInfo]:
        models = self.get_available_models()
        candidates = [m for m in models if m.downloaded] if downloaded_only else models
        if not candidates:
            return None

        return max(candidates, key=lambda m: self._score_model(m, objective))

    def recommend_model_for_task(
        self,
        task: str,
        objective: str = "quality",
        downloaded_only: bool = True,
    ) -> Optional[ModelInfo]:
        route = self.get_task_route(
            task=task,
            objective=objective,
            downloaded_only=downloaded_only,
        )
        return route[0] if route else None

    def get_task_route(
        self,
        task: str,
        objective: str = "quality",
        downloaded_only: bool = True,
    ) -> List[ModelInfo]:
        task = task.lower().strip()
        fallback_chain = TASK_FALLBACK_CHAIN.get(task, [task])
        all_models = self.get_available_models()
        route: List[ModelInfo] = []
        seen = set()

        for stage_task in fallback_chain:
            stage_models = [
                m
                for m in all_models
                if HUGGINGFACE_MODELS.get(m.name, {}).get("task") == stage_task
                and (m.downloaded if downloaded_only else True)
            ]
            stage_models = sorted(
                stage_models,
                key=lambda m: self._score_model(m, objective),
                reverse=True,
            )
            for model in stage_models:
                if model.name in seen:
                    continue
                route.append(model)
                seen.add(model.name)

        return route

    def route_model_for_task(
        self,
        task: str,
        objective: str = "quality",
        downloaded_only: bool = True,
    ) -> ModelRouteResult:
        route = self.get_task_route(
            task=task,
            objective=objective,
            downloaded_only=downloaded_only,
        )
        selected = route[0] if route else None
        if selected:
            message = f"已选择 {selected.name} (task={task}, objective={objective})"
        elif downloaded_only:
            message = f"没有可用已下载模型 (task={task})"
        else:
            message = f"没有可用模型配置 (task={task})"

        return ModelRouteResult(
            task=task,
            objective=objective,
            selected=selected,
            route=route,
            message=message,
        )

    def ensure_model_for_task(
        self,
        task: str,
        objective: str = "quality",
        auto_download: bool = False,
    ) -> ModelRouteResult:
        routed = self.route_model_for_task(
            task=task,
            objective=objective,
            downloaded_only=True,
        )
        if routed.selected:
            return routed

        if not auto_download:
            return routed

        candidate = self.route_model_for_task(
            task=task,
            objective=objective,
            downloaded_only=False,
        ).selected
        if not candidate:
            return ModelRouteResult(
                task=task,
                objective=objective,
                selected=None,
                route=[],
                message=f"task={task} 没有可下载模型",
            )

        success, message = self.download_model(candidate.name)
        if not success:
            return ModelRouteResult(
                task=task,
                objective=objective,
                selected=None,
                route=[],
                message=message,
            )

        refreshed = self.route_model_for_task(
            task=task,
            objective=objective,
            downloaded_only=True,
        )
        if refreshed.selected:
            refreshed.message = f"已自动下载并选择 {refreshed.selected.name}"
        return refreshed

    def download_model(
        self, model_name: str, progress_callback=None, force: bool = False
    ) -> Tuple[bool, str]:
        if model_name not in HUGGINGFACE_MODELS:
            return False, f"未知模型: {model_name}"

        model_info = HUGGINGFACE_MODELS[model_name]
        repo_id = model_info["repo_id"]
        local_path = self.models_dir / model_name

        if progress_callback:
            progress_callback(0.0, f"准备下载 {model_name}...")

        try:
            from huggingface_hub import snapshot_download

            remote, _ = self._fetch_remote_meta(repo_id, force=True, ttl_hours=0)
            expected_revision = remote.get("revision") if remote else None

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=force,
            )

            if not self._is_model_valid(local_path):
                return False, f"下载后的模型不完整: {model_name}"

            record = {
                "version": "1.0",
                "downloaded_at": self._now(),
                "repo_id": repo_id,
                "revision": expected_revision,
                "source": "user",
            }
            self._set_model_record(model_name, record)
            self._save_config()

            if progress_callback:
                progress_callback(1.0, f"下载完成: {model_name}")

            return True, f"模型 {model_name} 下载成功"

        except ImportError:
            return self._download_manual(model_name, repo_id, local_path, progress_callback)
        except Exception as e:
            return False, f"下载失败: {str(e)}"

    def _download_manual(
        self, model_name: str, repo_id: str, local_path: Path, progress_callback
    ) -> Tuple[bool, str]:
        if progress_callback:
            progress_callback(0.0, "使用备用下载方式...")

        try:
            cmd = [
                "huggingface-cli",
                "download",
                repo_id,
                "--local-dir",
                str(local_path),
                "--local-dir-use-symlinks",
                "False",
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line and progress_callback:
                    # Try to parse a percentage from e.g. "  45%|..." output
                    import re as _re
                    m = _re.search(r"(\d+)%", line)
                    pct = float(m.group(1)) / 100.0 if m else 0.5
                    progress_callback(pct, line.strip())

            if process.returncode != 0:
                return False, "下载失败，请检查网络连接"

            if not self._is_model_valid(local_path):
                return False, f"下载后的模型不完整: {model_name}"

            remote, _ = self._fetch_remote_meta(repo_id, force=True, ttl_hours=0)
            revision = remote.get("revision") if remote else None

            record = {
                "version": "1.0",
                "downloaded_at": self._now(),
                "repo_id": repo_id,
                "revision": revision,
                "source": "user",
            }
            self._set_model_record(model_name, record)
            self._save_config()
            return True, f"模型 {model_name} 下载成功"

        except FileNotFoundError:
            return False, "请先安装 huggingface_hub: pip install huggingface_hub"
        except Exception as e:
            return False, f"下载失败: {str(e)}"

    def update_model(self, model_name: str, progress_callback=None) -> Tuple[bool, str]:
        return self.download_model(model_name, progress_callback=progress_callback, force=True)

    def check_for_updates(self, model_name: str, force: bool = False) -> Tuple[bool, str]:
        if model_name not in HUGGINGFACE_MODELS:
            return False, "未知模型"

        current = next((m for m in self.get_available_models() if m.name == model_name), None)
        if not current or not current.downloaded:
            return False, "模型未下载"

        remote, error = self._fetch_remote_meta(HUGGINGFACE_MODELS[model_name]["repo_id"], force=force)
        if not remote:
            return False, f"检查更新失败: {error or '未知错误'}"

        local_revision = self._get_local_revision(model_name, current.local_path)
        remote_revision = remote.get("revision")

        if not local_revision:
            return False, "已下载（本地缺少版本记录），建议手动更新一次后启用精确检测"

        # Persist inferred revision so subsequent checks remain accurate.
        if not self._model_record(model_name).get("revision"):
            model_record = self._model_record(model_name).copy()
            model_record["revision"] = local_revision
            model_record.setdefault("repo_id", HUGGINGFACE_MODELS[model_name]["repo_id"])
            model_record.setdefault("source", current.source)
            self._set_model_record(model_name, model_record)
            self._save_config()

        if local_revision != remote_revision:
            return True, f"发现更新: {local_revision[:8]} -> {str(remote_revision)[:8]}"

        return False, "已是最新版本"

    def check_all_updates(self, force: bool = False) -> dict:
        downloaded_models = [m for m in self.get_available_models() if m.downloaded]

        updates = []
        errors = []
        for model in downloaded_models:
            has_update, message = self.check_for_updates(model.name, force=force)
            if has_update:
                updates.append({"model": model.name, "message": message})
            elif message.startswith("检查更新失败"):
                errors.append({"model": model.name, "message": message})

        summary = f"已检查 {len(downloaded_models)} 个模型"
        if updates:
            summary += f"，{len(updates)} 个可更新"
        else:
            summary += "，全部最新"

        result = {
            "checked": len(downloaded_models),
            "updates": updates,
            "errors": errors,
            "summary": summary,
            "checked_at": self._now(),
        }

        self.config["last_update_check"] = result
        self._save_config()
        return result

    def get_auto_update_enabled(self) -> bool:
        return bool(self.config.get("settings", {}).get("auto_update_best_model", DEFAULT_AUTO_UPDATE))

    def set_auto_update_enabled(self, enabled: bool):
        if "settings" not in self.config:
            self.config["settings"] = {}
        self.config["settings"]["auto_update_best_model"] = bool(enabled)
        self._save_config()

    def auto_check_updates(self, ttl_hours: int = UPDATE_CACHE_HOURS, force: bool = False) -> dict:
        last = self.config.get("last_update_check", {})
        last_checked = last.get("checked_at") if isinstance(last, dict) else None

        if not force and not self._is_stale(last_checked, ttl_hours):
            return {
                "checked": 0,
                "updates": last.get("updates", []),
                "errors": [],
                "summary": "最近已检查过更新，使用缓存结果",
                "checked_at": last_checked,
            }

        return self.check_all_updates(force=True)

    def auto_maintain_best_model(
        self,
        objective: str = "quality",
        force_check: bool = False,
        allow_download: bool = True,
    ) -> Tuple[bool, str]:
        best = self.recommend_best_model(objective=objective, downloaded_only=False)
        if not best:
            return False, "未找到可用模型"

        if not best.downloaded:
            if not allow_download:
                return False, f"最佳模型 {best.name} 未下载"
            success, message = self.download_model(best.name)
            if not success:
                return False, message
            return True, f"已下载最佳模型: {best.name}"

        has_update, message = self.check_for_updates(best.name, force=force_check)
        if not has_update:
            return True, f"最佳模型 {best.name} 无需更新"

        success, update_msg = self.update_model(best.name)
        if not success:
            return False, update_msg
        return True, f"最佳模型 {best.name} 已更新"

    def get_model_status_summary(self, objective: str = "quality", force: bool = False) -> dict:
        models = self.get_available_models()
        downloaded = [m for m in models if m.downloaded]
        best_downloaded = self.recommend_best_model(objective=objective, downloaded_only=True)
        best_overall = self.recommend_best_model(objective=objective, downloaded_only=False)

        update_result = self.check_all_updates(force=force) if force else self.auto_check_updates(force=False)

        return {
            "total_models": len(models),
            "downloaded_models": len(downloaded),
            "best_downloaded": best_downloaded.name if best_downloaded else None,
            "best_overall": best_overall.name if best_overall else None,
            "auto_update_enabled": self.get_auto_update_enabled(),
            "update_summary": update_result.get("summary", ""),
            "updates": update_result.get("updates", []),
            "checked_at": update_result.get("checked_at"),
        }

    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        local_path = self.models_dir / model_name
        if not local_path.exists():
            return False, "模型不存在"

        try:
            shutil.rmtree(local_path)

            if model_name in self.config.get("models", {}):
                del self.config["models"][model_name]
                self._save_config()

            return True, f"模型 {model_name} 已删除"
        except Exception as e:
            return False, f"删除失败: {str(e)}"

    def get_total_size(self) -> float:
        total = 0
        for model in self.get_available_models():
            if model.downloaded and model.local_path:
                for f in model.local_path.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total / (1024**3)


model_manager = ModelManager()
