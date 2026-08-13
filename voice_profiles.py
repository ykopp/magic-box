"""Local voice clone profile management."""

from __future__ import annotations

import json
import re
import shutil
from hashlib import sha256
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from utils import prepare_reference_pair


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    display_name: str
    ref_audio_path: Path
    ref_text_path: Path | None = None
    ref_text: str = ""
    description: str = ""
    default_preset: str = "自然播客"
    created_at: str = ""
    updated_at: str = ""
    built_in: bool = False
    schema_version: int = 1
    legacy: bool = False
    can_generate: bool = True
    status: str = "ready"

    @property
    def needs_transcript(self) -> bool:
        return not self.ref_text.strip()

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint that survives path changes and partial edits.

        Includes the profile id, the ref-audio content hash (NOT the path),
        the audio size, and the ref-text digest.  This makes checkpoint
        reuse portable across machines with the same audio file.
        """
        stat = None
        try:
            stat = self.ref_audio_path.stat() if self.ref_audio_path.exists() else None
        except OSError:
            stat = None
        size = stat.st_size if stat else 0

        audio_digest = _sha256_file(self.ref_audio_path) if stat else sha256(b"").hexdigest()
        text_digest = sha256(self.ref_text.strip().encode("utf-8")).hexdigest()
        return f"{self.id}:{audio_digest}:{size}:{text_digest}"


def text_fingerprint(text: str) -> str:
    return sha256(text.strip().encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it all into memory."""
    hasher = sha256()
    try:
        with open(path, "rb") as fp:
            while True:
                chunk = fp.read(chunk_bytes)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return sha256(b"").hexdigest()
    return hasher.hexdigest()


def slugify_profile_id(name: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_").lower()
    return slug or "voice_profile"


def profiles_dir(app_dir: Path) -> Path:
    return app_dir / "voices" / "profiles"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _metadata_path(app_dir: Path, profile_id: str) -> Path:
    return profiles_dir(app_dir) / profile_id / "metadata.json"


def _resolve_profile_path(app_dir: Path, profile_path: Path, value: str | None, fallback_name: str) -> Path:
    raw = value or fallback_name
    path = Path(raw)
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return profile_path / path
    return app_dir / path


def _audio_candidates(profile_path: Path) -> Iterable[Path]:
    for path in sorted(profile_path.iterdir()):
        if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            yield path


def load_profile(app_dir: Path, profile_id: str) -> VoiceProfile:
    metadata_path = _metadata_path(app_dir, profile_id)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    profile_path = metadata_path.parent
    schema_version = int(data.get("schema_version", 1) or 1)
    ref_audio_path = _resolve_profile_path(
        app_dir,
        profile_path,
        data.get("ref_audio_path"),
        "reference_clean.wav",
    )
    ref_text_path = _resolve_profile_path(
        app_dir,
        profile_path,
        data.get("ref_text_path"),
        "transcript.txt",
    )
    legacy = schema_version < 2
    can_generate = not legacy and ref_audio_path.exists() and ref_text_path.exists()
    status = "legacy" if legacy else ("ready" if can_generate else "incomplete")
    return VoiceProfile(
        id=data["id"],
        display_name=data.get("display_name") or data["id"],
        ref_audio_path=ref_audio_path,
        ref_text_path=ref_text_path,
        ref_text=_read_text(ref_text_path),
        description=data.get("description", ""),
        default_preset=data.get("default_preset", "自然播客"),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        built_in=False,
        schema_version=schema_version,
        legacy=legacy,
        can_generate=can_generate,
        status=status,
    )


def list_profiles(app_dir: Path) -> list[VoiceProfile]:
    profiles = _built_in_profiles(app_dir)
    root = profiles_dir(app_dir)
    if root.exists():
        for metadata_path in sorted(root.glob("*/metadata.json")):
            try:
                profiles.append(load_profile(app_dir, metadata_path.parent.name))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    return profiles


def save_profile(
    app_dir: Path,
    display_name: str,
    audio_source: Path,
    transcript: str,
    description: str = "",
    default_preset: str = "自然播客",
    profile_id: str | None = None,
) -> VoiceProfile:
    profile_id = slugify_profile_id(profile_id or display_name)
    root = profiles_dir(app_dir) / profile_id
    root.mkdir(parents=True, exist_ok=True)

    original_target = root / f"reference_original{audio_source.suffix.lower()}"
    if audio_source.resolve() != original_target.resolve():
        shutil.copy2(audio_source, original_target)
    reference_pair = prepare_reference_pair(original_target, transcript, root)

    existing_created = ""
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            existing_created = json.loads(metadata_path.read_text(encoding="utf-8")).get("created_at", "")
        except json.JSONDecodeError:
            existing_created = ""

    metadata = {
        "schema_version": 2,
        "id": profile_id,
        "display_name": display_name.strip() or profile_id,
        "ref_audio_path": reference_pair.clean_audio_path.name,
        "ref_text_path": reference_pair.clean_text_path.name,
        "ref_quality_path": reference_pair.quality_path.name,
        "description": description.strip(),
        "default_preset": default_preset,
        "created_at": existing_created or _now(),
        "updated_at": _now(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_profile(app_dir, profile_id)


def delete_profile(app_dir: Path, profile_id: str) -> bool:
    root = profiles_dir(app_dir) / slugify_profile_id(profile_id)
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True


def _built_in_profiles(app_dir: Path) -> list[VoiceProfile]:
    audio_root = app_dir / "audio_samples"
    if not audio_root.exists():
        return []

    profiles: list[VoiceProfile] = []
    audio_paths = sorted(audio_root.iterdir(), key=lambda path: path.name)
    for audio_path in audio_paths:
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        display_name = audio_path.stem
        profiles.append(
            VoiceProfile(
                id=f"builtin_{slugify_profile_id(display_name)}",
                display_name=display_name,
                ref_audio_path=audio_path,
                ref_text_path=None,
                ref_text="",
                description="本机源声音，请补全对应参考文本后使用",
                built_in=True,
                schema_version=0,
                legacy=False,
                can_generate=False,
                status="builtin_needs_transcript",
            )
        )
    return profiles
