"""Voice preset controls and text rhythm helpers for podcast generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class VoicePreset:
    """Immutable voice/personality settings for podcast TTS."""

    name: str
    speed: float
    temperature: float
    chunk_max_chars: int
    explanation: str


_PRESETS: tuple[VoicePreset, ...] = (
    VoicePreset(
        name="稳定清晰",
        speed=1.00,
        temperature=0.75,
        chunk_max_chars=260,
        explanation="适合信息密度高的内容，语速稳定、发音清楚，并减少短文切碎。",
    ),
    VoicePreset(
        name="自然播客",
        speed=1.00,
        temperature=0.85,
        chunk_max_chars=260,
        explanation="适合短文和故事稿，按大句段生成；Qwen 克隆会把 1.00 校准到参考音频语速。",
    ),
    VoicePreset(
        name="热情开场",
        speed=1.05,
        temperature=0.85,
        chunk_max_chars=260,
        explanation="适合开场、预告和重点段落，在稳定范围内略微增强表达。",
    ),
    VoicePreset(
        name="沉稳叙事",
        speed=0.95,
        temperature=0.80,
        chunk_max_chars=260,
        explanation="适合故事、复盘和解释型内容，节奏更从容。",
    ),
    VoicePreset(
        name="快速草稿",
        speed=1.05,
        temperature=0.70,
        chunk_max_chars=140,
        explanation="适合快速试听草稿，保持较短切分和低随机性以便排查坏段。",
    ),
)

_PRESET_BY_NAME: Mapping[str, VoicePreset] = {preset.name: preset for preset in _PRESETS}
_DEFAULT_PRESET_NAME = "自然播客"


def preset_names() -> list[str]:
    """Return preset names in display order."""

    return [preset.name for preset in _PRESETS]


def get_preset(name: str | None) -> VoicePreset:
    """Return a preset by name, falling back to the default podcast preset."""

    if not name:
        return _PRESET_BY_NAME[_DEFAULT_PRESET_NAME]
    return _PRESET_BY_NAME.get(name.strip(), _PRESET_BY_NAME[_DEFAULT_PRESET_NAME])


def optimize_podcast_rhythm(text: str) -> str:
    """Normalize text pacing for TTS without changing the wording or meaning."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    paragraphs = re.split(r"\n\s*\n+", normalized)
    optimized_paragraphs: list[str] = []

    for paragraph in paragraphs:
        paragraph = _normalize_paragraph_spacing(paragraph)
        if not paragraph:
            continue

        major_lines = _split_major_sentence_punctuation(paragraph)
        paced_lines: list[str] = []
        for line in major_lines:
            paced_lines.extend(_split_long_comma_run(line))

        optimized_paragraphs.append("\n".join(line for line in paced_lines if line))

    return "\n\n".join(optimized_paragraphs)


def _normalize_paragraph_spacing(paragraph: str) -> str:
    paragraph = re.sub(r"\s+", " ", paragraph.strip())
    paragraph = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", paragraph)
    paragraph = re.sub(r"([（(])\s+", r"\1", paragraph)
    paragraph = re.sub(r"\s+([）)])", r"\1", paragraph)
    return paragraph


def _split_major_sentence_punctuation(paragraph: str) -> list[str]:
    paragraph = re.sub(r"([。！？；!?;])\s*", r"\1\n", paragraph)
    paragraph = re.sub(r"(?<!\d)(\.)\s+", r"\1\n", paragraph)
    return [line.strip() for line in paragraph.split("\n") if line.strip()]


def _split_long_comma_run(line: str, max_line_chars: int = 70, min_split_chars: int = 34) -> list[str]:
    if len(line) <= max_line_chars or not re.search(r"[，,]", line):
        return [line]

    parts = re.split(r"([，,])", line)
    segments: list[str] = []
    current = ""

    for index in range(0, len(parts), 2):
        phrase = parts[index]
        comma = parts[index + 1] if index + 1 < len(parts) else ""
        token = phrase + comma
        if not token:
            continue

        current += token
        has_more = index + 2 < len(parts)
        if has_more and comma and len(current) >= min_split_chars:
            segments.append(current.strip())
            current = ""

    if current.strip():
        segments.append(current.strip())

    return segments or [line]
