import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model_manager import HUGGINGFACE_MODELS, ModelManager
from tts_backends import (
    BackendModelChoice,
    TTSGenerationResult,
    generate_qwen_chunk,
    generate_voxcpm_chunk,
    get_default_model_ref,
    _validate_qwen_model_files,
    validate_mlx_audio_runtime,
)


class ModelRoutingTest(unittest.TestCase):
    def _manager_with_models(self, model_names):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        manager = ModelManager()
        manager.models_dir = root / "user_models"
        manager.project_models_dir = root / "project_models"
        manager.config_file = root / "config.json"
        manager.config = {}

        for model_name in model_names:
            model_dir = manager.project_models_dir / model_name
            (model_dir / "speech_tokenizer").mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"fake")
            (model_dir / "speech_tokenizer" / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "speech_tokenizer" / "model.safetensors").write_bytes(b"fake")

        self.addCleanup(temp_dir.cleanup)
        return manager

    def test_registry_contains_customvoice_and_voicedesign_models(self):
        self.assertIn("Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", HUGGINGFACE_MODELS)
        self.assertIn("Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", HUGGINGFACE_MODELS)
        self.assertIn("Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit", HUGGINGFACE_MODELS)

    def test_clone_route_prefers_downloaded_base_for_reference_audio_cloning(self):
        manager = self._manager_with_models(
            [
                "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            ]
        )

        result = manager.route_model_for_task("clone")

        self.assertEqual(result.selected.name, "Qwen3-TTS-12Hz-1.7B-Base-8bit")
        self.assertEqual(
            manager.recommend_model_for_task("clone").name,
            "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        )

    def test_custom_route_prefers_downloaded_17b_customvoice_before_base(self):
        manager = self._manager_with_models(
            [
                "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            ]
        )

        result = manager.route_model_for_task("custom")

        self.assertEqual(result.selected.name, "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
        self.assertEqual(result.route[1].name, "Qwen3-TTS-12Hz-1.7B-Base-8bit")

    def test_clone_route_falls_back_to_base_when_customvoice_is_not_downloaded(self):
        manager = self._manager_with_models(["Qwen3-TTS-12Hz-1.7B-Base-8bit"])

        result = manager.route_model_for_task("clone")

        self.assertEqual(result.selected.name, "Qwen3-TTS-12Hz-1.7B-Base-8bit")

    def test_incomplete_model_is_not_marked_downloaded(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        manager = ModelManager()
        manager.models_dir = root / "user_models"
        manager.project_models_dir = root / "project_models"
        manager.config_file = root / "config.json"
        manager.config = {}

        model_dir = manager.project_models_dir / "Qwen3-TTS-12Hz-1.7B-Base-8bit"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"fake")

        self.addCleanup(temp_dir.cleanup)
        self.assertIsNone(manager.get_model_path("Qwen3-TTS-12Hz-1.7B-Base-8bit"))

    def test_qwen_loader_rejects_missing_speech_tokenizer_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "speech_tokenizer").mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"fake")
            (model_dir / "speech_tokenizer" / "config.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "speech_tokenizer/model.safetensors"):
                _validate_qwen_model_files(str(model_dir))

    def test_runtime_guard_rejects_missing_mlx_audio(self):
        with patch("tts_backends._get_package_version", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "没有安装 mlx-audio"):
                validate_mlx_audio_runtime()

    def test_runtime_guard_rejects_old_mlx_audio(self):
        with patch("tts_backends._get_package_version", return_value="0.3.1"):
            with self.assertRaisesRegex(RuntimeError, "mlx-audio==0.3.1"):
                validate_mlx_audio_runtime()

    def test_design_route_prefers_voicedesign_then_customvoice_then_base(self):
        manager = self._manager_with_models(
            [
                "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            ]
        )

        result = manager.route_model_for_task("design")

        self.assertEqual(
            [model.name for model in result.route],
            [
                "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
                "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ],
        )

    def test_qwen_default_model_ref_prefers_base_for_reference_audio_cloning(self):
        choices = [
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            ),
        ]

        with patch("tts_backends.get_backend_model_choices", return_value=choices):
            self.assertEqual(
                get_default_model_ref("qwen"),
                "/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            )

    def test_qwen_default_model_ref_prefers_downloaded_bf16_base(self):
        choices = [
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-bf16 | 用户",
                value="/Users/liuchang/podcast_generator_models/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            ),
        ]

        with patch("tts_backends.get_backend_model_choices", return_value=choices):
            self.assertEqual(
                get_default_model_ref("qwen"),
                "/Users/liuchang/podcast_generator_models/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            )

    def test_clone_generation_rejects_non_base_model_type(self):
        class FakeModel:
            class config:
                tts_model_type = "custom_voice"

            def generate(self, **kwargs):
                raise AssertionError("generate should not be called")

        with self.assertRaisesRegex(ValueError, "不支持参考音频克隆"):
            generate_qwen_chunk(
                model=FakeModel(),
                task_mode="clone",
                chunk="hello",
                ref_audio_path="ref.wav",
                ref_text="reference",
                custom_speaker="Vivian",
                custom_instruction="Normal tone",
                design_instruction="",
                speed=1.0,
                temperature=1.0,
            )

    def test_qwen_generation_concatenates_multiple_backend_results(self):
        class FakeModel:
            class config:
                tts_model_type = "base"

            def generate(self, **kwargs):
                return [
                    TTSGenerationResult(np.array([0.1, 0.2], dtype=np.float32), sample_rate=24000),
                    TTSGenerationResult(np.array([0.3], dtype=np.float32), sample_rate=24000),
                ]

        result = generate_qwen_chunk(
            model=FakeModel(),
            task_mode="clone",
            chunk="hello",
            ref_audio_path="ref.wav",
            ref_text="reference",
            custom_speaker="Vivian",
            custom_instruction="Normal tone",
            design_instruction="",
            speed=1.0,
            temperature=1.0,
        )

        self.assertEqual(result.sample_rate, 24000)
        np.testing.assert_allclose(result.audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    def test_voxcpm_generation_concatenates_multiple_backend_results(self):
        class FakeModel:
            sample_rate = 48000

            def generate(self, **kwargs):
                yield TTSGenerationResult(np.array([0.1], dtype=np.float32), sample_rate=48000)
                yield TTSGenerationResult(np.array([0.2, 0.3], dtype=np.float32), sample_rate=48000)

        result = generate_voxcpm_chunk(
            model=FakeModel(),
            task_mode="clone",
            chunk="hello",
            ref_audio_path="ref.wav",
            ref_text="reference",
            design_instruction="",
        )

        self.assertEqual(result.sample_rate, 48000)
        np.testing.assert_allclose(result.audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
