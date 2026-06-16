import unittest
from unittest.mock import patch

from streamlit_app import _filter_model_choices_for_clone, _get_default_choice_index, _load_model_cached
from tts_backends import BackendModelChoice


class StreamlitModelFilteringTest(unittest.TestCase):
    def test_qwen_clone_ui_only_keeps_base_models(self):
        choices = [
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            ),
        ]

        filtered = _filter_model_choices_for_clone("qwen", choices)

        self.assertEqual(
            [choice.value for choice in filtered],
            ["/models/Qwen3-TTS-12Hz-1.7B-Base-8bit"],
        )

    def test_voxcpm_clone_ui_keeps_all_model_choices(self):
        choices = [
            BackendModelChoice(label="VoxCPM local", value="/models/VoxCPM"),
            BackendModelChoice(label="VoxCPM remote", value="openbmb/VoxCPM2"),
        ]

        self.assertEqual(_filter_model_choices_for_clone("voxcpm", choices), choices)

    def test_default_index_uses_filtered_base_choices(self):
        choices = [
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ),
        ]
        filtered = _filter_model_choices_for_clone("qwen", choices)

        with patch("streamlit_app.get_default_model_ref", return_value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit"):
            self.assertEqual(_get_default_choice_index(filtered, "qwen"), 0)

    def test_streamlit_model_loader_validates_clone_task_at_load_time(self):
        _load_model_cached.clear()
        with patch("streamlit_app.load_backend_model", return_value=(object(), "/models/base")) as load_backend_model:
            _load_model_cached("qwen", "/models/base", "token-1")

        load_backend_model.assert_called_once_with("qwen", "/models/base", task_mode="clone")
        _load_model_cached.clear()

    def test_streamlit_generation_uses_model_cache_token(self):
        _load_model_cached.clear()
        with patch("streamlit_app.load_backend_model", return_value=(object(), "/models/base")) as load_backend_model:
            _load_model_cached("qwen", "/models/base", "token-1")
            _load_model_cached("qwen", "/models/base", "token-2")

        self.assertEqual(load_backend_model.call_count, 2)
        _load_model_cached.clear()


if __name__ == "__main__":
    unittest.main()
