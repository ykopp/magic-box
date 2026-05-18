import unittest
from unittest.mock import patch

from streamlit_app import _filter_model_choices_for_clone, _get_default_choice_index
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


if __name__ == "__main__":
    unittest.main()
