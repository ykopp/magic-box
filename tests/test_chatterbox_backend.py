import unittest
from unittest.mock import patch

import numpy as np

from tts_backends import TTSGenerationResult, generate_chatterbox_chunk, load_backend_model


class FakeChatterboxModel:
    sr = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.ones((1, 32), dtype=np.float32) * 0.1


FakeChatterboxModel.__module__ = "chatterbox.mtl_tts"


class ChatterboxBackendTests(unittest.TestCase):
    def test_multilingual_clone_preserves_sample_rate_and_uses_language_id(self):
        model = FakeChatterboxModel()

        result = generate_chatterbox_chunk(
            model=model,
            task_mode="clone",
            chunk="你好，欢迎收听。",
            ref_audio_path="ref.wav",
            ref_text="参考文本",
            speed=1.0,
            temperature=0.85,
        )

        self.assertIsInstance(result, TTSGenerationResult)
        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(result.audio.shape, (32,))
        self.assertEqual(model.calls[0][1]["audio_prompt_path"], "ref.wav")
        self.assertEqual(model.calls[0][1]["language_id"], "zh")

    def test_chatterbox_rejects_non_clone_task(self):
        with self.assertRaisesRegex(ValueError, "only supports voice clone"):
            generate_chatterbox_chunk(
                model=FakeChatterboxModel(),
                task_mode="design",
                chunk="hello",
                ref_audio_path="ref.wav",
                ref_text="reference",
                speed=1.0,
                temperature=0.85,
            )

    def test_missing_chatterbox_dependency_has_clear_error(self):
        with patch("importlib.util.find_spec", return_value=None), self.assertRaisesRegex(
            RuntimeError, "Chatterbox|chatterbox"
        ):
            load_backend_model("chatterbox", "chatterbox-multilingual-v3", task_mode="clone")


if __name__ == "__main__":
    unittest.main()
