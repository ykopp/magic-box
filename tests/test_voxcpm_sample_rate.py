import unittest
from types import SimpleNamespace

import numpy as np

from tts_backends import TTSGenerationResult, generate_voxcpm_chunk
from utils import SAMPLE_RATE


class FakeVoxCPMModel:
    def __init__(self, audio=None, **attrs):
        self.audio = audio if audio is not None else np.zeros(12, dtype=np.float32)
        self.calls = []
        for name, value in attrs.items():
            setattr(self, name, value)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.audio


class VoxCPMSampleRateTests(unittest.TestCase):
    def _generate(self, model, task_mode="clone"):
        return generate_voxcpm_chunk(
            model=model,
            task_mode=task_mode,
            chunk="hello",
            ref_audio_path="ref.wav",
            ref_text="reference",
            design_instruction="calm voice",
        )

    def test_clone_uses_direct_model_sample_rate_for_ndarray_output(self):
        result = self._generate(FakeVoxCPMModel(sample_rate=44100))

        self.assertIsInstance(result, TTSGenerationResult)
        self.assertEqual(result.sample_rate, 44100)

    def test_clone_uses_tts_model_sample_rate_for_ndarray_output(self):
        model = FakeVoxCPMModel(tts_model=SimpleNamespace(sample_rate=32000))

        result = self._generate(model)

        self.assertEqual(result.sample_rate, 32000)

    def test_clone_uses_args_audio_vae_config_sample_rate_for_ndarray_output(self):
        model = FakeVoxCPMModel(
            args=SimpleNamespace(audio_vae_config={"sample_rate": 22050})
        )

        result = self._generate(model)

        self.assertEqual(result.sample_rate, 22050)

    def test_design_uses_config_audio_vae_config_sample_rate_for_ndarray_output(self):
        model = FakeVoxCPMModel(
            config=SimpleNamespace(
                audio_vae_config=SimpleNamespace(sample_rate=48000)
            )
        )

        result = self._generate(model, task_mode="design")

        self.assertEqual(result.sample_rate, 48000)
        self.assertEqual(model.calls[0]["text"], "(calm voice)hello")

    def test_missing_model_sample_rate_falls_back_to_project_sample_rate(self):
        result = self._generate(FakeVoxCPMModel())

        self.assertEqual(result.sample_rate, SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
