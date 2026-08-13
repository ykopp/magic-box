import unittest
from types import SimpleNamespace

import numpy as np

from tts_backends import TTSGenerationResult, generate_voxcpm_chunk


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
        # MLX VoxCPM2 design API: text is passed as-is, instruction via `instruct` kwarg
        self.assertEqual(model.calls[0]["text"], "hello")
        self.assertEqual(model.calls[0]["instruct"], "calm voice")

    def test_missing_model_sample_rate_raises_clear_error(self):
        # Silently defaulting to SAMPLE_RATE=24000 when the model is actually
        # 48 kHz (VoxCPM2) would cause the output to be tagged with the wrong
        # rate and play back at half speed. The new behaviour is to fail
        # loudly so the caller (or test) can plug in the correct detection.
        with self.assertRaises(ValueError) as ctx:
            self._generate(FakeVoxCPMModel())
        # The error message is bilingual (zh/en) — match either token.
        message = str(ctx.exception)
        self.assertTrue(
            "sample rate" in message.lower() or "采样率" in message,
            f"Expected error to mention sample rate, got: {message!r}",
        )


if __name__ == "__main__":
    unittest.main()
