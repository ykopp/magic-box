import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from benchmark_tts_backends import run_case
from podcast_generator import generate_podcast
from tts_backends import TTSGenerationResult, as_tts_result


class SampleRatePropagationTests(unittest.TestCase):
    def test_as_tts_result_preserves_model_sample_rate(self):
        raw = SimpleNamespace(audio=np.zeros(10, dtype=np.float32), sample_rate=32000)

        result = as_tts_result(raw)

        self.assertIsInstance(result, TTSGenerationResult)
        self.assertEqual(result.sample_rate, 32000)
        np.testing.assert_array_equal(result.audio, raw.audio)

    def test_generate_podcast_writes_segments_and_final_with_backend_sample_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 32000
            sf.write(ref_path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)

            normalise_sample_rates = []

            def fake_normalise(audio, sample_rate_arg):
                normalise_sample_rates.append(sample_rate_arg)
                return audio

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(
                        audio=np.ones(sample_rate // 2, dtype=np.float32),
                        sample_rate=sample_rate,
                    ),
                ),
                patch("podcast_generator.normalise_loudness", side_effect=fake_normalise),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=True,
                )

            self.assertEqual(result_path, str(output_path))
            self.assertEqual(sf.info(output_path).samplerate, sample_rate)
            self.assertEqual(normalise_sample_rates, [sample_rate])

    def test_benchmark_uses_backend_sample_rate_for_file_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 16000
            audio = np.ones(sample_rate, dtype=np.float32)

            with patch(
                "benchmark_tts_backends.benchmark_generate_case",
                return_value=(
                    TTSGenerationResult(audio=audio, sample_rate=sample_rate),
                    {
                        "backend": "qwen",
                        "model_ref": "model",
                        "task_mode": "clone",
                        "generate_seconds": 2.0,
                    },
                ),
            ):
                metrics = run_case(
                    backend="qwen",
                    model_ref="model",
                    case={
                        "id": "case",
                        "task_mode": "clone",
                        "text": "text",
                        "notes": "notes",
                    },
                    ref_audio="ref.wav",
                    ref_text="reference",
                    output_dir=temp_path,
                )

            output_path = Path(metrics["output_path"])
            self.assertEqual(sf.info(output_path).samplerate, sample_rate)
            self.assertEqual(metrics["sample_rate"], sample_rate)
            self.assertEqual(metrics["audio_seconds"], 1.0)
            self.assertEqual(metrics["realtime_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()
