import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from benchmark_tts_backends import run_case
from podcast_generator import DEFAULT_SPEED, DEFAULT_TEMPERATURE, build_arg_parser, generate_podcast
from tts_backends import TTSGenerationResult, as_tts_result


class SampleRatePropagationTests(unittest.TestCase):
    def _tone(self, sample_rate: int, seconds: float = 0.5) -> np.ndarray:
        samples = int(sample_rate * seconds)
        t = np.arange(samples, dtype=np.float32) / sample_rate
        return (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

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
                        audio=self._tone(sample_rate, seconds=0.5),
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
            self.assertFalse((temp_path / ".out_segments").exists())

    def test_generate_podcast_uses_output_scoped_segment_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "episode.wav"
            sample_rate = 24000
            sf.write(ref_path, self._tone(sample_rate, seconds=0.5), sample_rate)

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(
                        audio=self._tone(sample_rate, seconds=0.5),
                        sample_rate=sample_rate,
                    ),
                ),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target.",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=False,
                )

            self.assertEqual(result_path, str(output_path))
            self.assertFalse((temp_path / ".episode_segments").exists())
            self.assertFalse((temp_path / "_seg_0000.wav").exists())

    def test_generate_podcast_retries_bad_chunk_before_checkpointing_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            checkpoint_path = temp_path / "out.ckpt"
            sample_rate = 24000
            sf.write(ref_path, self._tone(sample_rate, seconds=0.5), sample_rate)

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    side_effect=[
                        TTSGenerationResult(
                            audio=np.zeros(sample_rate // 2, dtype=np.float32),
                            sample_rate=sample_rate,
                        ),
                        TTSGenerationResult(
                            audio=self._tone(sample_rate, seconds=0.5),
                            sample_rate=sample_rate,
                        ),
                    ],
                ) as generate_chunk,
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target.",
                    output_path=str(output_path),
                    checkpoint_path=str(checkpoint_path),
                    output_format="wav",
                    normalise=False,
                )

            self.assertEqual(result_path, str(output_path))
            self.assertEqual(generate_chunk.call_count, 2)
            self.assertTrue(output_path.exists())
            self.assertFalse(checkpoint_path.exists())

    def test_generate_podcast_stops_when_chunk_stays_bad(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            checkpoint_path = temp_path / "out.ckpt"
            sample_rate = 24000
            sf.write(ref_path, self._tone(sample_rate, seconds=0.5), sample_rate)

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(
                        audio=np.zeros(sample_rate // 2, dtype=np.float32),
                        sample_rate=sample_rate,
                    ),
                ) as generate_chunk,
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target.",
                    output_path=str(output_path),
                    checkpoint_path=str(checkpoint_path),
                    output_format="wav",
                    normalise=False,
                )

            self.assertIsNone(result_path)
            self.assertEqual(generate_chunk.call_count, 3)
            self.assertFalse(output_path.exists())
            self.assertTrue(checkpoint_path.exists())

    def test_cli_defaults_are_stability_oriented(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.speed, DEFAULT_SPEED)
        self.assertEqual(args.temperature, DEFAULT_TEMPERATURE)
        self.assertLessEqual(args.speed, 1.05)
        self.assertGreaterEqual(args.temperature, 0.7)
        self.assertLessEqual(args.temperature, 0.9)

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
