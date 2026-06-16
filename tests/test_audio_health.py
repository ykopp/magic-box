import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import (
    SAMPLE_RATE,
    check_audio_health,
    convert_audio_if_needed,
    limit_audio_peak,
    measure_audio,
    smooth_join_audio,
    validate_generated_audio,
)


class AudioPreparationTest(unittest.TestCase):
    def test_convert_audio_fast_paths_valid_24k_mono_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "valid.wav"
            sf.write(wav_path, np.zeros(SAMPLE_RATE // 10, dtype=np.float32), SAMPLE_RATE)

            result = convert_audio_if_needed(str(wav_path))

            self.assertEqual(result, str(wav_path))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for audio conversion")
    def test_convert_audio_resamples_48k_wav_to_24k_mono_pcm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "source_48k.wav"
            sf.write(wav_path, np.zeros(48000 // 10, dtype=np.float32), 48000)

            result = convert_audio_if_needed(str(wav_path))
            self.assertIsNotNone(result)
            self.assertNotEqual(result, str(wav_path))

            try:
                with wave.open(result, "rb") as converted:
                    self.assertEqual(converted.getframerate(), SAMPLE_RATE)
                    self.assertEqual(converted.getnchannels(), 1)
                    self.assertEqual(converted.getsampwidth(), 2)
                self.assertIn(f"{os.sep}runtime{os.sep}", result)
            finally:
                if result and os.path.exists(result):
                    os.remove(result)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for audio conversion")
    def test_convert_audio_downmixes_stereo_wav_to_mono(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "stereo.wav"
            audio = np.zeros((SAMPLE_RATE // 10, 2), dtype=np.float32)
            sf.write(wav_path, audio, SAMPLE_RATE)

            result = convert_audio_if_needed(str(wav_path))
            self.assertIsNotNone(result)
            self.assertNotEqual(result, str(wav_path))

            try:
                with wave.open(result, "rb") as converted:
                    self.assertEqual(converted.getframerate(), SAMPLE_RATE)
                    self.assertEqual(converted.getnchannels(), 1)
            finally:
                if result and os.path.exists(result):
                    os.remove(result)


class AudioHealthTest(unittest.TestCase):
    def test_audio_health_catches_empty_audio(self):
        warnings = check_audio_health(np.array([], dtype=np.float32), SAMPLE_RATE)

        self.assertTrue(any("empty audio" in warning for warning in warnings))

    def test_audio_health_catches_nan_or_inf(self):
        warnings = check_audio_health(np.array([0.0, np.nan, np.inf], dtype=np.float32), SAMPLE_RATE)

        self.assertTrue(any("NaN or Inf" in warning for warning in warnings))

    def test_audio_health_catches_low_rms(self):
        warnings = check_audio_health(np.full(1000, 1e-6, dtype=np.float32), SAMPLE_RATE)

        self.assertTrue(any("very low RMS" in warning for warning in warnings))

    def test_limit_audio_peak_scales_without_hard_clipping(self):
        audio = np.array([-2.0, -0.5, 0.5, 1.0], dtype=np.float32)

        limited = limit_audio_peak(audio, ceiling=0.95)

        self.assertAlmostEqual(float(np.max(np.abs(limited))), 0.95, places=6)
        self.assertAlmostEqual(float(limited[1]), -0.2375, places=6)

    def test_limited_audio_does_not_trigger_near_clipping_warning(self):
        limited = limit_audio_peak(np.array([-2.0, 1.0], dtype=np.float32), ceiling=0.95)

        warnings = check_audio_health(limited, SAMPLE_RATE)

        self.assertFalse(any("near clipping" in warning for warning in warnings))

    def test_measure_audio_reports_adjacent_jump_metrics(self):
        audio = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)

        metrics = measure_audio(audio, SAMPLE_RATE)

        self.assertAlmostEqual(metrics.duration_seconds, 4 / SAMPLE_RATE)
        self.assertEqual(metrics.sample_rate, SAMPLE_RATE)
        self.assertAlmostEqual(metrics.peak, 0.5)
        self.assertAlmostEqual(metrics.max_adjacent_jump, 1.0)
        self.assertGreater(metrics.p999_adjacent_jump, 0.0)

    def test_validate_generated_audio_rejects_bad_chunks(self):
        warnings = validate_generated_audio(np.array([], dtype=np.float32), SAMPLE_RATE, label="chunk")
        self.assertTrue(any("empty audio" in warning for warning in warnings))

        warnings = validate_generated_audio(np.array([0.0, np.nan], dtype=np.float32), SAMPLE_RATE)
        self.assertTrue(any("NaN or Inf" in warning for warning in warnings))

        warnings = validate_generated_audio(np.zeros(SAMPLE_RATE // 100, dtype=np.float32), SAMPLE_RATE)
        self.assertTrue(any("too short" in warning for warning in warnings))
        self.assertTrue(any("very low RMS" in warning for warning in warnings))
        self.assertTrue(any("very low peak" in warning for warning in warnings))

        warnings = validate_generated_audio(np.full(SAMPLE_RATE // 10, 0.99, dtype=np.float32), SAMPLE_RATE)
        self.assertTrue(any("near clipping" in warning for warning in warnings))

        warnings = validate_generated_audio(np.ones(SAMPLE_RATE // 10, dtype=np.float32) * 1.2, SAMPLE_RATE)
        self.assertTrue(any("exceeds full scale" in warning for warning in warnings))

        warnings = validate_generated_audio(np.ones(SAMPLE_RATE // 10, dtype=np.float32), 0)
        self.assertTrue(any("invalid sample rate" in warning for warning in warnings))

    def test_smooth_join_audio_fades_segments_and_inserts_silence_without_mutating_inputs(self):
        first = np.ones(1000, dtype=np.float32)
        second = np.ones((1000, 2), dtype=np.float32) * 0.5
        first_before = first.copy()
        second_before = second.copy()

        joined = smooth_join_audio([first, second], SAMPLE_RATE, pause_seconds=0.01, fade_ms=10)

        self.assertEqual(joined.ndim, 1)
        self.assertEqual(joined.dtype, np.float32)
        self.assertEqual(joined.size, 1000 + int(SAMPLE_RATE * 0.01) + 1000)
        self.assertAlmostEqual(float(joined[0]), 0.0)
        self.assertAlmostEqual(float(joined[999]), 0.0)
        pause = joined[1000:1000 + int(SAMPLE_RATE * 0.01)]
        self.assertTrue(np.allclose(pause, 0.0))
        self.assertTrue(np.array_equal(first, first_before))
        self.assertTrue(np.array_equal(second, second_before))


if __name__ == "__main__":
    unittest.main()
