import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import SAMPLE_RATE, check_audio_health, convert_audio_if_needed, limit_audio_peak


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


if __name__ == "__main__":
    unittest.main()
