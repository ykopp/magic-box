import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import SAMPLE_RATE, convert_wav_to_mp3


class UtilsTest(unittest.TestCase):
    def test_convert_wav_to_mp3_creates_non_empty_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample.wav"
            mp3_path = temp_path / "sample.mp3"
            audio = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
            sf.write(wav_path, audio, SAMPLE_RATE)

            result = convert_wav_to_mp3(str(wav_path), str(mp3_path))

            self.assertEqual(result, str(mp3_path))
            self.assertTrue(mp3_path.exists())
            self.assertGreater(mp3_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
