import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from utils import SAMPLE_RATE, convert_wav_to_mp3, get_smart_path, split_text


class UtilsTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for MP3 conversion")
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

    def test_split_text_keeps_commas_inside_short_sentences(self):
        text = "第一句，有逗号，但不应该拆开。Second sentence, with a comma."

        chunks = split_text(text, max_chars=35)

        self.assertEqual(chunks, ["第一句，有逗号，但不应该拆开。", "Second sentence, with a comma."])

    def test_split_text_uses_commas_only_for_oversized_sentence(self):
        text = "第一段很长很长，第二段也很长很长，第三段继续很长很长。"

        chunks = split_text(text, max_chars=14)

        self.assertEqual(chunks, ["第一段很长很长，", "第二段也很长很长，", "第三段继续很长很长。"])

    def test_split_text_never_splits_english_words(self):
        text = "Alpha beta gamma delta epsilon."

        chunks = split_text(text, max_chars=12)

        self.assertEqual(chunks, ["Alpha beta", "gamma delta", "epsilon."])
        self.assertTrue(all(" " not in word for chunk in chunks for word in chunk.split()))

    def test_get_smart_path_finds_app_models_when_launched_from_other_cwd(self):
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as cwd_dir:
            app_dir = Path(temp_dir) / "Magic Box"
            model_dir = app_dir / "models" / "ExampleModel"
            model_dir.mkdir(parents=True)

            with patch("utils.APP_DIR", app_dir), patch("os.getcwd", return_value=cwd_dir):
                self.assertEqual(get_smart_path("ExampleModel"), str(model_dir))

    def test_get_smart_path_uses_huggingface_refs_main_for_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_dir = Path(temp_dir) / "Magic Box"
            model_root = app_dir / "models" / "SnapshotModel"
            target_snapshot = model_root / "snapshots" / "abc123"
            older_snapshot = model_root / "snapshots" / "zzz999"
            target_snapshot.mkdir(parents=True)
            older_snapshot.mkdir(parents=True)
            (model_root / "refs").mkdir()
            (model_root / "refs" / "main").write_text("abc123", encoding="utf-8")

            with patch("utils.APP_DIR", app_dir):
                self.assertEqual(get_smart_path("SnapshotModel"), str(target_snapshot))


if __name__ == "__main__":
    unittest.main()
