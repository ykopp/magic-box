import tempfile
import unittest
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import SAMPLE_RATE, convert_wav_to_mp3, sanitise_tts_text, split_text


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

    def test_sanitise_tts_text_removes_format_and_control_chars(self):
        dirty = "第一句\u200b\u2060。\x00第二句"

        cleaned = sanitise_tts_text(dirty)

        self.assertEqual(cleaned, "第一句。第二句")
        self.assertFalse(any(unicodedata.category(char) == "Cf" for char in cleaned))
        self.assertFalse(any(unicodedata.category(char) == "Cc" for char in cleaned))

    def test_split_text_sanitises_invisible_chars_before_chunking(self):
        chunks = split_text("第一句\u200b。\u2060第二句。", max_chars=20)

        joined = "".join(chunks)
        self.assertEqual(joined, "第一句。第二句。")
        self.assertFalse(any("\u200b" in chunk or "\u2060" in chunk for chunk in chunks))

    def test_split_text_keeps_short_story_as_one_chunk_with_large_limit(self):
        text = (
            "王羲之教子习字王献之是王羲之的第七个儿子。很小的时候，众人就对王献之的书法和绘画赞不绝口，"
            "时间长了，小献之也渐渐滋长了骄傲自满的情绪。一天，小献之问母亲：“我只要再写上三年就行了吧？”"
            "母亲摇摇头。“五年总行了吧？”母亲又摇摇头。献之急了，冲着母亲说：“那您说究竟要多长时间？”"
            "母亲说：“写完院里这十八缸水，你的字才会有筋有骨，有血有肉！”"
            "献之一咬牙又练了五年，然后把一大堆写好的字拿给父亲看，希望听到几句表扬的话。"
            "谁知，王羲之一张张看过后，却一个劲地摇头。"
        )

        chunks = split_text(text, max_chars=500)

        self.assertEqual(chunks, [text])

    def test_split_text_prefers_major_sentence_boundaries_over_commas(self):
        text = (
            "第一句有铺垫，继续解释背景，最后自然收束。"
            "第二句也有铺垫，继续解释背景，最后自然收束。"
        )

        chunks = split_text(text, max_chars=28)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk.endswith("。") for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
