import unittest

from voice_controls import get_preset, optimize_podcast_rhythm, preset_names


class VoiceControlsTest(unittest.TestCase):
    def test_preset_names_preserve_display_order(self):
        self.assertEqual(
            preset_names(),
            [
                "稳定清晰",
                "自然播客",
                "热情开场",
                "沉稳叙事",
                "快速草稿",
            ],
        )

    def test_get_preset_returns_expected_values_and_default(self):
        preset = get_preset("热情开场")
        self.assertEqual(preset.speed, 1.05)
        self.assertEqual(preset.temperature, 0.90)
        self.assertEqual(preset.chunk_max_chars, 60)

        self.assertEqual(get_preset("").name, "自然播客")
        self.assertEqual(get_preset("不存在").name, "自然播客")
        self.assertEqual(get_preset(None).name, "自然播客")

    def test_default_podcast_preset_uses_conservative_generation_values(self):
        preset = get_preset(None)

        self.assertGreaterEqual(preset.temperature, 0.70)
        self.assertLessEqual(preset.temperature, 0.90)
        self.assertGreaterEqual(preset.speed, 1.00)
        self.assertLessEqual(preset.speed, 1.05)

    def test_optimize_podcast_rhythm_normalizes_and_splits_without_new_words(self):
        text = (
            "  第一段  有很多   空格。第二句很长，包含一个铺垫，"
            "继续解释背景，说明重点，最后收束。\n\n第二段继续！  "
        )

        self.assertEqual(
            optimize_podcast_rhythm(text),
            (
                "第一段 有很多 空格。\n"
                "第二句很长，包含一个铺垫，继续解释背景，说明重点，最后收束。\n\n"
                "第二段继续！"
            ),
        )

    def test_optimize_podcast_rhythm_splits_long_comma_runs(self):
        text = (
            "这是一个很长的段落，前面先说明背景信息，中间继续补充原因，后面解释影响范围，"
            "同时补充听众需要知道的上下文，并保留原有的表达顺序，最后给出自然收尾。"
        )

        self.assertEqual(
            optimize_podcast_rhythm(text),
            (
                "这是一个很长的段落，前面先说明背景信息，中间继续补充原因，后面解释影响范围，\n"
                "同时补充听众需要知道的上下文，并保留原有的表达顺序，最后给出自然收尾。"
            ),
        )


if __name__ == "__main__":
    unittest.main()
