import unittest

from podcast_generator import build_arg_parser
from streamlit_app import _normalise_output_path


class OutputDefaultsTest(unittest.TestCase):
    def test_cli_format_defaults_to_mp3(self):
        args = build_arg_parser().parse_args(["--text", "hello"])

        self.assertEqual(args.format, "mp3")

    def test_streamlit_requested_name_defaults_to_mp3(self):
        output_path = _normalise_output_path("hello", "episode")

        self.assertEqual(output_path.name, "episode.mp3")

    def test_streamlit_requested_name_preserves_wav_when_selected(self):
        output_path = _normalise_output_path("hello", "episode.mp3", output_format="wav")

        self.assertEqual(output_path.name, "episode.wav")

    def test_streamlit_both_uses_wav_base_for_pair_output(self):
        output_path = _normalise_output_path("hello", "episode", output_format="both")

        self.assertEqual(output_path.name, "episode.wav")


if __name__ == "__main__":
    unittest.main()
