import tempfile
import unittest
from pathlib import Path

from voice_profiles import delete_profile, list_profiles, save_profile, slugify_profile_id


class VoiceProfilesTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app_dir = Path(self.temp_dir.name)
        (self.app_dir / "audio_samples").mkdir()
        (self.app_dir / "audio_samples" / "sample_voice_a.m4a").write_bytes(b"default")
        (self.app_dir / "audio_samples" / "sample_voice_b.m4a").write_bytes(b"other")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_slugify_profile_id_keeps_chinese_and_normalizes_symbols(self):
        self.assertEqual(slugify_profile_id(" Thomas Voice! "), "thomas_voice")
        self.assertEqual(slugify_profile_id("样例声音 ref"), "样例声音_ref")

    def test_list_profiles_exposes_builtin_audio_samples(self):
        profiles = list_profiles(self.app_dir)
        by_name = {profile.display_name: profile for profile in profiles}

        self.assertIn("sample_voice_a", by_name)
        self.assertEqual(by_name["sample_voice_a"].ref_text, "")
        self.assertTrue(by_name["sample_voice_a"].needs_transcript)
        self.assertTrue(by_name["sample_voice_b"].needs_transcript)
        self.assertTrue(by_name["sample_voice_b"].built_in)

    def test_save_and_delete_user_profile(self):
        source = self.app_dir / "sample.wav"
        source.write_bytes(b"wav")

        saved = save_profile(
            self.app_dir,
            display_name="Thomas",
            audio_source=source,
            transcript="hello profile",
            description="test voice",
            profile_id="Thomas Voice",
        )

        self.assertEqual(saved.id, "thomas_voice")
        self.assertEqual(saved.ref_text, "hello profile")
        self.assertTrue(saved.ref_audio_path.exists())
        self.assertFalse(saved.built_in)

        profiles = list_profiles(self.app_dir)
        self.assertIn("Thomas", [profile.display_name for profile in profiles])

        self.assertTrue(delete_profile(self.app_dir, "Thomas Voice"))
        self.assertFalse((self.app_dir / "voices" / "profiles" / "thomas_voice").exists())


if __name__ == "__main__":
    unittest.main()
