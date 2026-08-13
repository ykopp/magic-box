import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import SAMPLE_RATE
from voice_profiles import delete_profile, list_profiles, load_profile, save_profile, slugify_profile_id


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
        t = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
        audio = 0.12 * np.sin(2 * np.pi * 180 * t)
        sf.write(source, audio, SAMPLE_RATE)

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
        self.assertEqual(saved.ref_audio_path.name, "reference_clean.wav")
        self.assertEqual(saved.ref_text_path.name, "reference_clean.txt")
        self.assertTrue(saved.ref_audio_path.exists())
        self.assertFalse(saved.built_in)
        self.assertEqual(saved.schema_version, 2)
        self.assertFalse(saved.legacy)
        self.assertTrue(saved.can_generate)
        self.assertEqual(saved.status, "ready")
        self.assertTrue((self.app_dir / "voices" / "profiles" / "thomas_voice" / "reference_original.wav").exists())
        quality_path = self.app_dir / "voices" / "profiles" / "thomas_voice" / "reference_quality.json"
        self.assertTrue(quality_path.exists())
        self.assertTrue(json.loads(quality_path.read_text(encoding="utf-8"))["ok"])
        metadata_path = self.app_dir / "voices" / "profiles" / "thomas_voice" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["ref_audio_path"], "reference_clean.wav")
        self.assertEqual(metadata["ref_text_path"], "reference_clean.txt")

        profiles = list_profiles(self.app_dir)
        self.assertIn("Thomas", [profile.display_name for profile in profiles])

        self.assertTrue(delete_profile(self.app_dir, "Thomas Voice"))
        self.assertFalse((self.app_dir / "voices" / "profiles" / "thomas_voice").exists())

    def test_legacy_profile_loads_but_is_not_generation_ready(self):
        profile_root = self.app_dir / "voices" / "profiles" / "old_voice"
        profile_root.mkdir(parents=True)
        source = profile_root / "reference_clean.wav"
        t = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
        sf.write(source, 0.12 * np.sin(2 * np.pi * 180 * t), SAMPLE_RATE)
        (profile_root / "transcript.txt").write_text("old transcript", encoding="utf-8")
        (profile_root / "metadata.json").write_text(
            json.dumps(
                {
                    "id": "old_voice",
                    "display_name": "Old Voice",
                    "ref_audio_path": "voices/profiles/old_voice/reference_clean.wav",
                    "ref_text_path": "transcript.txt",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        loaded = load_profile(self.app_dir, "old_voice")

        self.assertEqual(loaded.ref_text, "old transcript")
        self.assertEqual(loaded.schema_version, 1)
        self.assertTrue(loaded.legacy)
        self.assertFalse(loaded.can_generate)
        self.assertEqual(loaded.status, "legacy")

        profiles = {profile.id: profile for profile in list_profiles(self.app_dir)}
        self.assertIn("old_voice", profiles)
        self.assertTrue(profiles["old_voice"].legacy)


if __name__ == "__main__":
    unittest.main()
