import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit_app import (
    _can_generate_with_reference_quality,
    _call_optional_reference_audit,
    _filter_model_choices_for_clone,
    _generation_failure_message,
    _get_default_choice_index,
    _is_legacy_profile,
    _load_quality_report,
    _normalise_reference_quality_report,
    _profile_has_saved_reference,
    _profile_ref_text_key,
    _quality_report_issue_lines,
    _quality_report_failed_segments,
    _quality_report_summary,
    _reference_quality_status,
    _segments_dir_from_report,
)
from tts_backends import BackendModelChoice


class StreamlitModelFilteringTest(unittest.TestCase):
    def test_qwen_clone_ui_only_keeps_base_models(self):
        choices = [
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            ),
        ]

        filtered = _filter_model_choices_for_clone("qwen", choices)

        self.assertEqual(
            [choice.value for choice in filtered],
            ["/models/Qwen3-TTS-12Hz-1.7B-Base-8bit"],
        )

    def test_voxcpm_clone_ui_keeps_all_model_choices(self):
        choices = [
            BackendModelChoice(label="VoxCPM local", value="/models/VoxCPM"),
            BackendModelChoice(label="VoxCPM remote", value="openbmb/VoxCPM2"),
        ]

        self.assertEqual(_filter_model_choices_for_clone("voxcpm", choices), choices)

    def test_default_index_uses_filtered_base_choices(self):
        choices = [
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ),
        ]
        filtered = _filter_model_choices_for_clone("qwen", choices)

        with patch("streamlit_app.get_default_model_ref", return_value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit"):
            self.assertEqual(_get_default_choice_index(filtered, "qwen"), 0)

    def test_profile_ref_text_widget_key_is_profile_specific(self):
        self.assertNotEqual(_profile_ref_text_key("thomas"), _profile_ref_text_key("vivian"))
        self.assertEqual(_profile_ref_text_key("thomas"), "voice_profile_ref_text_thomas")

    def test_saved_profile_reference_can_be_used_without_retyping(self):
        ready_profile = SimpleNamespace(
            built_in=False,
            can_generate=True,
            ref_text="saved transcript",
        )
        builtin_profile = SimpleNamespace(
            built_in=True,
            can_generate=False,
            ref_text="",
        )

        self.assertTrue(_profile_has_saved_reference(ready_profile))
        self.assertFalse(_profile_has_saved_reference(builtin_profile))

    def test_quality_report_loader_prefers_shared_report_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            final_path = output_dir / "episode.mp3"
            requested_path = output_dir / "episode.wav"
            report_path = output_dir / "quality_report.json"
            report_path.write_text(json.dumps({"summary": {"total_segments": 3}}), encoding="utf-8")

            loaded_path, report = _load_quality_report(final_path, requested_path)

            self.assertEqual(loaded_path, report_path)
            self.assertEqual(report, {"summary": {"total_segments": 3}})

    def test_quality_report_summary_and_failed_segments_are_format_tolerant(self):
        report = {
            "summary": {
                "segment_count": 3,
                "ok_segments": 2,
                "failed_count": 1,
                "status": "warning",
            },
            "segments": [
                {"index": 0, "status": "ok"},
                {"index": 1, "status": "failed"},
                {"index": 2, "warnings": ["low rms"]},
            ],
        }

        self.assertEqual(
            _quality_report_summary(report),
            {
                "total_segments": 3,
                "passed_segments": 2,
                "failed_segments": 1,
                "status": "warning",
            },
        )
        self.assertEqual(_quality_report_failed_segments(report), [2, 3])

    def test_segments_dir_uses_report_path_or_output_segments_folder(self):
        final_path = Path("/tmp/magic/outputs/episode.mp3")
        requested_path = Path("/tmp/magic/outputs/episode.wav")

        self.assertEqual(
            _segments_dir_from_report({"segments_dir": "episode_segments"}, final_path, requested_path),
            Path("/tmp/magic/outputs/episode_segments"),
        )
        self.assertEqual(
            _segments_dir_from_report(None, final_path, requested_path),
            Path("/tmp/magic/outputs/.episode_segments"),
        )

    def test_quality_report_supports_generator_chunk_issue_format(self):
        report = {
            "status": "failed",
            "chunks": [
                {"index": 0, "issues": []},
                {"index": 1, "issues": ["chunk 2/3: too quiet"]},
                {"index": 2, "issues": []},
            ],
        }

        self.assertEqual(
            _quality_report_summary(report),
            {
                "total_segments": 3,
                "passed_segments": 2,
                "failed_segments": 1,
                "status": "failed",
            },
        )
        self.assertEqual(_quality_report_failed_segments(report), [2])

    def test_quality_report_issue_lines_include_delivery_diagnostics(self):
        report = {
            "issues": ["chunk 1/1: very low RMS"],
            "final": {
                "output_diagnostics": ["final audio after loudness: 2 sample jumps > 0.5"],
                "decoded_mp3_diagnostics": ["decoded MP3: 1 sample jumps > 0.5"],
            },
        }

        self.assertEqual(
            _quality_report_issue_lines(report),
            [
                "chunk 1/1: very low RMS",
                "final audio after loudness: 2 sample jumps > 0.5",
                "decoded MP3: 1 sample jumps > 0.5",
            ],
        )

    def test_generation_failure_message_uses_quality_report_issues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_path = output_dir / "episode.mp3"
            (output_dir / "quality_report.json").write_text(
                json.dumps({"status": "failed", "issues": ["chunk 2/3: too quiet"]}),
                encoding="utf-8",
            )

            self.assertIn("chunk 2/3: too quiet", _generation_failure_message(output_path, output_path))

    def test_reference_quality_gate_does_not_affect_model_filtering(self):
        choices = [
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-Base-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            ),
            BackendModelChoice(
                label="Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit | 项目",
                value="/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            ),
        ]
        failed_status = _normalise_reference_quality_report({"ok": False, "issues": ["too quiet"]})

        self.assertFalse(_can_generate_with_reference_quality(failed_status))
        self.assertEqual(
            [choice.value for choice in _filter_model_choices_for_clone("qwen", choices)],
            ["/models/Qwen3-TTS-12Hz-1.7B-Base-8bit"],
        )

    def test_legacy_profile_is_not_generation_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)
            audio_path = profile_dir / "reference_original.wav"
            audio_path.write_bytes(b"not real audio")
            (profile_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}),
                encoding="utf-8",
            )
            profile = SimpleNamespace(ref_audio_path=audio_path)

            status = _reference_quality_status(
                ref_audio_path=audio_path,
                ref_text="逐字文本",
                profile=profile,
                profile_mode="选择已保存/内置 Profile",
            )

            self.assertTrue(_is_legacy_profile(profile))
            self.assertFalse(_can_generate_with_reference_quality(status))
            self.assertIn("schema v1", " ".join(status["issues"]))

    def test_failed_reference_audit_is_not_generation_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)
            audio_path = profile_dir / "reference_clean.wav"
            text_path = profile_dir / "reference_clean.txt"
            audio_path.write_bytes(b"not real audio")
            text_path.write_text("逐字文本", encoding="utf-8")
            (profile_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 2}),
                encoding="utf-8",
            )
            (profile_dir / "reference_quality.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "metrics": {"duration_seconds": 1.2, "rms": 0.003},
                        "issues": ["duration too short", "rms too low"],
                    }
                ),
                encoding="utf-8",
            )
            profile = SimpleNamespace(ref_audio_path=audio_path)

            status = _reference_quality_status(
                ref_audio_path=audio_path,
                ref_text="逐字文本",
                profile=profile,
                profile_mode="选择已保存/内置 Profile",
            )

            self.assertFalse(_is_legacy_profile(profile))
            self.assertFalse(_can_generate_with_reference_quality(status))
            self.assertEqual(status["metrics"]["duration_seconds"], 1.2)
            self.assertIn("duration too short", status["issues"])

    def test_reference_audit_calls_utils_audit_reference_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "reference.wav"
            audio_path.write_bytes(b"fake audio bytes")

            with patch(
                "utils.audit_reference_audio",
                return_value={"ok": True, "metrics": {"duration_seconds": 5.0}, "issues": []},
            ) as audit:
                report = _call_optional_reference_audit(audio_path, "逐字文本")

            self.assertEqual(report["metrics"]["duration_seconds"], 5.0)
            audit.assert_called()


if __name__ == "__main__":
    unittest.main()
