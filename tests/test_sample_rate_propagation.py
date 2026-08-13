import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from benchmark_tts_backends import run_case
from podcast_generator import generate_podcast
from tts_backends import TTSGenerationResult, as_tts_result


class SampleRatePropagationTests(unittest.TestCase):
    def test_as_tts_result_preserves_model_sample_rate(self):
        raw = SimpleNamespace(audio=np.zeros(10, dtype=np.float32), sample_rate=32000)

        result = as_tts_result(raw)

        self.assertIsInstance(result, TTSGenerationResult)
        self.assertEqual(result.sample_rate, 32000)
        np.testing.assert_array_equal(result.audio, raw.audio)

    def test_generate_podcast_writes_segments_and_final_with_backend_sample_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 32000
            sf.write(ref_path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)

            normalise_sample_rates = []

            def fake_normalise(audio, sample_rate_arg):
                normalise_sample_rates.append(sample_rate_arg)
                return audio

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(
                        audio=np.full(sample_rate // 2, 0.1, dtype=np.float32),
                        sample_rate=sample_rate,
                    ),
                ),
                patch("podcast_generator.normalise_loudness", side_effect=fake_normalise),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=True,
                    model_ref="test-model",
                )

            self.assertEqual(result_path, str(output_path))
            self.assertEqual(sf.info(output_path).samplerate, sample_rate)
            self.assertEqual(normalise_sample_rates, [sample_rate])
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["sample_rate"], sample_rate)
            self.assertEqual(report["generation"]["model_ref"], "test-model")
            self.assertEqual(report["generation"]["checkpoint_metadata"]["model_ref"], "test-model")
            self.assertEqual(report["generation"]["checkpoint_metadata"]["backend"], "qwen")
            self.assertEqual(report["chunks"][0]["sample_rate"], sample_rate)
            self.assertTrue((temp_path / ".out_segments" / "_seg_0000.wav").exists())

    def test_generate_podcast_gates_audible_pauses_after_trim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 24000
            sf.write(ref_path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)
            t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
            speech = 0.08 * np.sin(2 * np.pi * 220 * t)
            low_pause = np.full(int(2.0 * sample_rate), 8.5e-4, dtype=np.float32)
            raw_chunk = np.concatenate([speech, low_pause, speech]).astype(np.float32)

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=raw_chunk, sample_rate=sample_rate),
                ),
                patch("podcast_generator.normalise_loudness", side_effect=lambda audio, sr: audio),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=True,
                    model_ref="test-model",
                )

            self.assertEqual(result_path, str(output_path))
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["chunks"][0]["issues"], [])
            self.assertTrue(report["chunks"][0]["pre_trim_diagnostics"])
            self.assertLess(report["final"]["metrics"]["longest_rms_low_energy_run"], 1.2)

    def test_generate_podcast_uses_prepared_reference_pair_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 24000
            sf.write(ref_path, np.full(sample_rate // 10, 0.1, dtype=np.float32), sample_rate)

            def fake_generate(**kwargs):
                self.assertEqual(kwargs["ref_audio_path"], str(ref_path))
                self.assertEqual(kwargs["ref_text"], "audited reference text")
                return TTSGenerationResult(
                    audio=np.full(sample_rate // 2, 0.1, dtype=np.float32),
                    sample_rate=sample_rate,
                )

            with (
                patch(
                    "podcast_generator.prepare_reference_audio",
                    return_value={
                        "clean_audio_path": str(ref_path),
                        "clean_ref_text": "audited reference text",
                        "audit": {"ok": True, "source": "test"},
                    },
                ),
                patch("podcast_generator.cleanup_reference_audio"),
                patch("podcast_generator.generate_backend_chunk", side_effect=fake_generate),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="raw full transcript that must not be used",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=False,
                )

            self.assertEqual(result_path, str(output_path))
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["reference"]["clean_ref_text"], "audited reference text")
            self.assertEqual(report["reference"]["audit"]["source"], "test")

    def test_generate_podcast_quality_gate_keeps_bad_segment_evidence_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "bad.wav"
            sample_rate = 24000
            sf.write(ref_path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)
            bad_audio = np.zeros(sample_rate, dtype=np.float32)
            bad_audio[sample_rate // 2] = 1.0

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=bad_audio, sample_rate=sample_rate),
                ),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=False,
                    quality_gate=True,
                )

            self.assertIsNone(result_path)
            self.assertFalse(output_path.exists())
            segment_path = temp_path / ".bad_segments" / "_seg_0000.wav"
            self.assertTrue(segment_path.exists())
            report_path = temp_path / "quality_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["failed"])
            self.assertEqual(report["chunks"][0]["metrics"]["jump_count_gt_0_5"], 0)
            self.assertGreaterEqual(report["chunks"][0]["declick"]["repaired_jumps"], 1)
            self.assertTrue(any("low-energy" in issue for issue in report["issues"]))

    def test_generate_podcast_fails_when_audio_is_too_short_for_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "truncated.wav"
            sample_rate = 24000
            sf.write(ref_path, np.full(sample_rate // 10, 0.1, dtype=np.float32), sample_rate)
            short_audio = np.full(sample_rate, 0.1, dtype=np.float32)
            long_text = "这是一段很长的测试文本，" * 8

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=short_audio, sample_rate=sample_rate),
                ),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text=long_text,
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=False,
                    quality_gate=True,
                )

            self.assertIsNone(result_path)
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["chunks"][0]["metrics"]["chars_per_second"], 14.0)
            self.assertTrue(any("generation likely truncated" in issue for issue in report["issues"]))

    def test_loudness_gain_sample_jumps_are_declicked_before_delivery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 24000
            sf.write(ref_path, np.full(sample_rate // 10, 0.1, dtype=np.float32), sample_rate)

            audio = np.full(sample_rate, 0.1, dtype=np.float32)
            audio[sample_rate // 2] = -0.3

            def fake_normalise(audio_arg, sample_rate_arg):
                return audio_arg * 3.0

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=audio, sample_rate=sample_rate),
                ),
                patch("podcast_generator.normalise_loudness", side_effect=fake_normalise),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=True,
                    quality_gate=True,
                )

            self.assertEqual(result_path, str(output_path))
            self.assertTrue(output_path.exists())
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["failed"])
            self.assertEqual(report["final"]["metrics"]["jump_count_gt_0_5"], 0)
            self.assertGreaterEqual(report["chunks"][0]["declick"]["repaired_jumps"], 1)
            self.assertEqual(report["final"]["output_metrics"]["jump_count_gt_0_5"], 0)
            self.assertEqual(report["final"]["written_wav_metrics"]["jump_count_gt_0_5"], 0)
            self.assertEqual(report["final"]["output_diagnostics"], [])
            self.assertEqual(report["issues"], [])

    def test_post_loudness_declick_does_not_overrepair_normal_transients(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 24000
            sf.write(ref_path, np.full(sample_rate // 10, 0.1, dtype=np.float32), sample_rate)

            audio = np.zeros(sample_rate, dtype=np.float32)
            audio[::4] = 0.12

            def fake_normalise(audio_arg, sample_rate_arg):
                return audio_arg * 3.0

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=audio, sample_rate=sample_rate),
                ),
                patch("podcast_generator.normalise_loudness", side_effect=fake_normalise),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=True,
                    quality_gate=True,
                )

            self.assertEqual(result_path, str(output_path))
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["final"]["post_loudness_declick"]["threshold"], 0.5)
            self.assertEqual(report["final"]["post_loudness_declick"]["repaired_jumps"], 0)
            self.assertEqual(report["final"]["output_metrics"]["jump_count_gt_0_5"], 0)
            self.assertGreater(report["final"]["output_metrics"]["jump_count_gt_0_3"], 0)
            self.assertEqual(report["issues"], [])

    def test_chunk_sample_jumps_are_declicked_before_quality_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 24000
            sf.write(ref_path, np.full(sample_rate // 10, 0.1, dtype=np.float32), sample_rate)

            audio = np.full(sample_rate, 0.1, dtype=np.float32)
            audio[sample_rate // 2] = -0.42

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=audio, sample_rate=sample_rate),
                ),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="wav",
                    normalise=False,
                    quality_gate=True,
                )

            self.assertEqual(result_path, str(output_path))
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["chunks"][0]["metrics"]["jump_count_gt_0_5"], 0)
            self.assertGreaterEqual(report["chunks"][0]["declick"]["repaired_jumps"], 1)
            self.assertEqual(report["issues"], [])

    def test_checkpoint_metadata_mismatch_regenerates_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            checkpoint_path = temp_path / "out.ckpt"
            segment_dir = temp_path / ".out_segments"
            segment_dir.mkdir()
            stale_segment = segment_dir / "_seg_0000.wav"
            sample_rate = 24000
            sf.write(ref_path, np.full(sample_rate // 10, 0.1, dtype=np.float32), sample_rate)
            sf.write(stale_segment, np.full(sample_rate // 2, 0.2, dtype=np.float32), sample_rate)
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "chunks": ["short target"],
                        "completed": [0],
                        "segments": [str(stale_segment)],
                        "metadata": {"model_ref": "old-model"},
                    }
                ),
                encoding="utf-8",
            )

            generated_audio = np.full(sample_rate // 2, 0.05, dtype=np.float32)

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=generated_audio, sample_rate=sample_rate),
                ) as generate_chunk,
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    checkpoint_path=str(checkpoint_path),
                    output_format="wav",
                    normalise=False,
                    model_ref="new-model",
                )

            self.assertEqual(result_path, str(output_path))
            generate_chunk.assert_called_once()
            saved, _ = sf.read(stale_segment, dtype="float32")
            self.assertAlmostEqual(float(np.mean(saved)), 0.05, places=3)
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["generation"]["checkpoint_metadata"]["model_ref"], "new-model")

    def test_qwen_clone_speed_is_calibrated_to_reference_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.wav"
            sample_rate = 24000
            t = np.linspace(0, 10.0, sample_rate * 10, endpoint=False, dtype=np.float32)
            ref_audio = 0.05 * np.sin(2 * np.pi * 220 * t)
            sf.write(ref_path, ref_audio, sample_rate)

            captured_speeds: list[float] = []
            generated_t = np.linspace(0, 10.0, sample_rate * 10, endpoint=False, dtype=np.float32)
            generated_audio = 0.05 * np.sin(2 * np.pi * 180 * generated_t)

            def fake_generate_backend_chunk(**kwargs):
                captured_speeds.append(kwargs["speed"])
                return TTSGenerationResult(audio=generated_audio, sample_rate=sample_rate)

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch("podcast_generator.generate_backend_chunk", side_effect=fake_generate_backend_chunk),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="一" * 45,
                    target_text="二" * 45,
                    output_path=str(output_path),
                    speed=1.0,
                    output_format="wav",
                    normalise=False,
                    model_ref="test-model",
                )

            self.assertEqual(result_path, str(output_path))
            self.assertEqual(len(captured_speeds), 1)
            self.assertAlmostEqual(captured_speeds[0], 45 / 10 / 6.3, places=3)

            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            calibration = report["generation"]["rate_calibration"]
            self.assertTrue(calibration["enabled"])
            self.assertEqual(calibration["mode"], "reference_relative")
            self.assertEqual(report["generation"]["speed"], 1.0)
            self.assertAlmostEqual(report["generation"]["model_speed"], captured_speeds[0], places=3)
            self.assertAlmostEqual(
                report["generation"]["checkpoint_metadata"]["model_speed"],
                captured_speeds[0],
                places=3,
            )

    def test_mp3_output_records_clean_decoded_metrics_after_declick(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ref_path = temp_path / "ref.wav"
            output_path = temp_path / "out.mp3"
            sample_rate = 24000
            sf.write(ref_path, np.full(sample_rate // 10, 0.1, dtype=np.float32), sample_rate)

            audio = np.full(sample_rate, 0.1, dtype=np.float32)
            audio[sample_rate // 2] = -0.3

            def fake_normalise(audio_arg, sample_rate_arg):
                return audio_arg * 3.0

            def fake_convert(wav_path, mp3_path):
                Path(mp3_path).write_bytes(Path(wav_path).read_bytes())
                return mp3_path

            with (
                patch("podcast_generator.prepare_reference_audio", return_value=str(ref_path)),
                patch("podcast_generator.cleanup_reference_audio"),
                patch(
                    "podcast_generator.generate_backend_chunk",
                    return_value=TTSGenerationResult(audio=audio, sample_rate=sample_rate),
                ),
                patch("podcast_generator.normalise_loudness", side_effect=fake_normalise),
                patch("podcast_generator.convert_wav_to_mp3", side_effect=fake_convert),
            ):
                result_path = generate_podcast(
                    model=object(),
                    backend="qwen",
                    ref_audio_path=str(ref_path),
                    ref_text="reference",
                    target_text="short target",
                    output_path=str(output_path),
                    output_format="mp3",
                    normalise=True,
                    quality_gate=True,
                )

            self.assertEqual(result_path, str(output_path))
            self.assertTrue(output_path.exists())
            report = json.loads((temp_path / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertGreaterEqual(report["chunks"][0]["declick"]["repaired_jumps"], 1)
            self.assertEqual(report["final"]["decoded_mp3_metrics"]["jump_count_gt_0_5"], 0)
            self.assertEqual(report["final"]["decoded_mp3_diagnostics"], [])
            self.assertEqual(report["issues"], [])

    def test_benchmark_uses_backend_sample_rate_for_file_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 16000
            audio = np.ones(sample_rate, dtype=np.float32)

            with patch(
                "benchmark_tts_backends.benchmark_generate_case",
                return_value=(
                    TTSGenerationResult(audio=audio, sample_rate=sample_rate),
                    {
                        "backend": "qwen",
                        "model_ref": "model",
                        "task_mode": "clone",
                        "generate_seconds": 2.0,
                    },
                ),
            ):
                metrics = run_case(
                    backend="qwen",
                    model_ref="model",
                    case={
                        "id": "case",
                        "task_mode": "clone",
                        "text": "text",
                        "notes": "notes",
                    },
                    ref_audio="ref.wav",
                    ref_text="reference",
                    output_dir=temp_path,
                )

            output_path = Path(metrics["output_path"])
            self.assertEqual(sf.info(output_path).samplerate, sample_rate)
            self.assertEqual(metrics["sample_rate"], sample_rate)
            self.assertEqual(metrics["audio_seconds"], 1.0)
            self.assertEqual(metrics["realtime_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()
