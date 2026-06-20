import os
import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import (
    SAMPLE_RATE,
    ReferenceAudioError,
    audio_quality_issues,
    audit_reference_audio,
    check_audio_health,
    compute_audio_quality_metrics,
    convert_audio_if_needed,
    crossfade_concat,
    limit_audio_peak,
    prepare_reference_pair,
    prepare_reference_audio_clip,
    remove_dc_offset,
    sanitise_tts_text,
    smooth_sample_jumps,
    trim_silence,
    write_quality_report,
)


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

    def test_prepare_reference_audio_clip_writes_clean_wav_and_quality_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            t = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
            voice = 0.12 * np.sin(2 * np.pi * 180 * t)
            audio = np.concatenate([
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                voice,
                np.zeros(SAMPLE_RATE, dtype=np.float32),
            ])
            sf.write(source, audio, SAMPLE_RATE)

            clean_path, quality_path, quality = prepare_reference_audio_clip(source, root / "profile")

            self.assertEqual(clean_path.name, "reference_clean.wav")
            self.assertEqual(quality_path.name, "reference_quality.json")
            self.assertTrue(clean_path.exists())
            self.assertTrue(quality_path.exists())
            self.assertTrue(quality["ok"])
            self.assertGreaterEqual(quality["duration_seconds"], 3.0)
            self.assertLessEqual(quality["duration_seconds"], 8.0)
            self.assertGreaterEqual(quality["active_ratio"], 0.65)

            saved_audio, saved_rate = sf.read(clean_path)
            self.assertEqual(saved_rate, SAMPLE_RATE)
            self.assertEqual(np.asarray(saved_audio).ndim, 1)
            payload = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["clean_path"], str(clean_path))

    def test_prepare_reference_audio_clip_rejects_low_rms_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "quiet.wav"
            audio = np.full(int(4.0 * SAMPLE_RATE), 0.001, dtype=np.float32)
            sf.write(source, audio, SAMPLE_RATE)

            with self.assertRaisesRegex(ReferenceAudioError, "RMS|detectable"):
                prepare_reference_audio_clip(source, root / "profile")

            quality_path = root / "profile" / "reference_quality.json"
            payload = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertIn("failure_reason", payload)

    def test_prepare_reference_audio_clip_rejects_long_pause(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "paused.wav"
            tone_t = np.linspace(0, 2.0, int(2.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
            tone = 0.12 * np.sin(2 * np.pi * 180 * tone_t)
            audio = np.concatenate([
                tone,
                np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32),
                tone,
            ])
            sf.write(source, audio, SAMPLE_RATE)

            with self.assertRaisesRegex(ReferenceAudioError, "pause|silence|3 seconds"):
                prepare_reference_audio_clip(source, root / "profile")

    def test_audit_reference_audio_rejects_jump05(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "jump.wav"
            t = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
            audio = 0.12 * np.sin(2 * np.pi * 180 * t)
            audio[SAMPLE_RATE] = 0.4
            audio[SAMPLE_RATE + 1] = -0.4
            sf.write(source, audio, SAMPLE_RATE)

            audit = audit_reference_audio(source, "hello reference")

            self.assertFalse(audit["ok"])
            self.assertGreater(audit["metrics"]["jump_count_gt_0_5"], 0)
            self.assertTrue(any("sample jumps" in issue for issue in audit["issues"]))

    def test_audit_reference_audio_reads_m4a_via_ffmpeg_fallback(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is required to encode the m4a fixture")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav_source = root / "source.wav"
            m4a_source = root / "source.m4a"
            t = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
            audio = 0.12 * np.sin(2 * np.pi * 180 * t)
            sf.write(wav_source, audio, SAMPLE_RATE)
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(wav_source), str(m4a_source)],
                check=True,
            )

            audit = audit_reference_audio(m4a_source, "hello reference")

            self.assertTrue(audit["ok"])
            self.assertEqual(audit["sample_rate"], SAMPLE_RATE)
            self.assertAlmostEqual(audit["duration_seconds"], 5.0, places=1)

    def test_prepare_reference_pair_writes_full_clean_reference_when_audit_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            t = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
            audio = 0.12 * np.sin(2 * np.pi * 180 * t) + 0.01
            sf.write(source, audio, SAMPLE_RATE)

            result = prepare_reference_pair(source, "full reference transcript", root / "profile")

            self.assertEqual(result.clean_audio_path.name, "reference_clean.wav")
            self.assertEqual(result.clean_text_path.name, "reference_clean.txt")
            self.assertEqual(result.quality_path.name, "reference_quality.json")
            self.assertTrue(result.payload["ok"])
            self.assertEqual(result.clean_text_path.read_text(encoding="utf-8"), "full reference transcript")

            saved_audio, saved_rate = sf.read(result.clean_audio_path)
            self.assertEqual(saved_rate, SAMPLE_RATE)
            self.assertEqual(len(saved_audio), len(audio))
            self.assertAlmostEqual(float(np.mean(saved_audio)), 0.0, places=3)

    def test_prepare_reference_pair_does_not_write_profile_artifacts_when_audit_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "jump.wav"
            t = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
            audio = 0.12 * np.sin(2 * np.pi * 180 * t)
            audio[SAMPLE_RATE] = 0.4
            audio[SAMPLE_RATE + 1] = -0.4
            sf.write(source, audio, SAMPLE_RATE)

            with self.assertRaisesRegex(ReferenceAudioError, "sample jumps"):
                prepare_reference_pair(source, "bad reference transcript", root / "profile")

            self.assertFalse((root / "profile" / "reference_clean.wav").exists())
            self.assertFalse((root / "profile" / "reference_clean.txt").exists())
            self.assertFalse((root / "profile" / "reference_quality.json").exists())


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

    def test_limit_audio_peak_uses_soft_saturation(self):
        """Soft (tanh) limiter keeps the peak below the ceiling without
        hard clipping, so over-amplitude samples are smoothly compressed
        instead of truncated (which would create audible clicks)."""
        audio = np.array([-2.0, -0.5, 0.5, 1.0], dtype=np.float32)

        limited = limit_audio_peak(audio, ceiling=0.95)

        # Peak is below the ceiling (no hard step), and the originally
        # -2.0 sample is smoothly compressed.
        peak = float(np.max(np.abs(limited)))
        self.assertLessEqual(peak, 0.95)
        self.assertGreater(peak, 0.5)  # actually compressed, not zero
        # Small values are smoothly compressed by ~5% (tanh knee) but stay
        # close to their original amplitude (linear region of tanh).
        self.assertAlmostEqual(float(limited[2]), 0.46, places=2)
        self.assertAlmostEqual(float(limited[3]), 0.74, places=2)

    def test_limited_audio_does_not_trigger_near_clipping_warning(self):
        limited = limit_audio_peak(np.array([-2.0, 1.0], dtype=np.float32), ceiling=0.95)

        warnings = check_audio_health(limited, SAMPLE_RATE)

        self.assertFalse(any("near clipping" in warning for warning in warnings))

    def test_smooth_sample_jumps_repairs_short_clicks(self):
        audio = np.full(100, 0.1, dtype=np.float32)
        audio[50] = -0.35

        repaired, stats = smooth_sample_jumps(audio, threshold=0.3, radius=2)
        metrics = compute_audio_quality_metrics(repaired, SAMPLE_RATE)

        self.assertGreaterEqual(stats["detected_jumps"], 2)
        self.assertGreaterEqual(stats["repaired_jumps"], 1)
        self.assertEqual(metrics["jump_count_gt_0_3"], 0)
        self.assertLess(metrics["max_jump"], 0.3)

    def test_smooth_sample_jumps_preserves_clean_audio(self):
        audio = np.linspace(-0.1, 0.1, 100, dtype=np.float32)

        repaired, stats = smooth_sample_jumps(audio, threshold=0.3, radius=2)

        self.assertEqual(stats["detected_jumps"], 0)
        np.testing.assert_allclose(repaired, audio)

    def test_compute_audio_quality_metrics_reports_required_fields(self):
        audio = np.array([0.0, 0.4, -0.2, 0.6, -0.1], dtype=np.float32)

        metrics = compute_audio_quality_metrics(audio, sample_rate=5)

        self.assertEqual(metrics["duration"], 1.0)
        self.assertAlmostEqual(metrics["peak"], 0.6, places=6)
        self.assertGreater(metrics["rms"], 0.0)
        self.assertEqual(metrics["jump_count_gt_0_3"], 4)
        self.assertEqual(metrics["jump_count_gt_0_5"], 3)
        self.assertAlmostEqual(metrics["max_jump"], 0.8, places=6)
        self.assertIn("low_energy_ratio", metrics)
        self.assertIn("longest_low_energy_run", metrics)
        self.assertIn("rms_low_energy_ratio", metrics)
        self.assertIn("longest_rms_low_energy_run", metrics)

    def test_audio_quality_issues_flags_jump_and_low_energy(self):
        metrics = compute_audio_quality_metrics(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

        issues = audio_quality_issues(metrics, label="chunk 1/1")

        self.assertTrue(any("very low RMS" in issue for issue in issues))
        self.assertTrue(any("mostly low-energy" in issue for issue in issues))

    def test_audio_quality_issues_flags_audible_pause_dropout(self):
        sr = SAMPLE_RATE
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        speech = 0.08 * np.sin(2 * np.pi * 220 * t)
        low_floor = np.full(int(3.0 * sr), 8.5e-4, dtype=np.float32)
        audio = np.concatenate([speech, low_floor, speech])

        metrics = compute_audio_quality_metrics(audio, sr)
        issues = audio_quality_issues(metrics, label="final audio")

        self.assertGreaterEqual(metrics["longest_rms_low_energy_run"], 3.0)
        self.assertTrue(any("audible pause/dropout" in issue for issue in issues))

    def test_write_quality_report_creates_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "quality_report.json"

            result = write_quality_report(report_path, {"status": "failed", "failed": True})

            self.assertEqual(result, str(report_path))
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["failed"])

    def test_remove_dc_offset_centres_waveform(self):
        # Audio with a clear DC bias of 0.5 and a 0.1 amplitude variation
        audio = np.full(1000, 0.5, dtype=np.float32)
        audio[100:200] = 0.6  # 0.1 above the bias
        cleaned = remove_dc_offset(audio)
        # Mean is brought to (approximately) zero
        self.assertLess(abs(float(np.mean(cleaned))), 1e-4)
        # The 0.6 samples are now 0.09 (0.6 minus the new mean of ~0.51),
        # i.e. the 0.1 *relative* variation is preserved.
        self.assertAlmostEqual(float(cleaned[100]), 0.09, places=2)

    def test_crossfade_concat_smoother_than_hard(self):
        """Crossfade between two segments must produce a smaller sample-to-
        sample step at the boundary than a hard np.concatenate."""
        # Two segments with a deliberate step at the boundary
        a = np.zeros(1000, dtype=np.float32)
        a[-1] = 0.4
        b = np.full(1000, 0.8, dtype=np.float32)
        b[0] = -0.4  # big jump at boundary
        hard = np.concatenate([a, b])
        hard_jump = float(abs(hard[len(a)] - hard[len(a) - 1]))
        fade = 240  # 10 ms at 24 kHz
        xfaded = crossfade_concat([a, b], fade_ms=10, sample_rate=SAMPLE_RATE)
        # The hard boundary is len(a) samples into the output.
        boundary_jump = float(abs(xfaded[len(a) - 1] - xfaded[len(a)]))
        # The crossfade boundary should be smoother
        self.assertLess(boundary_jump, hard_jump)
        # The output is shorter by `fade` samples (one fade window consumed
        # at the boundary between a and b).
        self.assertEqual(len(xfaded), len(hard) - fade)

    def test_crossfade_concat_handles_short_segments(self):
        """Segments shorter than the fade window should still concatenate."""
        a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        b = np.array([0.4, 0.5], dtype=np.float32)
        result = crossfade_concat([a, b], fade_ms=10, sample_rate=SAMPLE_RATE)
        # When both segments are shorter than the fade window, we fall back
        # to a hard concat, so length is preserved.
        self.assertEqual(len(result), len(a) + len(b))
        self.assertAlmostEqual(float(result[-1]), 0.5, places=6)

    def test_trim_silence_shortens_long_silence_runs(self):
        """A 1 second silence in the middle should be shortened to ~200ms,
        while the active audio on either side is preserved."""
        sr = 24000
        # 1s of tone (440Hz, low amplitude) + 1s of silence + 1s of tone
        t1 = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
        tone_a = 0.1 * np.sin(2 * np.pi * 440 * t1)
        tone_b = 0.1 * np.sin(2 * np.pi * 440 * t1)
        silence = np.zeros(sr, dtype=np.float32)
        audio = np.concatenate([tone_a, silence, tone_b])

        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=0, tail_trim_ms=0, pad_ms=0,
        )

        # Result should be: tone_a (1s) + ~200ms silence + tone_b (1s) = ~2.2s
        # Allow a wide tolerance for envelope window alignment.
        self.assertLess(len(trimmed), len(audio))
        # The middle gap should be roughly 200ms, not 1000ms
        # (it can be a bit longer because of envelope quantization)
        self.assertLess(len(trimmed), int(0.5 * len(audio)) + sr)

    def test_trim_silence_refuses_aggressive_low_energy_speech_cut(self):
        """Low-energy cloned speech can sit below the old fixed threshold.
        It must not be mistaken for silence and deleted wholesale."""
        sr = 24000
        t = np.linspace(0, 14.4, int(14.4 * sr), endpoint=False, dtype=np.float32)
        low_energy_speech = 8e-4 * np.sin(2 * np.pi * 180 * t)

        trimmed = trim_silence(
            low_energy_speech,
            sr,
            threshold=5e-3,
            min_silence_ms=300,
            keep_silence_ms=200,
            max_trim_ratio=0.30,
            head_trim_ms=0,
            tail_trim_ms=0,
            pad_ms=0,
        )

        self.assertGreaterEqual(len(trimmed), int(len(low_energy_speech) * 0.95))

    def test_trim_silence_rejects_aggressive_trim_ratio(self):
        """A trim that would remove most of a speech-like chunk is skipped."""
        sr = 24000
        tone = np.full(int(2.0 * sr), 0.1, dtype=np.float32)
        long_gap = np.zeros(int(10.0 * sr), dtype=np.float32)
        audio = np.concatenate([tone, long_gap, tone])

        trimmed = trim_silence(
            audio,
            sr,
            threshold=5e-3,
            min_silence_ms=300,
            keep_silence_ms=200,
            max_trim_ratio=0.35,
            head_trim_ms=0,
            tail_trim_ms=0,
            pad_ms=0,
        )

        self.assertEqual(len(trimmed), len(audio))

    def test_sanitise_tts_text_removes_zero_width_format_chars(self):
        dirty = "母亲说：\u200b“写完院里这十八缸水”\x00\r\n下一句"

        cleaned = sanitise_tts_text(dirty)

        self.assertNotIn("\u200b", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertEqual(cleaned, "母亲说：“写完院里这十八缸水”\n下一句")

    def test_trim_silence_preserves_short_pauses(self):
        """A short 150ms pause should NOT be trimmed — only long ones."""
        sr = 24000
        tone_a = np.ones(int(0.5 * sr), dtype=np.float32) * 0.1
        silence = np.zeros(int(0.15 * sr), dtype=np.float32)
        tone_b = np.ones(int(0.5 * sr), dtype=np.float32) * 0.1
        audio = np.concatenate([tone_a, silence, tone_b])
        original_len = len(audio)

        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=0, tail_trim_ms=0, pad_ms=0,
        )

        # The 150ms pause is shorter than min_silence_ms (300), so it should
        # be preserved.
        self.assertEqual(len(trimmed), original_len)

    def test_trim_silence_skips_force_trim_when_edges_contain_speech(self):
        """Smart trim must NOT cut head/tail when the candidate region is
        actually speech — this is the fix for the "一句话没读完就开始读下一句"
        regression where an unconditional 80ms tail trim clipped sentence
        endings that contained trailing decay / breath / final consonants
        above the silence floor.
        """
        sr = 24000
        audio = np.ones(sr, dtype=np.float32)  # 1s of constant signal
        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=100, tail_trim_ms=100, pad_ms=0,
        )
        # Smart trim skipped because the edges are 1.0 RMS (well above
        # threshold 5e-3). The whole 1s chunk is preserved.
        self.assertEqual(len(trimmed), sr)

    def test_trim_silence_cuts_silent_edges_by_default(self):
        """When the edges are actually silence, the smart trim still
        removes them — it only refuses to clip speech.
        """
        sr = 24000
        # 100ms silence head + 800ms 0.5-amplitude speech + 100ms silence tail
        audio = np.concatenate([
            np.zeros(int(0.1 * sr), dtype=np.float32),
            np.full(int(0.8 * sr), 0.5, dtype=np.float32),
            np.zeros(int(0.1 * sr), dtype=np.float32),
        ])
        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=100, tail_trim_ms=100, pad_ms=0,
        )
        # Both edges were below threshold → both get trimmed.
        self.assertEqual(len(trimmed), int(0.8 * sr))

    def test_trim_silence_preserves_trailing_speech_at_chunk_end(self):
        """Regression: a TTS chunk that ends with a small burst of speech
        (e.g. final consonant / breath / trailing decay above the silence
        floor) must NOT be eaten by the silence-run detector. The old
        ``sample_end = j*hop + win`` formula extended every silence run
        by one window (50 ms), so a 100 ms 0.5-amplitude tail at the
        end of an otherwise-silent chunk was being dropped — and that's
        the "一句话没读完就开始读下一句了" symptom in production.
        """
        sr = 24000
        # 23 000 samples of pure silence + 100 samples of 0.5-amp speech
        audio = np.zeros(sr, dtype=np.float32)
        audio[-100:] = 0.5
        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=80, tail_trim_ms=80, pad_ms=0,
        )
        # The 100-sample 0.5-amp tail must survive somewhere in the output.
        self.assertGreaterEqual(len(trimmed), 100)
        tail_rms = float(np.sqrt(np.mean(trimmed[-100:] ** 2)))
        self.assertAlmostEqual(tail_rms, 0.5, places=2)

    def test_trim_silence_force_trim_speech_opt_in(self):
        """force_trim_speech=True preserves the old behaviour for callers
        that explicitly want unconditional cuts."""
        sr = 24000
        audio = np.ones(sr, dtype=np.float32)
        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=100, tail_trim_ms=100, pad_ms=0,
            force_trim_speech=True,
        )
        self.assertEqual(len(trimmed), int(0.8 * sr))

    def test_trim_silence_handles_empty_input(self):
        """Empty input should return empty without raising."""
        audio = np.array([], dtype=np.float32)
        trimmed = trim_silence(audio, SAMPLE_RATE)
        self.assertEqual(len(trimmed), 0)

    def test_trim_silence_handles_all_silence_input(self):
        """All-silence input should return without raising (may be empty)."""
        sr = 24000
        audio = np.zeros(sr, dtype=np.float32)  # 1s of pure silence
        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=80, tail_trim_ms=80, pad_ms=30,
        )
        # Force trim may consume everything; either empty or short pad
        self.assertLessEqual(len(trimmed), sr)

    def test_trim_silence_adds_pad(self):
        """pad_ms > 0 should prepend/append 30ms of silence (at 24kHz)."""
        sr = 24000
        audio = np.ones(sr, dtype=np.float32)  # 1s of constant signal
        trimmed = trim_silence(
            audio, sr,
            threshold=5e-3, min_silence_ms=300, keep_silence_ms=200,
            head_trim_ms=0, tail_trim_ms=0, pad_ms=30,
        )
        # 1000ms + 30ms head + 30ms tail = 1060ms
        self.assertEqual(len(trimmed), int(1.06 * sr))
        # First 30ms should be silence
        self.assertLess(float(np.max(np.abs(trimmed[: int(0.03 * sr)]))), 1e-6)
        # Last 30ms should be silence
        self.assertLess(float(np.max(np.abs(trimmed[-int(0.03 * sr):]))), 1e-6)


if __name__ == "__main__":
    unittest.main()
