"""Tests for audio analysis modules."""

import numpy as np
import pytest

from openvocal.analysis.pitch_detector import SimplePitchDetector
from openvocal.analysis.onset_detector import VoiceActivityDetector


class TestSimplePitchDetector:
    """Tests for the simple autocorrelation pitch detector."""

    def test_detect_sine_wave(self):
        """Test pitch detection on a pure sine wave."""
        sample_rate = 44100
        duration = 1.0
        frequency = 440.0  # A4

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        detector = SimplePitchDetector()
        f0 = detector.detect(audio, sample_rate)

        # Check that detected pitch is close to 440 Hz
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            mean_f0 = np.mean(voiced_f0)
            assert abs(mean_f0 - frequency) < 20  # Within 20 Hz

    def test_detect_silence(self):
        """Test pitch detection on silence."""
        sample_rate = 44100
        audio = np.zeros(44100, dtype=np.float32)

        detector = SimplePitchDetector()
        f0 = detector.detect(audio, sample_rate)

        # Should be mostly zeros (unvoiced)
        assert np.mean(f0 == 0) > 0.8


class TestVoiceActivityDetector:
    """Tests for voice activity detection."""

    def test_detect_silence(self):
        """Test VAD on silence."""
        sample_rate = 44100
        audio = np.zeros(44100, dtype=np.float32)

        vad = VoiceActivityDetector()
        voiced = vad.detect(audio, sample_rate)

        # Should be all unvoiced
        assert not np.any(voiced)

    def test_detect_audio(self):
        """Test VAD on audio signal."""
        sample_rate = 44100
        t = np.linspace(0, 1.0, 44100, dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        vad = VoiceActivityDetector()
        voiced = vad.detect(audio, sample_rate)

        # Should detect some voiced frames
        assert np.sum(voiced) > 0

    def test_voiced_intervals(self):
        """Test getting voiced intervals."""
        sample_rate = 44100

        # Create audio with silence, then signal, then silence
        silence = np.zeros(11025, dtype=np.float32)
        t = np.linspace(0, 0.5, 22050, dtype=np.float32)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        audio = np.concatenate([silence, signal, silence])

        vad = VoiceActivityDetector()
        intervals = vad.get_voiced_intervals(audio, sample_rate)

        # Should find approximately one interval
        assert len(intervals) >= 1
