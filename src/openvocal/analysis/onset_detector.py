"""Onset detection for note segmentation using librosa."""

from __future__ import annotations

from typing import Optional
import numpy as np

# Lazy import
_librosa = None


def _ensure_librosa():
    """Lazily import librosa."""
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa


class OnsetDetector:
    """
    Detect note onsets (start points) in audio using librosa.

    Uses a combination of spectral flux and amplitude envelope
    for robust onset detection on vocal audio.

    Attributes:
        hop_length: Hop size for analysis
        backtrack: Whether to backtrack onsets to nearest local minimum
        units: Return units ('samples', 'time', or 'frames')
    """

    def __init__(
        self,
        hop_length: int = 512,
        backtrack: bool = True,
        units: str = "samples",
    ):
        """
        Initialize onset detector.

        Args:
            hop_length: Hop size in samples for onset detection
            backtrack: Adjust onsets to nearest preceding minimum
            units: 'samples', 'time', or 'frames'
        """
        self.hop_length = hop_length
        self.backtrack = backtrack
        self.units = units

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Detect onset positions in audio.

        Args:
            audio: Mono audio samples as float32 array
            sample_rate: Sample rate in Hz
            threshold: Onset strength threshold (None = auto)

        Returns:
            Array of onset positions in specified units
        """
        _ensure_librosa()

        # Compute onset envelope
        onset_env = _librosa.onset.onset_strength(
            y=audio,
            sr=sample_rate,
            hop_length=self.hop_length,
        )

        # Detect onsets
        onsets = _librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            units=self.units,
        )

        # Apply threshold if specified
        if threshold is not None and self.units == "frames":
            onset_strengths = onset_env[onsets]
            onsets = onsets[onset_strengths >= threshold]

        return onsets

    def get_onset_strength(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Get the onset strength envelope.

        Useful for visualization and manual threshold adjustment.

        Args:
            audio: Mono audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Onset strength envelope array
        """
        _ensure_librosa()

        return _librosa.onset.onset_strength(
            y=audio,
            sr=sample_rate,
            hop_length=self.hop_length,
        )


class VoiceActivityDetector:
    """
    Detect voiced vs unvoiced regions in audio.

    Complements onset detection by identifying silence and
    unvoiced regions that should not be treated as notes.
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        threshold_db: float = -40.0,
    ):
        """
        Initialize voice activity detector.

        Args:
            frame_length: Frame size for RMS computation
            hop_length: Hop size between frames
            threshold_db: RMS threshold in dB (below = silence)
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold_db = threshold_db

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Detect voiced regions.

        Args:
            audio: Mono audio samples
            sample_rate: Sample rate (unused, for API consistency)

        Returns:
            Boolean array where True = voiced frame
        """
        _ensure_librosa()

        # Compute RMS energy
        rms = _librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        ).squeeze()

        # Convert to dB
        rms_db = _librosa.amplitude_to_db(rms, ref=np.max)

        # Threshold
        voiced = rms_db >= self.threshold_db

        return voiced

    def get_voiced_intervals(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[tuple[int, int]]:
        """
        Get intervals of voiced audio.

        Args:
            audio: Mono audio samples
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_sample, end_sample) tuples
        """
        voiced_frames = self.detect(audio, sample_rate)

        intervals = []
        in_voiced = False
        start_frame = 0

        for i, is_voiced in enumerate(voiced_frames):
            if is_voiced and not in_voiced:
                start_frame = i
                in_voiced = True
            elif not is_voiced and in_voiced:
                start_sample = start_frame * self.hop_length
                end_sample = i * self.hop_length
                intervals.append((start_sample, end_sample))
                in_voiced = False

        # Handle case where audio ends while voiced
        if in_voiced:
            start_sample = start_frame * self.hop_length
            end_sample = len(voiced_frames) * self.hop_length
            intervals.append((start_sample, end_sample))

        return intervals
