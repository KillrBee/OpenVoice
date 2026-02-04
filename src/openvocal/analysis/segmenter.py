"""Audio segmentation into note blobs combining pitch and onset detection."""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from openvocal.models.note_blob import NoteBlob
from openvocal.analysis.pitch_detector import PitchDetector, SimplePitchDetector
from openvocal.analysis.onset_detector import OnsetDetector, VoiceActivityDetector
from openvocal.utils.audio_io import compute_rms_envelope


class Segmenter:
    """
    Segment audio into NoteBlob objects.

    Combines pitch detection (torchcrepe) with onset detection (librosa)
    to identify individual note segments suitable for pitch correction.

    The segmentation pipeline:
    1. Detect voice activity (silence removal)
    2. Detect pitch curve (f0 via torchcrepe)
    3. Detect note onsets (librosa)
    4. Split pitch curve at onsets
    5. Create NoteBlob objects with pitch and amplitude data
    """

    def __init__(
        self,
        pitch_model: str = "full",
        device: Optional[str] = None,
        min_note_duration: float = 0.05,  # 50ms minimum
        use_simple_pitch: bool = False,
    ):
        """
        Initialize segmenter.

        Args:
            pitch_model: torchcrepe model size
            device: Compute device for pitch detection
            min_note_duration: Minimum note length in seconds
            use_simple_pitch: Use simple autocorrelation instead of torchcrepe
        """
        self.min_note_duration = min_note_duration
        self.use_simple_pitch = use_simple_pitch

        # Initialize detectors
        if use_simple_pitch:
            self.pitch_detector = SimplePitchDetector()
        else:
            self.pitch_detector = PitchDetector(model=pitch_model, device=device)

        self.onset_detector = OnsetDetector(units="samples")
        self.vad = VoiceActivityDetector()

        self._blob_id_counter = 0

    def segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[NoteBlob]:
        """
        Segment audio into NoteBlob objects.

        Args:
            audio: Mono audio samples as float32 array
            sample_rate: Sample rate in Hz
            progress_callback: Optional callback(stage, progress) for UI updates
                             stage: 'pitch', 'onset', 'segment'
                             progress: 0.0 to 1.0

        Returns:
            List of NoteBlob objects sorted by start time
        """
        if progress_callback:
            progress_callback("pitch", 0.0)

        # Step 1: Detect pitch
        f0 = self.pitch_detector.detect(audio, sample_rate)

        if progress_callback:
            progress_callback("pitch", 1.0)
            progress_callback("onset", 0.0)

        # Step 2: Detect onsets
        onsets = self.onset_detector.detect(audio, sample_rate)

        if progress_callback:
            progress_callback("onset", 1.0)
            progress_callback("segment", 0.0)

        # Step 3: Detect voiced regions
        voiced_intervals = self.vad.get_voiced_intervals(audio, sample_rate)

        # Step 4: Compute amplitude envelope
        amplitude_env = compute_rms_envelope(audio)

        # Step 5: Create blobs from segments
        blobs = self._create_blobs(
            audio=audio,
            sample_rate=sample_rate,
            f0=f0,
            onsets=onsets,
            voiced_intervals=voiced_intervals,
            amplitude_env=amplitude_env,
        )

        if progress_callback:
            progress_callback("segment", 1.0)

        return blobs

    def _create_blobs(
        self,
        audio: np.ndarray,
        sample_rate: int,
        f0: np.ndarray,
        onsets: np.ndarray,
        voiced_intervals: list[tuple[int, int]],
        amplitude_env: np.ndarray,
    ) -> list[NoteBlob]:
        """Create NoteBlob objects from analysis data."""
        blobs = []
        min_samples = int(self.min_note_duration * sample_rate)

        # Add boundaries (0 and end) to onsets
        boundaries = np.concatenate([[0], onsets, [len(audio)]])
        boundaries = np.unique(boundaries)

        for i in range(len(boundaries) - 1):
            start_sample = int(boundaries[i])
            end_sample = int(boundaries[i + 1])

            # Skip if too short
            if end_sample - start_sample < min_samples:
                continue

            # Check if segment overlaps with a voiced region
            is_voiced = any(
                start_sample < vi_end and end_sample > vi_start
                for vi_start, vi_end in voiced_intervals
            )

            if not is_voiced:
                continue

            # Extract f0 segment
            f0_segment = self._extract_f0_segment(
                f0, start_sample, end_sample, sample_rate
            )

            # Skip if mostly unvoiced (f0 = 0)
            voiced_ratio = np.sum(f0_segment > 0) / len(f0_segment)
            if voiced_ratio < 0.3:
                continue

            # Extract amplitude segment
            amp_segment = self._extract_amplitude_segment(
                amplitude_env, start_sample, end_sample, sample_rate
            )

            # Create blob
            self._blob_id_counter += 1
            blob = NoteBlob(
                id=self._blob_id_counter,
                start_sample=start_sample,
                end_sample=end_sample,
                sample_rate=sample_rate,
                original_f0=f0_segment,
                amplitude_envelope=amp_segment,
            )
            blobs.append(blob)

        return blobs

    def _extract_f0_segment(
        self,
        f0: np.ndarray,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract f0 values for a sample range."""
        if isinstance(self.pitch_detector, SimplePitchDetector):
            hop = self.pitch_detector.hop_length
            start_idx = start_sample // hop
            end_idx = end_sample // hop
        else:
            # torchcrepe uses 16kHz internal rate with hop_length
            crepe_sr = 16000
            hop = self.pitch_detector.hop_length
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            start_idx = int(start_time * crepe_sr / hop)
            end_idx = int(end_time * crepe_sr / hop)

        start_idx = max(0, start_idx)
        end_idx = min(len(f0), end_idx)

        if start_idx >= end_idx:
            return np.array([0.0], dtype=np.float32)

        return f0[start_idx:end_idx].copy()

    def _extract_amplitude_segment(
        self,
        amplitude_env: np.ndarray,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Extract amplitude envelope for a sample range."""
        start_idx = start_sample // hop_length
        end_idx = end_sample // hop_length

        start_idx = max(0, start_idx)
        end_idx = min(len(amplitude_env), end_idx)

        if start_idx >= end_idx:
            return np.array([0.0], dtype=np.float32)

        return amplitude_env[start_idx:end_idx].copy()

    def resegment_at_position(
        self,
        blobs: list[NoteBlob],
        split_sample: int,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[NoteBlob]:
        """
        Split an existing blob at a specific position.

        Used for manual segmentation adjustments.

        Args:
            blobs: Current list of blobs
            split_sample: Sample position to split at
            audio: Original audio data
            sample_rate: Sample rate

        Returns:
            Updated list of blobs
        """
        # Find blob containing the split point
        target_blob = None
        target_idx = -1
        for i, blob in enumerate(blobs):
            if blob.start_sample < split_sample < blob.end_sample:
                target_blob = blob
                target_idx = i
                break

        if target_blob is None:
            return blobs

        # Create two new blobs
        f0 = target_blob.original_f0
        amp = target_blob.amplitude_envelope

        # Calculate split point in f0 array
        duration = target_blob.duration
        num_samples = target_blob.num_samples
        if num_samples == 0:
            return blobs  # Cannot split zero-length blob

        relative_pos = (split_sample - target_blob.start_sample) / num_samples
        # Clamp to valid range to prevent index out of bounds
        f0_split = min(max(0, int(relative_pos * len(f0))), len(f0))
        amp_split = min(max(0, int(relative_pos * len(amp))), len(amp)) if amp is not None else 0

        self._blob_id_counter += 1
        blob1 = NoteBlob(
            id=self._blob_id_counter,
            start_sample=target_blob.start_sample,
            end_sample=split_sample,
            sample_rate=sample_rate,
            original_f0=f0[:f0_split].copy(),
            amplitude_envelope=amp[:amp_split].copy() if amp is not None else None,
        )

        self._blob_id_counter += 1
        blob2 = NoteBlob(
            id=self._blob_id_counter,
            start_sample=split_sample,
            end_sample=target_blob.end_sample,
            sample_rate=sample_rate,
            original_f0=f0[f0_split:].copy(),
            amplitude_envelope=amp[amp_split:].copy() if amp is not None else None,
        )

        # Replace original with two new blobs
        new_blobs = blobs[:target_idx] + [blob1, blob2] + blobs[target_idx + 1:]
        return new_blobs
