"""NoteBlob data model representing a detected vocal note segment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class NoteBlob:
    """
    Represents a single note segment with pitch and timing information.

    A "Blob" visualizes three data layers:
    1. Energy (Shape): Thickness determined by amplitude envelope
    2. Pitch Curve (Line): Continuous torchcrepe detection (vibrato, scoops, falls)
    3. Anchor (Ghost): Rectangular snap target representing quantized note

    Attributes:
        id: Unique identifier for the blob
        start_sample: Starting sample index in the audio buffer
        end_sample: Ending sample index in the audio buffer
        sample_rate: Audio sample rate for time conversion
        original_f0: Raw frequency (Hz) array from pitch detection
        amplitude_envelope: RMS amplitude over time for blob thickness
        shift_semitones: Pitch adjustment in semitones (can be fractional)
        stretch_ratio: Time stretch factor (1.0 = original, 2.0 = double length)
        quantized_midi: The MIDI note number for the "snap target"
    """

    id: int
    start_sample: int
    end_sample: int
    sample_rate: int
    original_f0: np.ndarray
    amplitude_envelope: Optional[np.ndarray] = None
    shift_semitones: float = 0.0
    stretch_ratio: float = 1.0
    quantized_midi: Optional[int] = None
    _selected: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Compute derived values after initialization."""
        if self.quantized_midi is None:
            self.quantized_midi = self._compute_median_midi()

    def _compute_median_midi(self) -> int:
        """Compute the median MIDI note from the f0 curve."""
        # Filter out unvoiced frames (f0 <= 0)
        voiced_f0 = self.original_f0[self.original_f0 > 0]
        if len(voiced_f0) == 0:
            return 60  # Default to middle C
        median_f0 = np.median(voiced_f0)
        return self.hz_to_midi(median_f0)

    @staticmethod
    def hz_to_midi(freq: float) -> int:
        """Convert frequency in Hz to MIDI note number."""
        if freq <= 0:
            return 0
        return int(round(69 + 12 * np.log2(freq / 440.0)))

    @staticmethod
    def midi_to_hz(midi: int) -> float:
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2 ** ((midi - 69) / 12.0))

    @staticmethod
    def midi_to_note_name(midi: int) -> str:
        """Convert MIDI note number to note name (e.g., 'C4', 'F#5')."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = note_names[midi % 12]
        return f"{note}{octave}"

    @property
    def start_time(self) -> float:
        """Start time in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return self.start_sample / self.sample_rate

    @property
    def end_time(self) -> float:
        """End time in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return self.end_sample / self.sample_rate

    @property
    def duration(self) -> float:
        """Duration in seconds (original, not stretched)."""
        if self.sample_rate <= 0:
            return 0.0
        return (self.end_sample - self.start_sample) / self.sample_rate

    @property
    def stretched_duration(self) -> float:
        """Duration in seconds after stretch applied."""
        return self.duration * self.stretch_ratio

    @property
    def shifted_midi(self) -> float:
        """MIDI note number after pitch shift (can be fractional)."""
        if self.quantized_midi is None:
            return 60.0
        return self.quantized_midi + self.shift_semitones

    @property
    def shifted_f0(self) -> np.ndarray:
        """F0 curve after pitch shift applied."""
        shift_ratio = 2 ** (self.shift_semitones / 12.0)
        return self.original_f0 * shift_ratio

    @property
    def num_samples(self) -> int:
        """Number of samples in this blob."""
        return self.end_sample - self.start_sample

    @property
    def is_selected(self) -> bool:
        """Whether this blob is currently selected."""
        return self._selected

    def select(self) -> None:
        """Mark this blob as selected."""
        self._selected = True

    def deselect(self) -> None:
        """Mark this blob as not selected."""
        self._selected = False

    def contains_time(self, time_seconds: float) -> bool:
        """Check if a time point falls within this blob."""
        return self.start_time <= time_seconds <= self.end_time

    def get_f0_at_time(self, time_seconds: float) -> float:
        """Get the f0 value at a specific time within the blob."""
        if not self.contains_time(time_seconds):
            return 0.0

        if len(self.original_f0) == 0:
            return 0.0

        duration = self.duration
        if duration <= 0:
            return self.shifted_f0[0] if len(self.shifted_f0) > 0 else 0.0

        # Map time to f0 array index
        relative_time = time_seconds - self.start_time
        relative_pos = relative_time / duration
        idx = int(relative_pos * (len(self.original_f0) - 1))
        idx = max(0, min(idx, len(self.original_f0) - 1))

        return self.shifted_f0[idx]

    def to_rust_params(self) -> dict:
        """
        Serialize blob parameters for transfer to Rust engine.

        Returns dict suitable for PyO3 conversion.
        """
        return {
            "id": self.id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "shift_semitones": self.shift_semitones,
            "stretch_ratio": self.stretch_ratio,
        }

    def clone(self) -> NoteBlob:
        """Create a deep copy of this blob."""
        return NoteBlob(
            id=self.id,
            start_sample=self.start_sample,
            end_sample=self.end_sample,
            sample_rate=self.sample_rate,
            original_f0=self.original_f0.copy(),
            amplitude_envelope=self.amplitude_envelope.copy() if self.amplitude_envelope is not None else None,
            shift_semitones=self.shift_semitones,
            stretch_ratio=self.stretch_ratio,
            quantized_midi=self.quantized_midi,
        )
