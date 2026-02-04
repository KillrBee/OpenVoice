"""Project model containing audio data and note blobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from openvocal.models.note_blob import NoteBlob


@dataclass
class Project:
    """
    Represents an OpenVocal project with audio and analyzed notes.

    Attributes:
        name: Project name (derived from filename if not set)
        audio_path: Path to the source audio file
        audio_data: Raw audio samples as mono float32 array
        sample_rate: Audio sample rate in Hz
        blobs: List of detected/edited note blobs
        playback_position: Current playback position in samples
        loop_start: Loop region start in samples (None = no loop)
        loop_end: Loop region end in samples (None = no loop)
    """

    name: str = "Untitled"
    audio_path: Optional[Path] = None
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 44100
    blobs: list[NoteBlob] = field(default_factory=list)
    playback_position: int = 0
    loop_start: Optional[int] = None
    loop_end: Optional[int] = None
    _is_modified: bool = field(default=False, repr=False)
    _blob_id_counter: int = field(default=0, repr=False)

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if self.audio_data is None or self.sample_rate <= 0:
            return 0.0
        return len(self.audio_data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Total number of audio samples."""
        if self.audio_data is None:
            return 0
        return len(self.audio_data)

    @property
    def playback_time(self) -> float:
        """Current playback position in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return self.playback_position / self.sample_rate

    @property
    def is_modified(self) -> bool:
        """Whether the project has unsaved changes."""
        return self._is_modified

    @property
    def has_audio(self) -> bool:
        """Whether audio has been loaded."""
        return self.audio_data is not None and len(self.audio_data) > 0

    @property
    def loop_enabled(self) -> bool:
        """Whether loop region is active."""
        return self.loop_start is not None and self.loop_end is not None

    def mark_modified(self) -> None:
        """Mark project as having unsaved changes."""
        self._is_modified = True

    def mark_saved(self) -> None:
        """Mark project as saved (no pending changes)."""
        self._is_modified = False

    def next_blob_id(self) -> int:
        """Generate a unique ID for a new blob."""
        self._blob_id_counter += 1
        return self._blob_id_counter

    def add_blob(self, blob: NoteBlob) -> None:
        """Add a blob to the project."""
        self.blobs.append(blob)
        self.blobs.sort(key=lambda b: b.start_sample)
        self.mark_modified()

    def remove_blob(self, blob_id: int) -> bool:
        """Remove a blob by ID. Returns True if found and removed."""
        for i, blob in enumerate(self.blobs):
            if blob.id == blob_id:
                self.blobs.pop(i)
                self.mark_modified()
                return True
        return False

    def get_blob_by_id(self, blob_id: int) -> Optional[NoteBlob]:
        """Find a blob by its ID."""
        for blob in self.blobs:
            if blob.id == blob_id:
                return blob
        return None

    def get_blobs_at_time(self, time_seconds: float) -> list[NoteBlob]:
        """Get all blobs that contain the given time."""
        return [blob for blob in self.blobs if blob.contains_time(time_seconds)]

    def get_blobs_in_range(
        self, start_time: float, end_time: float
    ) -> list[NoteBlob]:
        """Get all blobs that overlap with the given time range."""
        return [
            blob for blob in self.blobs
            if blob.start_time < end_time and blob.end_time > start_time
        ]

    def get_selected_blobs(self) -> list[NoteBlob]:
        """Get all currently selected blobs."""
        return [blob for blob in self.blobs if blob.is_selected]

    def select_blob(self, blob_id: int, extend: bool = False) -> None:
        """
        Select a blob by ID.

        Args:
            blob_id: ID of blob to select
            extend: If True, add to selection; if False, replace selection
        """
        if not extend:
            self.deselect_all()

        blob = self.get_blob_by_id(blob_id)
        if blob:
            blob.select()

    def deselect_all(self) -> None:
        """Deselect all blobs."""
        for blob in self.blobs:
            blob.deselect()

    def set_loop_region(self, start_time: float, end_time: float) -> None:
        """Set the loop region by time in seconds."""
        self.loop_start = int(start_time * self.sample_rate)
        self.loop_end = int(end_time * self.sample_rate)

    def clear_loop(self) -> None:
        """Clear the loop region."""
        self.loop_start = None
        self.loop_end = None

    def get_audio_segment(
        self, start_sample: int, end_sample: int
    ) -> Optional[np.ndarray]:
        """Get a segment of audio data."""
        if self.audio_data is None:
            return None
        start = max(0, start_sample)
        end = min(len(self.audio_data), end_sample)
        return self.audio_data[start:end]

    def to_rust_playback_params(self) -> dict:
        """
        Serialize project state for Rust playback engine.

        Returns dict suitable for PyO3 conversion.
        """
        return {
            "sample_rate": self.sample_rate,
            "playback_position": self.playback_position,
            "loop_start": self.loop_start,
            "loop_end": self.loop_end,
            "blobs": [blob.to_rust_params() for blob in self.blobs],
        }
