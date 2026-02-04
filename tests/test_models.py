"""Tests for data models."""

import numpy as np
import pytest

from openvocal.models.note_blob import NoteBlob
from openvocal.models.project import Project


class TestNoteBlob:
    """Tests for NoteBlob model."""

    def test_create_blob(self):
        """Test basic blob creation."""
        f0 = np.array([440.0, 440.0, 440.0], dtype=np.float32)
        blob = NoteBlob(
            id=1,
            start_sample=0,
            end_sample=44100,
            sample_rate=44100,
            original_f0=f0,
        )

        assert blob.id == 1
        assert blob.start_sample == 0
        assert blob.end_sample == 44100
        assert blob.duration == 1.0
        assert blob.shift_semitones == 0.0
        assert blob.stretch_ratio == 1.0

    def test_hz_to_midi(self):
        """Test frequency to MIDI conversion."""
        assert NoteBlob.hz_to_midi(440.0) == 69  # A4
        assert NoteBlob.hz_to_midi(261.63) == 60  # C4 (approx)
        assert NoteBlob.hz_to_midi(880.0) == 81  # A5

    def test_midi_to_hz(self):
        """Test MIDI to frequency conversion."""
        assert NoteBlob.midi_to_hz(69) == 440.0
        assert abs(NoteBlob.midi_to_hz(60) - 261.63) < 0.1

    def test_midi_to_note_name(self):
        """Test MIDI to note name conversion."""
        assert NoteBlob.midi_to_note_name(60) == "C4"
        assert NoteBlob.midi_to_note_name(69) == "A4"
        assert NoteBlob.midi_to_note_name(61) == "C#4"

    def test_shifted_f0(self):
        """Test pitch shift calculation."""
        f0 = np.array([440.0], dtype=np.float32)
        blob = NoteBlob(
            id=1,
            start_sample=0,
            end_sample=44100,
            sample_rate=44100,
            original_f0=f0,
        )

        # Shift up one octave (12 semitones)
        blob.shift_semitones = 12.0
        shifted = blob.shifted_f0
        assert abs(shifted[0] - 880.0) < 0.01

    def test_time_properties(self):
        """Test time-related properties."""
        f0 = np.array([440.0], dtype=np.float32)
        blob = NoteBlob(
            id=1,
            start_sample=44100,
            end_sample=88200,
            sample_rate=44100,
            original_f0=f0,
        )

        assert blob.start_time == 1.0
        assert blob.end_time == 2.0
        assert blob.duration == 1.0

        blob.stretch_ratio = 2.0
        assert blob.stretched_duration == 2.0

    def test_selection(self):
        """Test blob selection."""
        f0 = np.array([440.0], dtype=np.float32)
        blob = NoteBlob(
            id=1,
            start_sample=0,
            end_sample=44100,
            sample_rate=44100,
            original_f0=f0,
        )

        assert not blob.is_selected
        blob.select()
        assert blob.is_selected
        blob.deselect()
        assert not blob.is_selected

    def test_to_rust_params(self):
        """Test serialization for Rust."""
        f0 = np.array([440.0], dtype=np.float32)
        blob = NoteBlob(
            id=42,
            start_sample=1000,
            end_sample=2000,
            sample_rate=44100,
            original_f0=f0,
            shift_semitones=2.5,
            stretch_ratio=1.1,
        )

        params = blob.to_rust_params()
        assert params["id"] == 42
        assert params["start_sample"] == 1000
        assert params["end_sample"] == 2000
        assert params["shift_semitones"] == 2.5
        assert params["stretch_ratio"] == 1.1


class TestProject:
    """Tests for Project model."""

    def test_create_project(self):
        """Test basic project creation."""
        project = Project(name="Test")
        assert project.name == "Test"
        assert not project.has_audio
        assert project.duration == 0.0

    def test_add_blob(self):
        """Test adding blobs."""
        project = Project()
        f0 = np.array([440.0], dtype=np.float32)
        blob = NoteBlob(
            id=project.next_blob_id(),
            start_sample=0,
            end_sample=44100,
            sample_rate=44100,
            original_f0=f0,
        )

        project.add_blob(blob)
        assert len(project.blobs) == 1
        assert project.is_modified

    def test_remove_blob(self):
        """Test removing blobs."""
        project = Project()
        f0 = np.array([440.0], dtype=np.float32)
        blob = NoteBlob(
            id=1,
            start_sample=0,
            end_sample=44100,
            sample_rate=44100,
            original_f0=f0,
        )

        project.add_blob(blob)
        assert project.remove_blob(1)
        assert len(project.blobs) == 0
        assert not project.remove_blob(999)  # Non-existent

    def test_get_blob_by_id(self):
        """Test finding blobs by ID."""
        project = Project()
        f0 = np.array([440.0], dtype=np.float32)
        blob = NoteBlob(
            id=42,
            start_sample=0,
            end_sample=44100,
            sample_rate=44100,
            original_f0=f0,
        )

        project.add_blob(blob)
        found = project.get_blob_by_id(42)
        assert found is blob
        assert project.get_blob_by_id(999) is None

    def test_get_blobs_in_range(self):
        """Test finding blobs in time range."""
        project = Project()
        project.sample_rate = 44100

        f0 = np.array([440.0], dtype=np.float32)
        blob1 = NoteBlob(
            id=1, start_sample=0, end_sample=44100,
            sample_rate=44100, original_f0=f0,
        )
        blob2 = NoteBlob(
            id=2, start_sample=44100, end_sample=88200,
            sample_rate=44100, original_f0=f0,
        )
        blob3 = NoteBlob(
            id=3, start_sample=88200, end_sample=132300,
            sample_rate=44100, original_f0=f0,
        )

        project.add_blob(blob1)
        project.add_blob(blob2)
        project.add_blob(blob3)

        # Should find blobs 1 and 2 (0-2 seconds)
        found = project.get_blobs_in_range(0.0, 2.0)
        assert len(found) == 2
        assert blob1 in found
        assert blob2 in found

    def test_selection(self):
        """Test blob selection in project."""
        project = Project()
        f0 = np.array([440.0], dtype=np.float32)
        blob1 = NoteBlob(
            id=1, start_sample=0, end_sample=44100,
            sample_rate=44100, original_f0=f0,
        )
        blob2 = NoteBlob(
            id=2, start_sample=44100, end_sample=88200,
            sample_rate=44100, original_f0=f0,
        )

        project.add_blob(blob1)
        project.add_blob(blob2)

        project.select_blob(1)
        assert blob1.is_selected
        assert not blob2.is_selected

        project.select_blob(2, extend=True)
        assert blob1.is_selected
        assert blob2.is_selected

        project.deselect_all()
        assert not blob1.is_selected
        assert not blob2.is_selected

    def test_loop_region(self):
        """Test loop region settings."""
        project = Project()
        project.sample_rate = 44100

        assert not project.loop_enabled

        project.set_loop_region(1.0, 3.0)
        assert project.loop_enabled
        assert project.loop_start == 44100
        assert project.loop_end == 132300

        project.clear_loop()
        assert not project.loop_enabled
