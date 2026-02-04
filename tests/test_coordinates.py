"""Tests for coordinate transformations."""

import pytest
from openvocal.utils.coordinates import (
    ViewTransform,
    hz_to_midi,
    midi_to_hz,
    time_to_x,
    x_to_time,
)


class TestViewTransform:
    """Tests for ViewTransform class."""

    def test_time_to_x_roundtrip(self):
        """Test time <-> X conversion roundtrip."""
        vt = ViewTransform(pixels_per_second=100)

        for time in [0.0, 1.0, 5.5, 10.0]:
            x = vt.time_to_x(time)
            result = vt.x_to_time(x)
            assert abs(result - time) < 0.001

    def test_midi_to_y_roundtrip(self):
        """Test MIDI <-> Y conversion roundtrip."""
        vt = ViewTransform(pixels_per_semitone=20)
        vt.set_canvas_height(600)

        for midi in [48, 60, 72, 84]:
            y = vt.midi_to_y(midi)
            result = vt.y_to_midi(y)
            assert abs(result - midi) < 0.001

    def test_zoom_horizontal(self):
        """Test horizontal zooming."""
        vt = ViewTransform(pixels_per_second=100)

        initial_pps = vt.pixels_per_second
        vt.zoom_horizontal(2.0, 0)  # Double zoom

        assert vt.pixels_per_second == initial_pps * 2

    def test_zoom_vertical(self):
        """Test vertical zooming."""
        vt = ViewTransform(pixels_per_semitone=20)
        vt.set_canvas_height(600)

        initial_pps = vt.pixels_per_semitone
        vt.zoom_vertical(2.0, 300)  # Double zoom at center

        assert vt.pixels_per_semitone == initial_pps * 2

    def test_scroll(self):
        """Test scrolling."""
        vt = ViewTransform()

        vt.scroll(100, 50)
        assert vt.x_offset == 100
        assert vt.y_offset == 50

        # X offset should not go negative
        vt.scroll(-200, 0)
        assert vt.x_offset == 0


class TestConversionFunctions:
    """Tests for standalone conversion functions."""

    def test_hz_to_midi(self):
        """Test Hz to MIDI conversion."""
        assert abs(hz_to_midi(440.0) - 69.0) < 0.001  # A4
        assert abs(hz_to_midi(880.0) - 81.0) < 0.001  # A5
        assert abs(hz_to_midi(220.0) - 57.0) < 0.001  # A3

    def test_midi_to_hz(self):
        """Test MIDI to Hz conversion."""
        assert abs(midi_to_hz(69) - 440.0) < 0.001
        assert abs(midi_to_hz(81) - 880.0) < 0.001

    def test_time_x_functions(self):
        """Test time/X convenience functions."""
        assert time_to_x(1.0, pixels_per_second=100) == 100.0
        assert x_to_time(100.0, pixels_per_second=100) == 1.0
