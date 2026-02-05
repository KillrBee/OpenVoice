"""Coordinate transformation utilities for the pitch editor canvas."""

import numpy as np


# Default view parameters
DEFAULT_PIXELS_PER_SECOND = 100.0  # Horizontal zoom
DEFAULT_PIXELS_PER_SEMITONE = 12.0  # Vertical zoom (reduced for better default view)
DEFAULT_MIDI_CENTER = 60  # Middle C as default center

# Zoom limits
MIN_PIXELS_PER_SECOND = 5.0  # Allow zooming out to see long files
MAX_PIXELS_PER_SECOND = 2000.0  # Allow detailed zoom in
MIN_PIXELS_PER_SEMITONE = 2.0  # Allow compressing vertical axis significantly
MAX_PIXELS_PER_SEMITONE = 100.0  # Maximum vertical zoom


class ViewTransform:
    """
    Manages coordinate transformations between:
    - Time (seconds) <-> X pixels
    - Pitch (MIDI/Hz) <-> Y pixels

    The canvas uses:
    - X-axis: Linear time scale
    - Y-axis: Logarithmic pitch scale (linear in semitones/MIDI)

    Attributes:
        pixels_per_second: Horizontal zoom level
        pixels_per_semitone: Vertical zoom level
        x_offset: Horizontal scroll offset in pixels
        y_offset: Vertical scroll offset in pixels (from MIDI center)
        midi_center: The MIDI note at the vertical center of the view
    """

    def __init__(
        self,
        pixels_per_second: float = DEFAULT_PIXELS_PER_SECOND,
        pixels_per_semitone: float = DEFAULT_PIXELS_PER_SEMITONE,
        midi_center: int = DEFAULT_MIDI_CENTER,
    ):
        self.pixels_per_second = pixels_per_second
        self.pixels_per_semitone = pixels_per_semitone
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.midi_center = midi_center
        self._canvas_height = 600  # Will be updated by canvas

    def set_canvas_height(self, height: int) -> None:
        """Update canvas height for Y coordinate calculations."""
        self._canvas_height = height

    # === Time <-> X conversions ===

    def time_to_x(self, time_seconds: float) -> float:
        """Convert time in seconds to X pixel coordinate."""
        return time_seconds * self.pixels_per_second - self.x_offset

    def x_to_time(self, x: float) -> float:
        """Convert X pixel coordinate to time in seconds."""
        return (x + self.x_offset) / self.pixels_per_second

    # === MIDI <-> Y conversions ===

    def midi_to_y(self, midi_note: float) -> float:
        """
        Convert MIDI note number to Y pixel coordinate.

        Higher pitches are at the top (lower Y values).
        """
        semitones_from_center = midi_note - self.midi_center
        y_from_center = -semitones_from_center * self.pixels_per_semitone
        return self._canvas_height / 2 + y_from_center - self.y_offset

    def y_to_midi(self, y: float) -> float:
        """
        Convert Y pixel coordinate to MIDI note number.

        Can return fractional MIDI for cent-level precision.
        """
        y_from_center = y - self._canvas_height / 2 + self.y_offset
        semitones_from_center = -y_from_center / self.pixels_per_semitone
        return self.midi_center + semitones_from_center

    # === Hz <-> Y conversions (via MIDI) ===

    def hz_to_y(self, freq_hz: float) -> float:
        """Convert frequency in Hz to Y pixel coordinate."""
        if freq_hz <= 0:
            return self._canvas_height  # Off-screen bottom for unvoiced
        midi = hz_to_midi(freq_hz)
        return self.midi_to_y(midi)

    def y_to_hz(self, y: float) -> float:
        """Convert Y pixel coordinate to frequency in Hz."""
        midi = self.y_to_midi(y)
        return midi_to_hz(midi)

    # === Zoom operations ===

    def zoom_horizontal(self, factor: float, anchor_x: float) -> None:
        """
        Zoom horizontally around an anchor point.

        Args:
            factor: Zoom factor (> 1 = zoom in, < 1 = zoom out)
            anchor_x: X coordinate to keep fixed during zoom
        """
        # Get time at anchor before zoom
        anchor_time = self.x_to_time(anchor_x)

        # Apply zoom with configurable limits
        self.pixels_per_second *= factor
        self.pixels_per_second = max(
            MIN_PIXELS_PER_SECOND, min(MAX_PIXELS_PER_SECOND, self.pixels_per_second)
        )

        # Adjust offset to keep anchor time at same X position
        self.x_offset = anchor_time * self.pixels_per_second - anchor_x

    def zoom_vertical(self, factor: float, anchor_y: float) -> None:
        """
        Zoom vertically around an anchor point.

        Args:
            factor: Zoom factor (> 1 = zoom in, < 1 = zoom out)
            anchor_y: Y coordinate to keep fixed during zoom
        """
        # Get MIDI at anchor before zoom
        anchor_midi = self.y_to_midi(anchor_y)

        # Apply zoom with configurable limits
        self.pixels_per_semitone *= factor
        self.pixels_per_semitone = max(
            MIN_PIXELS_PER_SEMITONE, min(MAX_PIXELS_PER_SEMITONE, self.pixels_per_semitone)
        )

        # Adjust offset to keep anchor MIDI at same Y position
        semitones_from_center = anchor_midi - self.midi_center
        new_y_from_center = -semitones_from_center * self.pixels_per_semitone
        target_y_from_center = anchor_y - self._canvas_height / 2
        self.y_offset = new_y_from_center - target_y_from_center

    def scroll(self, dx: float, dy: float) -> None:
        """Scroll the view by pixel amounts."""
        self.x_offset += dx
        self.y_offset += dy

        # Clamp x_offset to not scroll before time 0
        self.x_offset = max(0, self.x_offset)

    def scroll_to_time(self, time_seconds: float, canvas_width: float) -> None:
        """Scroll to center a specific time in the view."""
        center_offset = canvas_width / 2
        self.x_offset = time_seconds * self.pixels_per_second - center_offset
        self.x_offset = max(0, self.x_offset)

    def scroll_to_midi(self, midi_note: float, canvas_height: float) -> None:
        """Scroll to center a specific MIDI note in the view."""
        self.midi_center = midi_note
        self.y_offset = 0


# === Module-level convenience functions ===

def hz_to_midi(freq_hz: float) -> float:
    """Convert frequency in Hz to MIDI note number (can be fractional)."""
    if freq_hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq_hz / 440.0)


def midi_to_hz(midi_note: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi_note - 69.0) / 12.0))


def time_to_x(
    time_seconds: float,
    pixels_per_second: float = DEFAULT_PIXELS_PER_SECOND,
    x_offset: float = 0.0,
) -> float:
    """Convenience function: convert time to X coordinate."""
    return time_seconds * pixels_per_second - x_offset


def x_to_time(
    x: float,
    pixels_per_second: float = DEFAULT_PIXELS_PER_SECOND,
    x_offset: float = 0.0,
) -> float:
    """Convenience function: convert X coordinate to time."""
    return (x + x_offset) / pixels_per_second


def midi_to_y(
    midi_note: float,
    canvas_height: float,
    pixels_per_semitone: float = DEFAULT_PIXELS_PER_SEMITONE,
    midi_center: float = DEFAULT_MIDI_CENTER,
    y_offset: float = 0.0,
) -> float:
    """Convenience function: convert MIDI note to Y coordinate."""
    semitones_from_center = midi_note - midi_center
    y_from_center = -semitones_from_center * pixels_per_semitone
    return canvas_height / 2 + y_from_center - y_offset


def y_to_midi(
    y: float,
    canvas_height: float,
    pixels_per_semitone: float = DEFAULT_PIXELS_PER_SEMITONE,
    midi_center: float = DEFAULT_MIDI_CENTER,
    y_offset: float = 0.0,
) -> float:
    """Convenience function: convert Y coordinate to MIDI note."""
    y_from_center = y - canvas_height / 2 + y_offset
    semitones_from_center = -y_from_center / pixels_per_semitone
    return midi_center + semitones_from_center


def hz_to_y(
    freq_hz: float,
    canvas_height: float,
    pixels_per_semitone: float = DEFAULT_PIXELS_PER_SEMITONE,
    midi_center: float = DEFAULT_MIDI_CENTER,
    y_offset: float = 0.0,
) -> float:
    """Convenience function: convert Hz to Y coordinate."""
    midi = hz_to_midi(freq_hz)
    return midi_to_y(midi, canvas_height, pixels_per_semitone, midi_center, y_offset)


def y_to_hz(
    y: float,
    canvas_height: float,
    pixels_per_semitone: float = DEFAULT_PIXELS_PER_SEMITONE,
    midi_center: float = DEFAULT_MIDI_CENTER,
    y_offset: float = 0.0,
) -> float:
    """Convenience function: convert Y coordinate to Hz."""
    midi = y_to_midi(y, canvas_height, pixels_per_semitone, midi_center, y_offset)
    return midi_to_hz(midi)
