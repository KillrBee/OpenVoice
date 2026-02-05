"""Piano roll sidebar showing pitch reference with audition capability."""

from __future__ import annotations

from typing import Optional
import math

from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QMouseEvent,
    QPaintEvent,
    QResizeEvent,
    QFont,
)

from openvocal.utils.coordinates import ViewTransform


class PianoRoll(QWidget):
    """
    Piano keyboard sidebar for pitch reference.

    Features:
    - Visual piano keys aligned with the pitch canvas
    - Click to audition (play sine tone)
    - Note names displayed on white keys
    """

    # Signals
    midi_range_changed = pyqtSignal(int, int)  # min_midi, max_midi
    note_pressed = pyqtSignal(int)  # midi note
    note_released = pyqtSignal(int)

    # Colors
    COLOR_WHITE_KEY = QColor(240, 240, 240)
    COLOR_WHITE_KEY_PRESSED = QColor(200, 220, 255)
    COLOR_BLACK_KEY = QColor(30, 30, 35)
    COLOR_BLACK_KEY_PRESSED = QColor(60, 80, 120)
    COLOR_BORDER = QColor(100, 100, 100)
    COLOR_TEXT = QColor(80, 80, 80)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.view_transform = ViewTransform()
        self._pressed_key: Optional[int] = None
        self._audio_generator: Optional[SineGenerator] = None

        # Default visible range (E2 to C6)
        self._min_midi = 40
        self._max_midi = 84

        self.setMinimumWidth(60)
        self.setMouseTracking(True)

    def update_view(
        self, y_offset: float, pixels_per_semitone: float, midi_center: float = 60.0
    ) -> None:
        """Sync view with pitch canvas."""
        self.view_transform.y_offset = y_offset
        self.view_transform.pixels_per_semitone = pixels_per_semitone
        self.view_transform.midi_center = midi_center
        self.view_transform.set_canvas_height(self.height())
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the piano keys."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.view_transform.set_canvas_height(self.height())

        # Black keys are drawn on top of white keys
        black_key_width = self.width() * 0.6
        white_key_width = self.width()

        # Black key semitones within octave
        black_keys = {1, 3, 6, 8, 10}

        # Note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Calculate visible MIDI range
        top_midi = self.view_transform.y_to_midi(0)
        bottom_midi = self.view_transform.y_to_midi(self.height())

        start_midi = int(min(top_midi, bottom_midi)) - 1
        end_midi = int(max(top_midi, bottom_midi)) + 1

        # First pass: draw white keys
        for midi in range(start_midi, end_midi + 1):
            semitone = midi % 12
            if semitone in black_keys:
                continue

            y = self.view_transform.midi_to_y(midi + 0.5)
            height = self.view_transform.pixels_per_semitone

            is_pressed = midi == self._pressed_key
            color = self.COLOR_WHITE_KEY_PRESSED if is_pressed else self.COLOR_WHITE_KEY

            rect = QRectF(0, y, white_key_width, height)
            painter.fillRect(rect, color)
            painter.setPen(QPen(self.COLOR_BORDER, 1))
            painter.drawRect(rect)

            # Draw note name on C notes
            if semitone == 0:
                octave = (midi // 12) - 1
                label = f"C{octave}"
                font = QFont()
                font.setPointSize(8)
                painter.setFont(font)
                painter.setPen(self.COLOR_TEXT)
                text_rect = QRectF(2, y, white_key_width - 4, height)
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    label,
                )

        # Second pass: draw black keys (on top)
        for midi in range(start_midi, end_midi + 1):
            semitone = midi % 12
            if semitone not in black_keys:
                continue

            y = self.view_transform.midi_to_y(midi + 0.5)
            height = self.view_transform.pixels_per_semitone

            is_pressed = midi == self._pressed_key
            color = self.COLOR_BLACK_KEY_PRESSED if is_pressed else self.COLOR_BLACK_KEY

            rect = QRectF(0, y, black_key_width, height)
            painter.fillRect(rect, color)
            painter.setPen(QPen(self.COLOR_BORDER, 1))
            painter.drawRect(rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press to play note."""
        if event.button() == Qt.MouseButton.LeftButton:
            midi = self._get_midi_at_y(event.pos().y())
            self._press_key(midi)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release to stop note."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._release_key()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for glissando effect."""
        if event.buttons() & Qt.MouseButton.LeftButton:
            midi = self._get_midi_at_y(event.pos().y())
            if midi != self._pressed_key:
                self._press_key(midi)

    def _get_midi_at_y(self, y: float) -> int:
        """Get MIDI note number at a Y coordinate."""
        self.view_transform.set_canvas_height(self.height())
        midi = self.view_transform.y_to_midi(y)
        return int(round(midi))

    def _press_key(self, midi: int) -> None:
        """Press a key and play its tone."""
        if self._pressed_key == midi:
            return

        self._pressed_key = midi
        self.note_pressed.emit(midi)
        self.update()

        # Play sine tone
        self._play_tone(midi)

    def _release_key(self) -> None:
        """Release the currently pressed key."""
        if self._pressed_key is not None:
            self.note_released.emit(self._pressed_key)
            self._pressed_key = None
            self.update()

        # Stop tone
        self._stop_tone()

    def _play_tone(self, midi: int) -> None:
        """Play a sine tone for audition."""
        if self._audio_generator is None:
            try:
                self._audio_generator = SineGenerator()
            except Exception:
                return  # Audio not available

        freq = 440.0 * (2 ** ((midi - 69) / 12.0))
        self._audio_generator.play(freq)

    def _stop_tone(self) -> None:
        """Stop the audition tone."""
        if self._audio_generator:
            self._audio_generator.stop()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle resize."""
        super().resizeEvent(event)
        self.view_transform.set_canvas_height(self.height())


class SineGenerator:
    """Simple sine wave generator for piano audition."""

    def __init__(self):
        self._stream = None
        self._playing = False
        self._frequency = 440.0
        self._phase = 0.0
        self._sample_rate = 44100
        self._amplitude = 0.3
        self._sounddevice = None

        self._init_audio()

    def _init_audio(self) -> None:
        """Initialize audio output using sounddevice."""
        try:
            import sounddevice as sd
            self._sounddevice = sd
        except ImportError:
            # sounddevice not available - audition won't work
            self._sounddevice = None

    def _audio_callback(self, outdata, frames, time_info, status):
        """Generate sine wave samples for the audio stream."""
        import numpy as np

        if not self._playing:
            outdata.fill(0)
            return

        # Generate sine wave
        t = (np.arange(frames) + self._phase) / self._sample_rate
        samples = self._amplitude * np.sin(2 * np.pi * self._frequency * t)

        # Apply simple envelope to reduce clicks
        if frames > 100:
            # Fade in first 50 samples
            samples[:50] *= np.linspace(0, 1, 50)

        outdata[:, 0] = samples.astype(np.float32)

        # Update phase for continuity
        self._phase += frames

    def play(self, frequency: float) -> None:
        """Start playing a frequency."""
        if self._sounddevice is None:
            return

        self._frequency = frequency
        self._phase = 0.0
        self._playing = True

        # Stop any existing stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

        # Create new stream
        try:
            self._stream = self._sounddevice.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=512,
            )
            self._stream.start()
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            self._playing = False

    def stop(self) -> None:
        """Stop playing."""
        self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def __del__(self):
        self.stop()
