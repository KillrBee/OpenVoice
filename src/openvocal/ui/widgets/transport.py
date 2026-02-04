"""Transport bar with playback controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QStyle,
)
from PyQt6.QtGui import QIcon


class TransportBar(QWidget):
    """
    Transport bar with play, stop, loop controls and time display.

    Layout:
    [Play] [Stop] [Loop] | Time: 00:00.00 / 00:00.00 | [Position Slider]
    """

    # Signals
    play_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    loop_toggled = pyqtSignal(bool)
    position_changed = pyqtSignal(float)  # time in seconds

    def __init__(self, parent=None):
        super().__init__(parent)

        self._duration = 0.0
        self._position = 0.0
        self._is_playing = False
        self._is_seeking = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Play button
        self.play_btn = QPushButton()
        self.play_btn.setFixedSize(32, 32)
        self._set_play_icon()
        self.play_btn.clicked.connect(self._on_play_clicked)
        layout.addWidget(self.play_btn)

        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setFixedSize(32, 32)
        self.stop_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.stop_btn)

        # Loop button
        self.loop_btn = QPushButton("Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setFixedWidth(50)
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        layout.addWidget(self.loop_btn)

        layout.addSpacing(16)

        # Time display
        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setFixedWidth(140)
        self.time_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.time_label)

        layout.addSpacing(16)

        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 10000)
        self.position_slider.setValue(0)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.position_slider, 1)

        self._update_time_display()

    def _set_play_icon(self) -> None:
        """Set play or pause icon based on state."""
        if self._is_playing:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        else:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.play_btn.setIcon(icon)

    def set_duration(self, duration: float) -> None:
        """Set the total duration."""
        self._duration = duration
        self._update_time_display()

    def set_position(self, position: float) -> None:
        """Set the current playback position."""
        if self._is_seeking:
            return

        self._position = position
        self._update_time_display()
        self._update_slider()

    def set_playing(self, playing: bool) -> None:
        """Set the playing state."""
        self._is_playing = playing
        self._set_play_icon()

    def _update_time_display(self) -> None:
        """Update the time label."""
        pos_str = self._format_time(self._position)
        dur_str = self._format_time(self._duration)
        self.time_label.setText(f"{pos_str} / {dur_str}")

    def _update_slider(self) -> None:
        """Update slider position."""
        if self._duration > 0:
            slider_pos = int((self._position / self._duration) * 10000)
            self.position_slider.setValue(slider_pos)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time as MM:SS.cc"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    def _on_play_clicked(self) -> None:
        """Handle play button click."""
        if self._is_playing:
            self.stop_clicked.emit()
        else:
            self.play_clicked.emit()

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self.stop_clicked.emit()
        self._position = 0.0
        self._update_time_display()
        self._update_slider()

    def _on_loop_toggled(self, checked: bool) -> None:
        """Handle loop toggle."""
        self.loop_toggled.emit(checked)

    def _on_slider_pressed(self) -> None:
        """Handle slider press (start seeking)."""
        self._is_seeking = True

    def _on_slider_released(self) -> None:
        """Handle slider release (end seeking)."""
        self._is_seeking = False
        if self._duration > 0:
            slider_value = self.position_slider.value()
            new_position = (slider_value / 10000) * self._duration
            self._position = new_position
            self.position_changed.emit(new_position)
            self._update_time_display()

    def _on_slider_moved(self, value: int) -> None:
        """Handle slider move during seeking."""
        if self._is_seeking and self._duration > 0:
            new_position = (value / 10000) * self._duration
            self._position = new_position
            self._update_time_display()
