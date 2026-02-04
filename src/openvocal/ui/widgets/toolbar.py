"""Editor toolbar with tool selection and correction macros."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QToolButton,
    QButtonGroup,
    QLabel,
    QSlider,
    QComboBox,
    QFrame,
)
from PyQt6.QtGui import QIcon

from openvocal.ui.widgets.pitch_canvas import EditTool, SnapMode


class EditorToolbar(QWidget):
    """
    Toolbar with Melodyne-style tool selection and correction macros.

    Features:
    - Main Tool (context-sensitive, like Melodyne)
    - Pitch/Time/Split specialized tools
    - Correct Pitch macro with intensity slider
    - Snap mode selector (Off/Chromatic/Scale)
    - Scale selector for scale-based snapping
    """

    tool_changed = pyqtSignal(EditTool)
    snap_mode_changed = pyqtSignal(SnapMode)
    scale_changed = pyqtSignal(int, str)  # root, scale_type
    correct_pitch_requested = pyqtSignal(float)  # intensity

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # === Tool Selection ===
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        tools = [
            (EditTool.MAIN, "Main", "Context-sensitive tool (1)\nCenter: Pitch\nEdges: Time Stretch"),
            (EditTool.PITCH, "Pitch", "Pitch adjustment only (2)"),
            (EditTool.TIME, "Time", "Time stretch only (3)"),
            (EditTool.SPLIT, "Split", "Split notes (4)"),
        ]

        for tool, name, tooltip in tools:
            btn = QToolButton()
            btn.setText(name)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedSize(50, 28)

            self.tool_group.addButton(btn, tool.value)
            layout.addWidget(btn)

            if tool == EditTool.MAIN:
                btn.setChecked(True)

        self.tool_group.idToggled.connect(self._on_tool_toggled)

        # Separator
        layout.addWidget(self._create_separator())

        # === Correct Pitch Macro ===
        layout.addWidget(QLabel("Correct:"))

        self.correct_slider = QSlider(Qt.Orientation.Horizontal)
        self.correct_slider.setRange(0, 100)
        self.correct_slider.setValue(100)
        self.correct_slider.setFixedWidth(80)
        self.correct_slider.setToolTip("Pitch correction intensity (0-100%)")
        layout.addWidget(self.correct_slider)

        self.correct_label = QLabel("100%")
        self.correct_label.setFixedWidth(35)
        layout.addWidget(self.correct_label)

        self.correct_slider.valueChanged.connect(self._on_correct_slider_changed)

        self.correct_btn = QToolButton()
        self.correct_btn.setText("Apply")
        self.correct_btn.setToolTip("Apply pitch correction to selected notes (Ctrl+Shift+P)")
        self.correct_btn.setFixedSize(45, 28)
        self.correct_btn.clicked.connect(self._on_correct_clicked)
        layout.addWidget(self.correct_btn)

        # Separator
        layout.addWidget(self._create_separator())

        # === Snap Mode ===
        layout.addWidget(QLabel("Snap:"))

        self.snap_combo = QComboBox()
        self.snap_combo.addItem("Off", SnapMode.OFF)
        self.snap_combo.addItem("Chromatic", SnapMode.CHROMATIC)
        self.snap_combo.addItem("Scale", SnapMode.SCALE)
        self.snap_combo.setCurrentIndex(1)  # Chromatic by default
        self.snap_combo.setFixedWidth(85)
        self.snap_combo.setToolTip("Pitch snapping mode")
        self.snap_combo.currentIndexChanged.connect(self._on_snap_changed)
        layout.addWidget(self.snap_combo)

        # Scale selector (enabled when Snap=Scale)
        self.scale_root_combo = QComboBox()
        roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i, root in enumerate(roots):
            self.scale_root_combo.addItem(root, i)
        self.scale_root_combo.setFixedWidth(50)
        self.scale_root_combo.setEnabled(False)
        self.scale_root_combo.currentIndexChanged.connect(self._on_scale_changed)
        layout.addWidget(self.scale_root_combo)

        self.scale_type_combo = QComboBox()
        self.scale_type_combo.addItem("Major", "major")
        self.scale_type_combo.addItem("Minor", "minor")
        self.scale_type_combo.addItem("Pentatonic", "pentatonic")
        self.scale_type_combo.addItem("Blues", "blues")
        self.scale_type_combo.setFixedWidth(80)
        self.scale_type_combo.setEnabled(False)
        self.scale_type_combo.currentIndexChanged.connect(self._on_scale_changed)
        layout.addWidget(self.scale_type_combo)

        layout.addStretch()

        # === Info Label ===
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #aaa;")
        self.info_label.setMinimumWidth(200)
        layout.addWidget(self.info_label)

    def _create_separator(self) -> QFrame:
        """Create a vertical separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    def _on_tool_toggled(self, id: int, checked: bool) -> None:
        """Handle tool selection change."""
        if checked:
            tool = EditTool(id)
            self.tool_changed.emit(tool)

    def _on_correct_slider_changed(self, value: int) -> None:
        """Handle correction intensity change."""
        self.correct_label.setText(f"{value}%")

    def _on_correct_clicked(self) -> None:
        """Handle correct pitch button click."""
        intensity = self.correct_slider.value() / 100.0
        self.correct_pitch_requested.emit(intensity)

    def _on_snap_changed(self, index: int) -> None:
        """Handle snap mode change."""
        mode = self.snap_combo.itemData(index)
        self.snap_mode_changed.emit(mode)

        # Enable/disable scale selectors
        scale_enabled = (mode == SnapMode.SCALE)
        self.scale_root_combo.setEnabled(scale_enabled)
        self.scale_type_combo.setEnabled(scale_enabled)

    def _on_scale_changed(self) -> None:
        """Handle scale change."""
        root = self.scale_root_combo.currentData()
        scale_type = self.scale_type_combo.currentData()
        self.scale_changed.emit(root, scale_type)

    def set_info(self, text: str) -> None:
        """Set the info label text."""
        self.info_label.setText(text)

    def set_tool(self, tool: EditTool) -> None:
        """Programmatically set the active tool."""
        for button in self.tool_group.buttons():
            if self.tool_group.id(button) == tool.value:
                button.setChecked(True)
                break
