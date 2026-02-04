"""High-performance pitch editor canvas using QGraphicsScene."""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, List
import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsRectItem,
    QGraphicsPathItem,
    QGraphicsLineItem,
    QGraphicsItem,
    QGraphicsTextItem,
    QToolTip,
)
from PyQt6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QPainterPath,
    QWheelEvent,
    QMouseEvent,
    QKeyEvent,
    QFont,
    QCursor,
)

from openvocal.models.project import Project
from openvocal.models.note_blob import NoteBlob
from openvocal.utils.coordinates import ViewTransform


class EditTool(Enum):
    """Available editing tools."""
    MAIN = auto()      # Context-sensitive main tool (like Melodyne)
    SELECT = auto()
    PITCH = auto()
    TIME = auto()
    SPLIT = auto()
    MODULATION = auto()  # Vibrato/pitch drift editing


class DragMode(Enum):
    """Current drag operation mode."""
    NONE = auto()
    PITCH = auto()          # Vertical drag = pitch shift
    TIME = auto()           # Horizontal drag = time shift
    STRETCH_LEFT = auto()   # Drag left edge = time stretch
    STRETCH_RIGHT = auto()  # Drag right edge = time stretch
    MODULATION = auto()     # Modulation/vibrato adjustment


class SnapMode(Enum):
    """Pitch snapping modes."""
    OFF = auto()
    CHROMATIC = auto()  # Snap to nearest semitone
    SCALE = auto()      # Snap to scale notes


class BlobItem(QGraphicsItem):
    """
    Visual representation of a NoteBlob with Melodyne-style interaction.

    Displays three layers:
    1. Energy envelope (blob shape/thickness)
    2. Pitch curve (f0 line over the blob)
    3. Quantized anchor (snap target rectangle)

    Context-sensitive behavior:
    - Center: Pitch/time adjustment
    - Edges: Time stretching
    - Top/bottom: Modulation editing
    """

    # Colors
    COLOR_BLOB = QColor(100, 149, 237, 180)  # Cornflower blue
    COLOR_BLOB_SELECTED = QColor(255, 140, 0, 200)  # Dark orange (like Melodyne red)
    COLOR_BLOB_HOVER = QColor(130, 170, 255, 190)
    COLOR_PITCH_CURVE = QColor(255, 255, 255, 220)
    COLOR_PITCH_CURVE_MODIFIED = QColor(255, 200, 100, 220)
    COLOR_ANCHOR = QColor(255, 255, 255, 60)
    COLOR_ANCHOR_LINE = QColor(255, 255, 255, 100)
    COLOR_HANDLE = QColor(255, 255, 255, 200)
    COLOR_DRAG_GUIDE = QColor(100, 255, 100, 150)

    # Interaction zones (in pixels)
    EDGE_ZONE = 12  # Width of edge drag zones
    TOP_BOTTOM_ZONE = 8  # Height of modulation zones

    def __init__(
        self,
        blob: NoteBlob,
        view_transform: ViewTransform,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.blob = blob
        self.view_transform = view_transform

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)

        # Interaction state
        self._hover_zone: Optional[DragMode] = None
        self._is_dragging = False
        self._drag_mode = DragMode.NONE
        self._drag_start_pos = QPointF()
        self._drag_start_shift = 0.0
        self._drag_start_stretch = 1.0
        self._is_hovered = False

        # Pitch correction preview
        self._preview_shift = 0.0
        self._show_correction_preview = False

        self._update_geometry()

    def _update_geometry(self) -> None:
        """Update item geometry based on blob data and view transform."""
        self.prepareGeometryChange()

        # Calculate bounding rect in scene coordinates
        x1 = self.view_transform.time_to_x(self.blob.start_time)
        x2 = self.view_transform.time_to_x(self.blob.end_time)

        # Get pitch range for height
        if len(self.blob.original_f0) > 0:
            voiced_f0 = self.blob.original_f0[self.blob.original_f0 > 0]
            if len(voiced_f0) > 0:
                min_midi = NoteBlob.hz_to_midi(np.min(voiced_f0)) + self.blob.shift_semitones
                max_midi = NoteBlob.hz_to_midi(np.max(voiced_f0)) + self.blob.shift_semitones
            else:
                min_midi = max_midi = self.blob.shifted_midi
        else:
            min_midi = max_midi = self.blob.shifted_midi

        # Add padding
        min_midi -= 0.5
        max_midi += 0.5

        y1 = self.view_transform.midi_to_y(max_midi)  # Higher pitch = lower y
        y2 = self.view_transform.midi_to_y(min_midi)

        self._bounds = QRectF(x1, y1, x2 - x1, y2 - y1)

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle."""
        return self._bounds

    def paint(
        self,
        painter: QPainter,
        option,
        widget=None,
    ) -> None:
        """Paint the blob item."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        is_selected = self.blob.is_selected or self.isSelected()

        # Determine blob color based on state
        if is_selected:
            blob_color = self.COLOR_BLOB_SELECTED
        elif self._is_hovered:
            blob_color = self.COLOR_BLOB_HOVER
        else:
            blob_color = self.COLOR_BLOB

        # 1. Draw anchor rectangle (quantized note target)
        self._draw_anchor(painter)

        # 2. Draw blob shape (energy envelope)
        self._draw_blob_shape(painter, blob_color)

        # 3. Draw pitch curve
        self._draw_pitch_curve(painter, is_selected)

        # 4. Draw interaction handles if hovered or selected
        if self._is_hovered or is_selected:
            self._draw_handles(painter)

        # 5. Draw drag guide during drag operations
        if self._is_dragging:
            self._draw_drag_guide(painter)

    def _draw_anchor(self, painter: QPainter) -> None:
        """Draw the quantized note anchor rectangle."""
        anchor_y = self.view_transform.midi_to_y(self.blob.shifted_midi + 0.5)
        anchor_height = self.view_transform.pixels_per_semitone
        anchor_rect = QRectF(
            self._bounds.x(),
            anchor_y,
            self._bounds.width(),
            anchor_height,
        )

        painter.setPen(QPen(self.COLOR_ANCHOR_LINE, 1))
        painter.setBrush(QBrush(self.COLOR_ANCHOR))
        painter.drawRect(anchor_rect)

        # Draw note name on anchor
        if self._bounds.width() > 40:
            note_name = NoteBlob.midi_to_note_name(int(self.blob.shifted_midi))
            painter.setPen(QColor(200, 200, 200, 150))
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(
                anchor_rect.adjusted(4, 0, 0, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                note_name
            )

    def _draw_blob_shape(self, painter: QPainter, color: QColor) -> None:
        """Draw the blob shape based on amplitude envelope."""
        if self.blob.amplitude_envelope is None or len(self.blob.amplitude_envelope) == 0:
            # Simple rounded rectangle if no envelope
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(self._bounds, 4, 4)
            return

        # Build path from envelope
        path = QPainterPath()
        env = self.blob.amplitude_envelope
        max_amp = np.max(env) if np.max(env) > 0 else 1.0

        # Normalize envelope
        normalized_env = env / max_amp

        # Map envelope to path
        num_points = len(normalized_env)
        width = self._bounds.width()
        center_y = self._bounds.center().y()
        max_height = self._bounds.height() / 2

        # Top edge (left to right)
        path.moveTo(self._bounds.x(), center_y - normalized_env[0] * max_height)
        for i in range(1, num_points):
            x = self._bounds.x() + (i / num_points) * width
            y = center_y - normalized_env[i] * max_height
            path.lineTo(x, y)

        # Bottom edge (right to left)
        for i in range(num_points - 1, -1, -1):
            x = self._bounds.x() + (i / num_points) * width
            y = center_y + normalized_env[i] * max_height
            path.lineTo(x, y)

        path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    def _draw_pitch_curve(self, painter: QPainter, is_selected: bool) -> None:
        """Draw the pitch (f0) curve over the blob."""
        if len(self.blob.original_f0) == 0:
            return

        f0 = self.blob.shifted_f0
        voiced_mask = f0 > 0

        if not np.any(voiced_mask):
            return

        path = QPainterPath()
        width = self._bounds.width()
        num_points = len(f0)

        started = False
        for i in range(num_points):
            if not voiced_mask[i]:
                if started:
                    started = False
                continue

            x = self._bounds.x() + (i / num_points) * width
            midi = NoteBlob.hz_to_midi(f0[i])
            y = self.view_transform.midi_to_y(midi)

            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)

        # Use different color if pitch has been modified
        if abs(self.blob.shift_semitones) > 0.01:
            curve_color = self.COLOR_PITCH_CURVE_MODIFIED
        else:
            curve_color = self.COLOR_PITCH_CURVE

        pen = QPen(curve_color, 2.5 if is_selected else 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _draw_handles(self, painter: QPainter) -> None:
        """Draw interaction handles."""
        handle_width = 6

        # Left edge handle
        left_rect = QRectF(
            self._bounds.x(),
            self._bounds.y(),
            handle_width,
            self._bounds.height(),
        )
        # Right edge handle
        right_rect = QRectF(
            self._bounds.right() - handle_width,
            self._bounds.y(),
            handle_width,
            self._bounds.height(),
        )

        # Highlight active zone
        if self._hover_zone == DragMode.STRETCH_LEFT:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.COLOR_HANDLE))
            painter.drawRect(left_rect)
        elif self._hover_zone == DragMode.STRETCH_RIGHT:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.COLOR_HANDLE))
            painter.drawRect(right_rect)

        # Draw subtle edge indicators
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
        painter.drawLine(
            QPointF(self._bounds.x() + 2, self._bounds.top() + 4),
            QPointF(self._bounds.x() + 2, self._bounds.bottom() - 4)
        )
        painter.drawLine(
            QPointF(self._bounds.right() - 2, self._bounds.top() + 4),
            QPointF(self._bounds.right() - 2, self._bounds.bottom() - 4)
        )

    def _draw_drag_guide(self, painter: QPainter) -> None:
        """Draw visual guide during drag operations."""
        if self._drag_mode == DragMode.PITCH:
            # Draw horizontal guide line at current pitch
            midi = self.blob.shifted_midi
            y = self.view_transform.midi_to_y(midi)
            painter.setPen(QPen(self.COLOR_DRAG_GUIDE, 1, Qt.PenStyle.DashLine))
            painter.drawLine(
                QPointF(self._bounds.x() - 50, y),
                QPointF(self._bounds.right() + 50, y)
            )

    def _get_zone_at_pos(self, pos: QPointF) -> DragMode:
        """Determine which interaction zone the position is in."""
        x = pos.x()
        y = pos.y()

        # Check left edge
        if x - self._bounds.x() < self.EDGE_ZONE:
            return DragMode.STRETCH_LEFT

        # Check right edge
        if self._bounds.right() - x < self.EDGE_ZONE:
            return DragMode.STRETCH_RIGHT

        # Check top/bottom for modulation
        if y - self._bounds.top() < self.TOP_BOTTOM_ZONE:
            return DragMode.MODULATION
        if self._bounds.bottom() - y < self.TOP_BOTTOM_ZONE:
            return DragMode.MODULATION

        # Center = pitch/time adjustment
        return DragMode.PITCH

    def hoverEnterEvent(self, event) -> None:
        """Handle hover enter."""
        self._is_hovered = True
        self.update()

    def hoverMoveEvent(self, event) -> None:
        """Track mouse position for context-sensitive cursor."""
        self._hover_zone = self._get_zone_at_pos(event.pos())

        # Set appropriate cursor
        if self._hover_zone in (DragMode.STRETCH_LEFT, DragMode.STRETCH_RIGHT):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._hover_zone == DragMode.MODULATION:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif self._hover_zone == DragMode.PITCH:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.update()

    def hoverLeaveEvent(self, event) -> None:
        """Clear hover state."""
        self._is_hovered = False
        self._hover_zone = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def update_from_blob(self) -> None:
        """Update visual from blob data changes."""
        self._update_geometry()
        self.update()


class LocalPitchRuler(QGraphicsItem):
    """
    Local pitch ruler that appears above hovered blobs.
    Shows note names and cents deviation.
    """

    def __init__(self, view_transform: ViewTransform):
        super().__init__()
        self.view_transform = view_transform
        self._target_blob: Optional[NoteBlob] = None
        self._visible = False
        self.setZValue(500)

    def set_target(self, blob: Optional[NoteBlob], x: float, width: float) -> None:
        """Set the blob to show ruler for."""
        self._target_blob = blob
        self._x = x
        self._width = width
        self._visible = blob is not None
        self.update()

    def boundingRect(self) -> QRectF:
        if not self._visible:
            return QRectF()
        return QRectF(self._x, 0, self._width, 30)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._visible or not self._target_blob:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        rect = self.boundingRect()
        painter.fillRect(rect, QColor(40, 40, 45, 230))

        # Draw note and cents
        midi = self._target_blob.shifted_midi
        note_name = NoteBlob.midi_to_note_name(int(round(midi)))
        cents = int((midi - round(midi)) * 100)
        cents_str = f"+{cents}" if cents > 0 else str(cents)

        painter.setPen(QColor(255, 255, 255))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect.adjusted(8, 0, 0, 0),
                        Qt.AlignmentFlag.AlignVCenter, f"{note_name} {cents_str}¢")


class PitchCanvas(QGraphicsView):
    """
    Main pitch editing canvas with Melodyne-style interaction.

    Features:
    - Context-sensitive Main Tool (cursor changes based on position)
    - Pitch snapping (chromatic/scale)
    - Multi-note selection with Shift+click
    - Correct Pitch macro support
    - Visual feedback (drag guides, pitch ruler)
    """

    # Signals
    blob_selected = pyqtSignal(int)  # blob_id
    blob_modified = pyqtSignal(int)  # blob_id
    view_changed = pyqtSignal(float, float)  # y_offset, pixels_per_semitone
    status_message = pyqtSignal(str)  # Status bar message

    def __init__(self, parent=None):
        super().__init__(parent)

        self.project: Optional[Project] = None
        self.view_transform = ViewTransform()
        self._blob_items: dict[int, BlobItem] = {}
        self._current_tool = EditTool.MAIN
        self._playhead_position = 0.0

        # Editing state
        self._snap_mode = SnapMode.CHROMATIC
        self._scale_root = 0  # C
        self._scale_type = "major"
        self._scale_notes = {0, 2, 4, 5, 7, 9, 11}  # Major scale

        # Drag state
        self._dragging_item: Optional[BlobItem] = None
        self._drag_mode = DragMode.NONE
        self._drag_start_pos = QPointF()
        self._drag_start_value = 0.0
        self._alt_held = False  # Fine-tune mode

        self._setup_scene()
        self._setup_view()

    def _setup_scene(self) -> None:
        """Initialize the graphics scene."""
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Set initial scene rect
        self._scene.setSceneRect(0, 0, 10000, 2000)

        # Playhead line
        self._playhead = QGraphicsLineItem()
        self._playhead.setPen(QPen(QColor(255, 100, 100), 2))
        self._playhead.setZValue(1000)
        self._scene.addItem(self._playhead)
        self._update_playhead()

        # Local pitch ruler
        self._pitch_ruler = LocalPitchRuler(self.view_transform)
        self._scene.addItem(self._pitch_ruler)

    def _setup_view(self) -> None:
        """Configure view settings for performance."""
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )
        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
        )
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw piano key background grid."""
        painter.fillRect(rect, QColor(30, 30, 35))

        # Draw horizontal bands for piano keys
        self.view_transform.set_canvas_height(self.height())

        # Determine visible MIDI range
        top_midi = self.view_transform.y_to_midi(rect.top())
        bottom_midi = self.view_transform.y_to_midi(rect.bottom())

        start_midi = int(min(top_midi, bottom_midi)) - 1
        end_midi = int(max(top_midi, bottom_midi)) + 1

        # Black keys (C#, D#, F#, G#, A#)
        black_keys = {1, 3, 6, 8, 10}

        for midi in range(start_midi, end_midi + 1):
            y = self.view_transform.midi_to_y(midi + 0.5)
            height = self.view_transform.pixels_per_semitone

            semitone = midi % 12
            if semitone in black_keys:
                color = QColor(25, 25, 30)
            else:
                color = QColor(35, 35, 40)

            # Highlight scale notes if snap to scale is enabled
            if self._snap_mode == SnapMode.SCALE:
                relative_note = (midi - self._scale_root) % 12
                if relative_note in self._scale_notes:
                    color = color.lighter(115)

            painter.fillRect(QRectF(rect.x(), y, rect.width(), height), color)

            # Draw subtle grid line
            painter.setPen(QPen(QColor(50, 50, 55), 1))
            painter.drawLine(QPointF(rect.x(), y), QPointF(rect.right(), y))

        # Draw vertical time grid
        self._draw_time_grid(painter, rect)

    def _draw_time_grid(self, painter: QPainter, rect: QRectF) -> None:
        """Draw vertical time grid lines."""
        pixels_per_second = self.view_transform.pixels_per_second

        # Choose interval based on zoom
        if pixels_per_second > 200:
            interval = 0.1
        elif pixels_per_second > 50:
            interval = 0.5
        elif pixels_per_second > 20:
            interval = 1.0
        elif pixels_per_second > 5:
            interval = 5.0
        else:
            interval = 10.0

        start_time = self.view_transform.x_to_time(rect.left())
        end_time = self.view_transform.x_to_time(rect.right())

        start_time = max(0, int(start_time / interval) * interval)

        time = start_time
        while time <= end_time:
            x = self.view_transform.time_to_x(time)

            # Major line every second
            if abs(time - round(time)) < 0.01:
                painter.setPen(QPen(QColor(60, 60, 65), 1))
            else:
                painter.setPen(QPen(QColor(45, 45, 50), 1))

            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            time += interval

    def set_project(self, project: Project) -> None:
        """Set the project and refresh blobs."""
        self.project = project

        # Clear existing blob items
        for item in self._blob_items.values():
            self._scene.removeItem(item)
        self._blob_items.clear()

        # Create blob items
        for blob in project.blobs:
            self._add_blob_item(blob)

        # Fit view to content
        self.zoom_fit()

    def _add_blob_item(self, blob: NoteBlob) -> BlobItem:
        """Add a visual item for a blob."""
        item = BlobItem(blob, self.view_transform)
        self._scene.addItem(item)
        self._blob_items[blob.id] = item
        return item

    def set_playhead_position(self, time_seconds: float) -> None:
        """Update playhead position."""
        self._playhead_position = time_seconds
        self._update_playhead()

    def _update_playhead(self) -> None:
        """Update playhead line geometry."""
        x = self.view_transform.time_to_x(self._playhead_position)
        scene_rect = self._scene.sceneRect()
        self._playhead.setLine(x, scene_rect.top(), x, scene_rect.bottom())

    def set_tool(self, tool: EditTool) -> None:
        """Set the current editing tool."""
        self._current_tool = tool

    def set_snap_mode(self, mode: SnapMode) -> None:
        """Set pitch snapping mode."""
        self._snap_mode = mode
        self.resetCachedContent()
        self.viewport().update()

    def set_scale(self, root: int, scale_type: str) -> None:
        """Set the scale for scale-based snapping."""
        self._scale_root = root
        self._scale_type = scale_type

        # Define scale intervals
        scales = {
            "major": {0, 2, 4, 5, 7, 9, 11},
            "minor": {0, 2, 3, 5, 7, 8, 10},
            "pentatonic": {0, 2, 4, 7, 9},
            "blues": {0, 3, 5, 6, 7, 10},
        }
        self._scale_notes = scales.get(scale_type, scales["major"])

        self.resetCachedContent()
        self.viewport().update()

    def _snap_pitch(self, midi: float) -> float:
        """Snap pitch value according to current snap mode."""
        if self._snap_mode == SnapMode.OFF:
            return midi
        elif self._snap_mode == SnapMode.CHROMATIC:
            return round(midi)
        elif self._snap_mode == SnapMode.SCALE:
            # Find nearest scale note
            base_note = int(midi)
            cents = midi - base_note

            for offset in range(7):  # Search within half octave
                for direction in [0, 1, -1]:
                    test_note = base_note + offset * direction
                    relative = (test_note - self._scale_root) % 12
                    if relative in self._scale_notes:
                        return float(test_note)

            return round(midi)  # Fallback

        return midi

    def correct_pitch(self, intensity: float = 1.0) -> None:
        """
        Apply automatic pitch correction to selected blobs.

        Args:
            intensity: Correction intensity from 0.0 (none) to 1.0 (full)
        """
        if not self.project:
            return

        for blob in self.project.get_selected_blobs():
            # Calculate target pitch (quantized)
            target_midi = self._snap_pitch(blob.quantized_midi or 60)
            current_midi = blob.shifted_midi

            # Calculate correction
            correction = (target_midi - current_midi) * intensity
            blob.shift_semitones += correction

            # Update visual
            if blob.id in self._blob_items:
                self._blob_items[blob.id].update_from_blob()

            self.blob_modified.emit(blob.id)

        self.status_message.emit(f"Pitch corrected at {int(intensity * 100)}% intensity")

    def set_midi_range(self, min_midi: int, max_midi: int) -> None:
        """Set the visible MIDI note range."""
        center = (min_midi + max_midi) / 2
        self.view_transform.midi_center = center
        self._refresh_all_blobs()
        self.viewport().update()

    def update_view(self) -> None:
        """Refresh view after external changes."""
        self._refresh_all_blobs()
        self.viewport().update()

    def _refresh_all_blobs(self) -> None:
        """Update all blob items from their data."""
        for item in self._blob_items.values():
            item.update_from_blob()

    # === Zoom and Pan ===

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zoom and scroll."""
        modifiers = event.modifiers()
        delta = event.angleDelta().y()

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            # Horizontal zoom
            factor = 1.1 if delta > 0 else 0.9
            pos = self.mapToScene(event.position().toPoint())
            self.view_transform.zoom_horizontal(factor, pos.x())
            self._refresh_all_blobs()
            self._update_playhead()
            self.resetCachedContent()
            self.viewport().update()

        elif modifiers & Qt.KeyboardModifier.AltModifier:
            # Vertical zoom
            factor = 1.1 if delta > 0 else 0.9
            pos = self.mapToScene(event.position().toPoint())
            self.view_transform.zoom_vertical(factor, pos.y())
            self._refresh_all_blobs()
            self.resetCachedContent()
            self.viewport().update()
            self.view_changed.emit(
                self.view_transform.y_offset,
                self.view_transform.pixels_per_semitone,
            )

        else:
            # Vertical scroll
            scroll_amount = -delta / 4
            self.view_transform.scroll(0, scroll_amount)
            self._refresh_all_blobs()
            self.resetCachedContent()
            self.viewport().update()
            self.view_changed.emit(
                self.view_transform.y_offset,
                self.view_transform.pixels_per_semitone,
            )

    def zoom_in(self) -> None:
        """Zoom in horizontally."""
        center = self.mapToScene(self.viewport().rect().center())
        self.view_transform.zoom_horizontal(1.2, center.x())
        self._refresh_all_blobs()
        self._update_playhead()
        self.resetCachedContent()
        self.viewport().update()

    def zoom_out(self) -> None:
        """Zoom out horizontally."""
        center = self.mapToScene(self.viewport().rect().center())
        self.view_transform.zoom_horizontal(0.8, center.x())
        self._refresh_all_blobs()
        self._update_playhead()
        self.resetCachedContent()
        self.viewport().update()

    def zoom_fit(self) -> None:
        """Fit view to show all content."""
        if not self.project or not self.project.has_audio:
            return

        # Calculate pixels per second to fit duration
        duration = self.project.duration
        if duration > 0:
            self.view_transform.pixels_per_second = (
                self.viewport().width() * 0.9 / duration
            )
            self.view_transform.x_offset = 0

        # Fit vertical to show vocal range
        self.view_transform.midi_center = 60  # Middle C
        self.view_transform.pixels_per_semitone = self.viewport().height() / 36

        self._refresh_all_blobs()
        self._update_playhead()
        self.resetCachedContent()
        self.viewport().update()

    # === Mouse Events ===

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for selection and dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self._scene.itemAt(scene_pos, self.transform())

            if isinstance(item, BlobItem):
                # Check modifiers for selection behavior
                extend_selection = bool(
                    event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                )

                if not extend_selection:
                    if self.project:
                        self.project.deselect_all()

                item.blob.select()
                self.blob_selected.emit(item.blob.id)

                # Start drag operation
                self._dragging_item = item
                self._drag_start_pos = scene_pos
                self._drag_mode = item._hover_zone or DragMode.PITCH

                if self._drag_mode == DragMode.PITCH:
                    self._drag_start_value = item.blob.shift_semitones
                elif self._drag_mode in (DragMode.STRETCH_LEFT, DragMode.STRETCH_RIGHT):
                    self._drag_start_value = item.blob.stretch_ratio

                item._is_dragging = True
                item._drag_mode = self._drag_mode
                item.update()

                self.viewport().update()

            else:
                # Click on empty space
                if self.project:
                    self.project.deselect_all()
                self._refresh_all_blobs()
                self.viewport().update()

        # Handle split tool click
        elif event.button() == Qt.MouseButton.LeftButton and self._current_tool == EditTool.SPLIT:
            scene_pos = self.mapToScene(event.pos())
            time_seconds = self.view_transform.x_to_time(scene_pos.x())
            self.split_blob_at_time(time_seconds)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for dragging."""
        scene_pos = self.mapToScene(event.pos())

        # Update pitch ruler for hovered blob
        item = self._scene.itemAt(scene_pos, self.transform())
        if isinstance(item, BlobItem):
            self._pitch_ruler.set_target(
                item.blob,
                item._bounds.x(),
                item._bounds.width()
            )
        else:
            self._pitch_ruler.set_target(None, 0, 0)

        if self._dragging_item and self._drag_mode != DragMode.NONE:
            delta = scene_pos - self._drag_start_pos

            # Check for fine-tune mode (Alt key)
            fine_tune = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            sensitivity = 0.1 if fine_tune else 1.0

            if self._drag_mode == DragMode.PITCH:
                # Vertical drag = pitch adjustment
                midi_delta = -delta.y() / self.view_transform.pixels_per_semitone
                midi_delta *= sensitivity

                new_shift = self._drag_start_value + midi_delta

                # Apply snapping unless fine-tuning
                if not fine_tune and self._snap_mode != SnapMode.OFF:
                    target = self._dragging_item.blob.quantized_midi + new_shift
                    snapped = self._snap_pitch(target)
                    new_shift = snapped - self._dragging_item.blob.quantized_midi

                self._dragging_item.blob.shift_semitones = new_shift
                self._dragging_item.update_from_blob()

                # Show feedback
                note = NoteBlob.midi_to_note_name(int(self._dragging_item.blob.shifted_midi))
                cents = int((self._dragging_item.blob.shifted_midi % 1) * 100)
                self.status_message.emit(f"Pitch: {note} ({cents:+d}¢)")

            elif self._drag_mode in (DragMode.STRETCH_LEFT, DragMode.STRETCH_RIGHT):
                # Horizontal drag on edges = time stretch
                # Calculate stretch ratio change
                original_width = (
                    self._dragging_item.blob.end_sample -
                    self._dragging_item.blob.start_sample
                ) / self._dragging_item.blob.sample_rate

                if self._drag_mode == DragMode.STRETCH_RIGHT:
                    time_delta = delta.x() / self.view_transform.pixels_per_second
                else:
                    time_delta = -delta.x() / self.view_transform.pixels_per_second

                time_delta *= sensitivity
                new_ratio = self._drag_start_value + (time_delta / original_width)
                new_ratio = max(0.25, min(4.0, new_ratio))  # Clamp to reasonable range

                self._dragging_item.blob.stretch_ratio = new_ratio
                self._dragging_item.update_from_blob()

                self.status_message.emit(f"Stretch: {new_ratio:.2f}x")

            self.viewport().update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_item:
            self._dragging_item._is_dragging = False
            self._dragging_item._drag_mode = DragMode.NONE
            self._dragging_item.update()

            # Emit modification signal
            self.blob_modified.emit(self._dragging_item.blob.id)

            self._dragging_item = None
            self._drag_mode = DragMode.NONE
            self.status_message.emit("")

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key presses."""
        key = event.key()

        if key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            # Delete selected blobs
            if self.project:
                for blob in self.project.get_selected_blobs():
                    if blob.id in self._blob_items:
                        self._scene.removeItem(self._blob_items[blob.id])
                        del self._blob_items[blob.id]
                    self.project.remove_blob(blob.id)

        elif key == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Select all
            if self.project:
                for blob in self.project.blobs:
                    blob.select()
                self._refresh_all_blobs()
                self.viewport().update()

        elif key == Qt.Key.Key_Up:
            # Nudge pitch up
            self._nudge_selected_pitch(1 if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.1)

        elif key == Qt.Key.Key_Down:
            # Nudge pitch down
            self._nudge_selected_pitch(-1 if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier else -0.1)

        elif key == Qt.Key.Key_P and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Correct pitch (Ctrl+P)
            self.correct_pitch(1.0)

        # Tool shortcuts
        elif key == Qt.Key.Key_1:
            self.set_tool(EditTool.MAIN)
        elif key == Qt.Key.Key_2:
            self.set_tool(EditTool.PITCH)
        elif key == Qt.Key.Key_3:
            self.set_tool(EditTool.TIME)
        elif key == Qt.Key.Key_4:
            self.set_tool(EditTool.SPLIT)

        # Snap mode shortcuts
        elif key == Qt.Key.Key_S and not event.modifiers():
            # Toggle snap mode
            if self._snap_mode == SnapMode.OFF:
                self.set_snap_mode(SnapMode.CHROMATIC)
            elif self._snap_mode == SnapMode.CHROMATIC:
                self.set_snap_mode(SnapMode.SCALE)
            else:
                self.set_snap_mode(SnapMode.OFF)

        super().keyPressEvent(event)

    def split_blob_at_time(self, time_seconds: float) -> None:
        """
        Split the blob at the given time position.
        Used by the Split tool.
        """
        if not self.project:
            return

        # Find blob at this time
        blobs_at_time = self.project.get_blobs_at_time(time_seconds)
        if not blobs_at_time:
            return

        from openvocal.analysis.segmenter import Segmenter

        blob = blobs_at_time[0]
        split_sample = int(time_seconds * blob.sample_rate)

        # Use segmenter to split
        segmenter = Segmenter(use_simple_pitch=True)
        segmenter._blob_id_counter = max(b.id for b in self.project.blobs) + 1

        new_blobs = segmenter.resegment_at_position(
            self.project.blobs.copy(),
            split_sample,
            self.project.audio_data,
            self.project.sample_rate,
        )

        # Update project
        # Remove old blob item
        if blob.id in self._blob_items:
            self._scene.removeItem(self._blob_items[blob.id])
            del self._blob_items[blob.id]

        self.project.blobs.clear()
        for b in new_blobs:
            self.project.add_blob(b)
            if b.id not in self._blob_items:
                self._add_blob_item(b)

        self._refresh_all_blobs()
        self.status_message.emit("Note split")

    def _nudge_selected_pitch(self, semitones: float) -> None:
        """Nudge selected blobs by semitones."""
        if not self.project:
            return

        for blob in self.project.get_selected_blobs():
            blob.shift_semitones += semitones
            if blob.id in self._blob_items:
                self._blob_items[blob.id].update_from_blob()
            self.blob_modified.emit(blob.id)

        self.viewport().update()

    def resizeEvent(self, event) -> None:
        """Handle resize to update view transform."""
        super().resizeEvent(event)
        self.view_transform.set_canvas_height(self.height())
        self._refresh_all_blobs()
        self._update_playhead()
