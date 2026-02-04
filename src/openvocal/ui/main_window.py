"""Main application window for OpenVocal."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QProgressDialog,
    QApplication,
)
from PyQt6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent, QCloseEvent

from openvocal.models.project import Project
from openvocal.ui.widgets.pitch_canvas import PitchCanvas
from openvocal.ui.widgets.piano_roll import PianoRoll
from openvocal.ui.widgets.transport import TransportBar
from openvocal.ui.widgets.toolbar import EditorToolbar


class MainWindow(QMainWindow):
    """
    Main application window containing the pitch editor.

    Layout:
    ┌─────────────────────────────────────────────┐
    │  Menu Bar                                    │
    ├─────────────────────────────────────────────┤
    │  Toolbar                                     │
    ├─────┬───────────────────────────────────────┤
    │     │                                        │
    │Piano│        Pitch Canvas                    │
    │Roll │        (Blob Editor)                   │
    │     │                                        │
    ├─────┴───────────────────────────────────────┤
    │  Transport Bar                               │
    ├─────────────────────────────────────────────┤
    │  Status Bar                                  │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.project = Project()
        self._audio_engine = None
        self._is_analyzing = False

        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._setup_timer()

        self.setWindowTitle("OpenVocal")
        self.resize(1280, 720)

    def _setup_ui(self) -> None:
        """Initialize the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        self.toolbar = EditorToolbar(self)
        main_layout.addWidget(self.toolbar)

        # Editor area (piano roll + canvas)
        editor_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.piano_roll = PianoRoll(self)
        self.piano_roll.setFixedWidth(80)
        editor_splitter.addWidget(self.piano_roll)

        self.pitch_canvas = PitchCanvas(self)
        editor_splitter.addWidget(self.pitch_canvas)

        # Don't allow piano roll to be collapsed
        editor_splitter.setCollapsible(0, False)
        editor_splitter.setStretchFactor(0, 0)
        editor_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(editor_splitter, 1)

        # Transport bar
        self.transport = TransportBar(self)
        main_layout.addWidget(self.transport)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Connect signals
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        # Transport controls
        self.transport.play_clicked.connect(self._on_play)
        self.transport.stop_clicked.connect(self._on_stop)
        self.transport.loop_toggled.connect(self._on_loop_toggle)

        # Piano roll and canvas coordination
        self.piano_roll.midi_range_changed.connect(self.pitch_canvas.set_midi_range)
        self.pitch_canvas.view_changed.connect(self.piano_roll.update_view)

        # Blob selection and modification
        self.pitch_canvas.blob_selected.connect(self._on_blob_selected)
        self.pitch_canvas.blob_modified.connect(self._on_blob_modified)

        # Canvas status messages
        self.pitch_canvas.status_message.connect(self._on_status_message)

        # Toolbar actions
        self.toolbar.tool_changed.connect(self.pitch_canvas.set_tool)
        self.toolbar.snap_mode_changed.connect(self.pitch_canvas.set_snap_mode)
        self.toolbar.scale_changed.connect(self.pitch_canvas.set_scale)
        self.toolbar.correct_pitch_requested.connect(self.pitch_canvas.correct_pitch)

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("&Export...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction("Select &All", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self._on_select_all)
        edit_menu.addAction(select_all_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.pitch_canvas.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.pitch_canvas.zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_fit_action = QAction("Zoom to &Fit", self)
        zoom_fit_action.setShortcut(QKeySequence("Ctrl+0"))
        zoom_fit_action.triggered.connect(self.pitch_canvas.zoom_fit)
        view_menu.addAction(zoom_fit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self) -> None:
        """Set up global keyboard shortcuts."""
        # Spacebar for play/stop toggle
        from PyQt6.QtGui import QShortcut

        play_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        play_shortcut.activated.connect(self._toggle_playback)

    def _setup_timer(self) -> None:
        """Set up the playback position update timer."""
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(16)  # ~60 FPS
        self._playback_timer.timeout.connect(self._update_playback_position)

    def _init_audio_engine(self) -> bool:
        """Initialize the Rust audio engine."""
        if self._audio_engine is not None:
            return True

        try:
            from openvocal_engine import AudioEngine
            self._audio_engine = AudioEngine()
            self._audio_engine.initialize()
            return True
        except ImportError:
            QMessageBox.warning(
                self,
                "Audio Engine Not Available",
                "The Rust audio engine is not installed.\n"
                "Build it with: cd rust_engine && maturin develop\n\n"
                "Playback will not be available.",
            )
            return False
        except Exception as e:
            QMessageBox.warning(
                self,
                "Audio Engine Error",
                f"Failed to initialize audio engine:\n{e}",
            )
            return False

    def load_audio(self, path: Path) -> None:
        """Load an audio file and analyze it."""
        from openvocal.utils.audio_io import load_audio
        from openvocal.analysis.segmenter import Segmenter

        self._is_analyzing = True
        self.status_bar.showMessage(f"Loading {path.name}...")

        try:
            # Load audio
            audio_data, sample_rate = load_audio(path)

            # Update project
            self.project.audio_path = path
            self.project.audio_data = audio_data
            self.project.sample_rate = sample_rate
            self.project.name = path.stem
            self.project.blobs.clear()

            # Initialize audio engine with data
            if self._init_audio_engine():
                import numpy as np
                self._audio_engine.load_audio(audio_data, sample_rate)

            # Create progress dialog for analysis
            progress = QProgressDialog("Analyzing audio...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500)

            def update_progress(stage: str, value: float):
                if progress.wasCanceled():
                    raise InterruptedError("Analysis canceled")
                stages = {"pitch": 0, "onset": 33, "segment": 66}
                base = stages.get(stage, 0)
                progress.setValue(int(base + value * 33))
                progress.setLabelText(f"Analyzing: {stage}...")
                QApplication.processEvents()

            # Run segmentation
            segmenter = Segmenter(use_simple_pitch=True)  # Use simple for faster MVP
            blobs = segmenter.segment(audio_data, sample_rate, update_progress)

            # Add blobs to project
            for blob in blobs:
                self.project.add_blob(blob)

            progress.close()

            # Update UI
            self.pitch_canvas.set_project(self.project)
            self.transport.set_duration(self.project.duration)
            self.setWindowTitle(f"OpenVocal - {self.project.name}")
            self.status_bar.showMessage(
                f"Loaded {path.name} ({len(blobs)} notes detected)"
            )

            # Send blob params to engine
            if self._audio_engine:
                try:
                    from openvocal_engine import BlobParams
                    engine_blobs = [
                        BlobParams(
                            b.id, b.start_sample, b.end_sample,
                            b.shift_semitones, b.stretch_ratio
                        )
                        for b in self.project.blobs
                    ]
                    self._audio_engine.set_blobs(engine_blobs)
                except ImportError:
                    pass  # Engine not available

        except InterruptedError:
            self.status_bar.showMessage("Analysis canceled")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading File",
                f"Failed to load audio file:\n{e}",
            )
            self.status_bar.showMessage("Error loading file")
        finally:
            self._is_analyzing = False

    # === Slots ===

    def _on_open(self) -> None:
        """Handle File > Open."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self.load_audio(Path(path))

    def _on_export(self) -> None:
        """Handle File > Export."""
        if not self.project.has_audio:
            QMessageBox.information(self, "Export", "No audio loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Audio",
            f"{self.project.name}_edited.wav",
            "WAV Files (*.wav);;FLAC Files (*.flac)",
        )
        if path:
            # TODO: Implement export with applied pitch/time changes
            self.status_bar.showMessage(f"Exported to {path}")

    def _on_undo(self) -> None:
        """Handle Edit > Undo."""
        # TODO: Implement undo stack
        pass

    def _on_redo(self) -> None:
        """Handle Edit > Redo."""
        # TODO: Implement redo
        pass

    def _on_select_all(self) -> None:
        """Handle Edit > Select All."""
        for blob in self.project.blobs:
            blob.select()
        self.pitch_canvas.update()

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About OpenVocal",
            "<h2>OpenVocal</h2>"
            "<p>Version 0.1.0</p>"
            "<p>Open-source vocal pitch correction with visual blob editing.</p>"
            "<p>A minimalist alternative to Melodyne for monophonic vocal editing.</p>",
        )

    def _on_play(self) -> None:
        """Start playback."""
        if self._audio_engine and self.project.has_audio:
            self._audio_engine.play()
            self._playback_timer.start()

    def _on_stop(self) -> None:
        """Stop playback."""
        if self._audio_engine:
            self._audio_engine.stop()
            self._playback_timer.stop()

    def _toggle_playback(self) -> None:
        """Toggle play/stop."""
        if self._audio_engine and self._audio_engine.is_playing():
            self._on_stop()
            self.transport.set_playing(False)
        else:
            self._on_play()
            self.transport.set_playing(True)

    def _on_loop_toggle(self, enabled: bool) -> None:
        """Handle loop toggle."""
        if self._audio_engine:
            if enabled and self.project.loop_enabled:
                self._audio_engine.set_loop(
                    self.project.loop_start, self.project.loop_end
                )
            else:
                self._audio_engine.set_loop(None, None)

    def _on_blob_selected(self, blob_id: int) -> None:
        """Handle blob selection."""
        self.project.select_blob(blob_id)
        # Update toolbar info with selected note
        blob = self.project.get_blob_by_id(blob_id)
        if blob:
            from openvocal.models.note_blob import NoteBlob
            note_name = NoteBlob.midi_to_note_name(int(blob.shifted_midi))
            self.toolbar.set_info(f"Selected: {note_name}")

    def _on_status_message(self, message: str) -> None:
        """Handle status message from canvas."""
        if message:
            self.status_bar.showMessage(message)
            self.toolbar.set_info(message)
        else:
            self.status_bar.showMessage("Ready")
            self.toolbar.set_info("")

    def _on_blob_modified(self, blob_id: int) -> None:
        """Handle blob modification (pitch/time change)."""
        blob = self.project.get_blob_by_id(blob_id)
        if blob and self._audio_engine:
            try:
                from openvocal_engine import BlobParams
                params = BlobParams(
                    blob.id, blob.start_sample, blob.end_sample,
                    blob.shift_semitones, blob.stretch_ratio
                )
                self._audio_engine.update_blob(params)
            except ImportError:
                pass  # Engine not available

    def _update_playback_position(self) -> None:
        """Update UI with current playback position."""
        if self._audio_engine:
            pos = self._audio_engine.get_position()
            time = pos / self.project.sample_rate if self.project.sample_rate > 0 else 0
            self.transport.set_position(time)
            self.pitch_canvas.set_playhead_position(time)

            # Check if playback ended
            if not self._audio_engine.is_playing():
                self._playback_timer.stop()
                self.transport.set_playing(False)

    # === Drag and Drop ===

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept audio file drops."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    suffix = Path(url.toLocalFile()).suffix.lower()
                    if suffix in {".wav", ".flac", ".mp3", ".ogg", ".aiff"}:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped audio files."""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".aiff"}:
                    self.load_audio(path)
                    break

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close."""
        if self.project.is_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        # Stop playback and clean up
        if self._audio_engine:
            self._audio_engine.stop()
        event.accept()
