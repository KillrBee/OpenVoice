"""Main application window for OpenVocal."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np

from PyQt6.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QProgressDialog,
)
from PyQt6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent, QCloseEvent

from openvocal.models.project import Project
from openvocal.models.note_blob import NoteBlob
from openvocal.ui.widgets.pitch_canvas import PitchCanvas
from openvocal.ui.widgets.piano_roll import PianoRoll
from openvocal.ui.widgets.transport import TransportBar
from openvocal.ui.widgets.toolbar import EditorToolbar
from openvocal.utils.audio_io import save_audio


class AnalysisWorker(QObject):
    """Performs audio analysis in a separate thread."""

    finished = pyqtSignal(list)
    progress = pyqtSignal(str, float)
    error = pyqtSignal(str)

    def __init__(self, path: Path, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.path = path
        self._is_interrupted = False

    def run(self) -> None:
        """Load audio file and run segmentation analysis."""
        from openvocal.utils.audio_io import load_audio
        from openvocal.analysis.segmenter import Segmenter

        try:
            # 1. Load audio
            self.progress.emit("load", 0)
            audio_data, sample_rate = load_audio(self.path)
            if self._is_interrupted:
                return

            # 2. Run segmentation
            segmenter = Segmenter(use_simple_pitch=True)

            def update_progress(stage: str, value: float):
                if self._is_interrupted:
                    raise InterruptedError("Analysis canceled")
                self.progress.emit(stage, value)

            blobs = segmenter.segment(audio_data, sample_rate, update_progress)
            if self._is_interrupted:
                return

            # Attach audio data to the first blob for project creation
            if blobs:
                blobs[0].__dict__["_temp_audio_data"] = audio_data
                blobs[0].__dict__["_temp_sample_rate"] = sample_rate

            self.finished.emit(blobs)

        except InterruptedError:
            self.error.emit("Analysis canceled by user.")
        except Exception as e:
            self.error.emit(f"Failed to analyze audio:\n{e}")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._is_interrupted = True


class MainWindow(QMainWindow):
    """
    Main application window containing the pitch editor.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.project = Project()
        self._audio_engine = None
        self._analysis_thread: Optional[QThread] = None
        self._analysis_worker: Optional[AnalysisWorker] = None
        self._progress_dialog: Optional[QProgressDialog] = None

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

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self.transport.play_clicked.connect(self._on_play)
        self.transport.stop_clicked.connect(self._on_stop)
        self.transport.loop_toggled.connect(self._on_loop_toggle)

        self.piano_roll.midi_range_changed.connect(self.pitch_canvas.set_midi_range)
        self.pitch_canvas.view_changed.connect(self.piano_roll.update_view)

        self.pitch_canvas.blob_selected.connect(self._on_blob_selected)
        self.pitch_canvas.blob_modified.connect(self._on_blob_modified)

        self.pitch_canvas.status_message.connect(self._on_status_message)

        self.toolbar.tool_changed.connect(self.pitch_canvas.set_tool)
        self.toolbar.snap_mode_changed.connect(self.pitch_canvas.set_snap_mode)
        self.toolbar.scale_changed.connect(self.pitch_canvas.set_scale)
        self.toolbar.correct_pitch_requested.connect(self.pitch_canvas.correct_pitch)

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

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

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self) -> None:
        """Set up global keyboard shortcuts."""
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
                self, "Audio Engine Not Available",
                "The Rust audio engine is not installed.\n"
                "Build it with: cd rust_engine && maturin develop\n\n"
                "Playback and export will not be available.",
            )
            return False
        except Exception as e:
            QMessageBox.warning(self, "Audio Engine Error", f"Failed to initialize audio engine:\n{e}")
            return False

    def load_audio(self, path: Path) -> None:
        """Load an audio file and analyze it using a background thread."""
        if self._analysis_thread and self._analysis_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Already analyzing an audio file.")
            return

        self.status_bar.showMessage(f"Loading {path.name}...")
        self._analysis_thread = QThread()
        self._analysis_worker = AnalysisWorker(path)
        self._analysis_worker.moveToThread(self._analysis_thread)

        self._progress_dialog = QProgressDialog("Analyzing audio...", "Cancel", 0, 100, self)
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setMinimumDuration(500)
        self._progress_dialog.canceled.connect(self._cancel_analysis)

        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
        self._analysis_thread.start()

    def _cancel_analysis(self) -> None:
        """Handle cancellation of the analysis task."""
        if self._analysis_worker:
            self._analysis_worker.stop()
        if self._analysis_thread and self._analysis_thread.isRunning():
            self._analysis_thread.quit()
            self._analysis_thread.wait()
        self.status_bar.showMessage("Analysis canceled")

    def _on_analysis_progress(self, stage: str, value: float) -> None:
        """Update analysis progress dialog."""
        if not self._progress_dialog:
            return
        stages = {"load": 0, "pitch": 10, "onset": 50, "segment": 80}
        base = stages.get(stage, 0)
        total_progress = int(base + value * (stages.get(stage, 100) / 100.0 * 90))
        if stage == "load":
            self._progress_dialog.setLabelText("Loading audio file...")
            self._progress_dialog.setValue(5)
        else:
            self._progress_dialog.setLabelText(f"Analyzing: {stage}...")
            self._progress_dialog.setValue(total_progress)

    def _on_analysis_error(self, message: str) -> None:
        """Handle errors from the analysis thread."""
        if self._progress_dialog:
            self._progress_dialog.close()
        QMessageBox.critical(self, "Error Analyzing File", message)
        self.status_bar.showMessage("Error analyzing file")
        self._cleanup_analysis_thread()

    def _on_analysis_finished(self, blobs: List[NoteBlob]) -> None:
        """Handle successful completion of the analysis."""
        if self._progress_dialog:
            self._progress_dialog.setValue(100)
            self._progress_dialog.close()

        if not blobs:
            self.status_bar.showMessage("No notes detected in audio file.")
            return

        audio_data = getattr(blobs[0], "_temp_audio_data", None)
        sample_rate = getattr(blobs[0], "_temp_sample_rate", None)
        path = self._analysis_worker.path

        if audio_data is None or sample_rate is None:
            self.status_bar.showMessage("Internal error: audio data not passed from worker.")
            return

        self.project.audio_path = path
        self.project.audio_data = audio_data
        self.project.sample_rate = sample_rate
        self.project.name = path.stem
        self.project.blobs.clear()
        for blob in blobs:
            self.project.add_blob(blob)

        if self._init_audio_engine():
            self._audio_engine.load_audio(audio_data, sample_rate)

        self.pitch_canvas.set_project(self.project)
        self.transport.set_duration(self.project.duration)
        self.setWindowTitle(f"OpenVocal - {self.project.name}")
        self.status_bar.showMessage(f"Loaded {path.name} ({len(blobs)} notes detected)")

        if self._audio_engine:
            try:
                from openvocal_engine import BlobParams
                engine_blobs = [
                    BlobParams(b.id, b.start_sample, b.end_sample, b.shift_semitones, b.stretch_ratio)
                    for b in self.project.blobs
                ]
                self._audio_engine.set_blobs(engine_blobs)
            except ImportError:
                pass

        self._cleanup_analysis_thread()

    def _cleanup_analysis_thread(self) -> None:
        """Clean up analysis thread and worker."""
        self._analysis_thread = None
        self._analysis_worker = None
        self._progress_dialog = None

    def _on_open(self) -> None:
        """Handle File > Open."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self.load_audio(Path(path))

    def _on_export(self) -> None:
        """Handle File > Export."""
        if not self.project.has_audio:
            QMessageBox.information(self, "Export", "No audio loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audio", f"{self.project.name}_edited.wav", "WAV Files (*.wav);;FLAC Files (*.flac)",
        )
        if not path:
            return

        # Show progress dialog
        progress = QProgressDialog("Exporting audio...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        self.status_bar.showMessage("Exporting audio...")

        try:
            processed_audio = self._process_audio_for_export(progress)
            if processed_audio is None:
                # Export was cancelled or failed
                return

            progress.setLabelText("Writing file...")
            QApplication.processEvents()

            save_audio(Path(path), processed_audio, self.project.sample_rate)

            progress.close()
            self.status_bar.showMessage(f"Exported successfully to {Path(path).name}")
            QMessageBox.information(
                self, "Export Complete",
                f"Audio exported successfully to:\n{Path(path).name}"
            )
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Export Error", f"Failed to export audio:\n{e}")
            self.status_bar.showMessage("Export failed")

    def _process_audio_for_export(self, progress: QProgressDialog) -> Optional[np.ndarray]:
        """
        Process audio with pitch/time modifications for export.

        Uses Rust engine if available, otherwise falls back to pure Python.
        """
        # Try Rust engine first (faster, better quality)
        if self._audio_engine:
            try:
                from openvocal_engine import BlobParams
                engine_blobs = [
                    BlobParams(b.id, b.start_sample, b.end_sample, b.shift_semitones, b.stretch_ratio)
                    for b in self.project.blobs
                ]
                progress.setLabelText("Processing audio (Rust engine)...")
                QApplication.processEvents()
                return self._audio_engine.process_offline(engine_blobs)
            except Exception as e:
                # Fall through to Python fallback
                print(f"Rust engine export failed, using Python fallback: {e}")

        # Pure Python fallback
        progress.setLabelText("Processing audio (Python fallback)...")
        QApplication.processEvents()
        return self._process_audio_python_fallback(progress)

    def _process_audio_python_fallback(self, progress: QProgressDialog) -> Optional[np.ndarray]:
        """
        Pure Python audio processing fallback when Rust engine unavailable.

        This is slower and lower quality than the Rust engine but allows
        basic export functionality without the native extension.
        """
        audio = self.project.audio_data.copy()
        sample_rate = self.project.sample_rate
        blobs = self.project.blobs

        if not blobs:
            return audio

        # Simple pitch shifting using resampling (no formant preservation)
        # This is a basic implementation - the Rust engine is preferred
        progress.setMaximum(len(blobs))

        for i, blob in enumerate(blobs):
            if progress.wasCanceled():
                self.status_bar.showMessage("Export cancelled")
                return None

            progress.setValue(i)
            progress.setLabelText(f"Processing note {i + 1} of {len(blobs)}...")
            QApplication.processEvents()

            # Skip if no modifications
            if abs(blob.shift_semitones) < 0.01 and abs(blob.stretch_ratio - 1.0) < 0.01:
                continue

            start = blob.start_sample
            end = blob.end_sample
            if start >= end or end > len(audio):
                continue

            chunk = audio[start:end].copy()

            # Apply pitch shift via resampling (simple but causes duration change)
            if abs(blob.shift_semitones) >= 0.01:
                pitch_ratio = 2.0 ** (blob.shift_semitones / 12.0)
                chunk = self._resample_chunk(chunk, pitch_ratio, sample_rate)

            # Apply time stretch via resampling
            if abs(blob.stretch_ratio - 1.0) >= 0.01:
                chunk = self._resample_chunk(chunk, 1.0 / blob.stretch_ratio, sample_rate)

            # Simple overlap-add back into output
            new_len = len(chunk)
            new_end = start + new_len

            # Resize output if needed
            if new_end > len(audio):
                audio = np.pad(audio, (0, new_end - len(audio)), mode='constant')

            # Crossfade to reduce clicks (10ms fade)
            fade_samples = int(sample_rate * 0.01)
            self._crossfade_insert(audio, chunk, start, fade_samples)

        progress.setValue(len(blobs))
        return audio

    def _resample_chunk(self, chunk: np.ndarray, ratio: float, sample_rate: int) -> np.ndarray:
        """Resample an audio chunk by the given ratio."""
        if abs(ratio - 1.0) < 0.001:
            return chunk

        new_length = int(len(chunk) / ratio)
        if new_length < 1:
            return chunk

        # Use linear interpolation for simplicity
        indices = np.linspace(0, len(chunk) - 1, new_length)
        return np.interp(indices, np.arange(len(chunk)), chunk).astype(np.float32)

    def _crossfade_insert(
        self, output: np.ndarray, chunk: np.ndarray, start: int, fade_len: int
    ) -> None:
        """Insert a chunk into output with crossfade to reduce clicks."""
        end = start + len(chunk)
        fade_len = min(fade_len, len(chunk) // 2, start, len(output) - end if end < len(output) else 0)

        if fade_len > 0:
            # Fade in
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            chunk[:fade_len] *= fade_in
            output[start:start + fade_len] *= (1 - fade_in)

            # Fade out
            fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
            if end <= len(output):
                chunk[-fade_len:] *= fade_out
                output[end - fade_len:end] *= (1 - fade_out)

        # Copy chunk to output
        output[start:end] = output[start:end] + chunk[:end - start]

    def _on_undo(self) -> None:
        """Handle Edit > Undo."""
        pass  # TODO: Implement undo stack

    def _on_redo(self) -> None:
        """Handle Edit > Redo."""
        pass  # TODO: Implement redo

    def _on_select_all(self) -> None:
        """Handle Edit > Select All."""
        for blob in self.project.blobs:
            blob.select()
        self.pitch_canvas.update()

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self, "About OpenVocal",
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
                self._audio_engine.set_loop(self.project.loop_start, self.project.loop_end)
            else:
                self._audio_engine.set_loop(None, None)

    def _on_blob_selected(self, blob_id: int) -> None:
        """Handle blob selection."""
        self.project.select_blob(blob_id)
        blob = self.project.get_blob_by_id(blob_id)
        if blob:
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
                pass

    def _update_playback_position(self) -> None:
        """Update UI with current playback position."""
        if self._audio_engine:
            pos = self._audio_engine.get_position()
            time = pos / self.project.sample_rate if self.project.sample_rate > 0 else 0
            self.transport.set_position(time)
            self.pitch_canvas.set_playhead_position(time)
            if not self._audio_engine.is_playing():
                self._playback_timer.stop()
                self.transport.set_playing(False)

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
        if self._analysis_thread and self._analysis_thread.isRunning():
            self._cancel_analysis()

        if self.project.is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        if self._audio_engine:
            self._audio_engine.stop()
        event.accept()
