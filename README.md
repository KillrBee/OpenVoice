# OpenVocal

Open-source vocal pitch correction with visual blob editing — a minimalist alternative to Melodyne for monophonic vocal editing.

## Features

- **Visual Blob Editing**: Intuitive graphical pitch correction interface
- **High-Performance**: Hybrid Python/Rust architecture for responsive UI and low-latency audio
- **Neural Pitch Detection**: Accurate pitch tracking using torchcrepe (or fast autocorrelation fallback)
- **Real-Time Processing**: Pitch shifting and time stretching during playback

## Architecture

OpenVocal uses a three-layer architecture:

1. **Python Layer (Brain)**: Analysis, UI, and project management using PyQt6
2. **Bridge Layer (PyO3)**: Data translation between Python and Rust
3. **Rust Layer (Muscle)**: Real-time audio DSP with cpal

## Installation

### Prerequisites

- Python 3.10+
- Rust toolchain (for building the audio engine)
- PyQt6

### Quick Start

```bash
# Clone the repository
git clone https://github.com/openvocal/openvocal.git
cd openvocal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -e .

# Build the Rust audio engine
cd rust_engine
pip install maturin
maturin develop --release
cd ..

# Run the application
python -m openvocal
```

### Using Without Rust Engine

The application can run without the Rust audio engine (playback will be disabled):

```bash
pip install -e .
python -m openvocal
```

## Usage

### Loading Audio

- Drag and drop a WAV or FLAC file onto the window
- Or use File → Open (Ctrl+O)

### Editing Notes

- **Select**: Click on a blob to select it
- **Move Pitch**: Drag a blob vertically (snaps to semitones)
- **Fine Tune**: Hold Alt while dragging for cent-level precision
- **Time Stretch**: Drag the edges of a blob to stretch/compress
- **Piano Roll**: Click keys on the left sidebar to audition pitches

### Navigation

- **Vertical Scroll**: Mouse wheel
- **Horizontal Scroll**: Shift + Mouse wheel
- **Zoom Time (horizontal)**: Ctrl + Mouse wheel
- **Zoom Pitch (vertical)**: Alt + Mouse wheel (or Ctrl+Shift + Mouse wheel)
- **Zoom In/Out**: + / - keys (horizontal), Shift + +/- (vertical)
- **Fit All Content**: Home or F key
- **Fit Selection**: Ctrl + F

### Playback

- **Play/Pause**: Spacebar
- **Stop**: Escape or click Stop button
- **Loop**: Toggle the Loop button

## Project Structure

```
openvocal/
├── src/openvocal/
│   ├── analysis/          # Pitch and onset detection
│   │   ├── pitch_detector.py
│   │   ├── onset_detector.py
│   │   └── segmenter.py
│   ├── models/            # Data models
│   │   ├── note_blob.py
│   │   └── project.py
│   ├── ui/                # PyQt6 user interface
│   │   ├── main_window.py
│   │   └── widgets/
│   │       ├── pitch_canvas.py
│   │       ├── piano_roll.py
│   │       ├── transport.py
│   │       └── toolbar.py
│   └── utils/             # Utilities
│       ├── audio_io.py
│       └── coordinates.py
├── rust_engine/           # Rust audio engine
│   └── src/
│       ├── lib.rs
│       ├── audio_engine.rs
│       ├── dsp.rs
│       └── types.rs
└── tests/
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Building Rust Engine for Development

```bash
cd rust_engine
maturin develop  # Debug build
maturin develop --release  # Release build
```

## Roadmap

- [x] Core data models (NoteBlob, Project)
- [x] PyQt6 UI framework
- [x] Blob visualization
- [x] Pitch detection (torchcrepe/autocorrelation)
- [x] Onset detection (librosa)
- [x] Rust audio engine with PyO3
- [x] Real-time pitch shifting
- [x] Export edited audio (WAV/FLAC)
- [ ] Formant preservation
- [ ] Polyphonic support
- [ ] Plugin format (VST/AU)

## Known Limitations

- **Monophonic only**: Designed for single voice/instrument
- **Chipmunk effect**: High pitch shifts may affect timbre (formant shifting planned)
- **No MIDI export**: Pitch data cannot be exported as MIDI yet

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
