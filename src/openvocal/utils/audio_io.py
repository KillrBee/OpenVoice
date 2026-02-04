"""Audio file I/O utilities."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_audio(
    path: Path | str,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return samples and sample rate.

    Args:
        path: Path to audio file (WAV, FLAC, etc.)
        target_sr: Target sample rate for resampling (None = keep original)
        mono: If True, convert to mono by averaging channels

    Returns:
        Tuple of (audio_data as float32 array, sample_rate)

    Raises:
        ImportError: If soundfile is not installed
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio file cannot be read
    """
    if sf is None:
        raise ImportError(
            "soundfile is required for audio I/O. "
            "Install with: pip install soundfile"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio_data, sample_rate = sf.read(path, dtype='float32')
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file: {e}") from e

    # Convert to mono if requested and stereo
    if mono and audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if requested
    if target_sr is not None and target_sr != sample_rate:
        audio_data = _resample(audio_data, sample_rate, target_sr)
        sample_rate = target_sr

    return audio_data.astype(np.float32), sample_rate


def save_audio(
    path: Path | str,
    audio_data: np.ndarray,
    sample_rate: int,
) -> None:
    """
    Save audio data to a file.

    Args:
        path: Output path (format inferred from extension)
        audio_data: Audio samples as float array
        sample_rate: Sample rate in Hz

    Raises:
        ImportError: If soundfile is not installed
        RuntimeError: If audio file cannot be written
    """
    if sf is None:
        raise ImportError(
            "soundfile is required for audio I/O. "
            "Install with: pip install soundfile"
        )

    path = Path(path)

    try:
        sf.write(path, audio_data, sample_rate)
    except Exception as e:
        raise RuntimeError(f"Failed to write audio file: {e}") from e


def _resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Uses resampy for high-quality resampling if available,
    falls back to linear interpolation otherwise.
    """
    try:
        import resampy
        return resampy.resample(audio, orig_sr, target_sr)
    except ImportError:
        # Fallback to simple linear interpolation
        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def compute_rms_envelope(
    audio: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute RMS amplitude envelope for visualization.

    Args:
        audio: Audio samples
        frame_length: Analysis frame size in samples
        hop_length: Hop size between frames

    Returns:
        RMS envelope array
    """
    # Pad audio to ensure we get frames for the entire signal
    pad_length = frame_length // 2
    padded = np.pad(audio, (pad_length, pad_length), mode='reflect')

    num_frames = 1 + (len(padded) - frame_length) // hop_length
    rms = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        frame = padded[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    return rms
