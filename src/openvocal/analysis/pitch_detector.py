"""Pitch detection using torchcrepe for high-accuracy vocal analysis."""

from __future__ import annotations

from typing import Optional, Tuple, Union, overload
import numpy as np

# Lazy imports to avoid slow startup
_torch = None
_torchcrepe = None


def _ensure_imports():
    """Lazily import torch and torchcrepe."""
    global _torch, _torchcrepe
    if _torch is None:
        import torch
        _torch = torch
    if _torchcrepe is None:
        import torchcrepe
        _torchcrepe = torchcrepe


class PitchDetector:
    """
    High-accuracy pitch detection using torchcrepe CNN model.

    torchcrepe provides state-of-the-art monophonic pitch tracking,
    particularly well-suited for vocal analysis.

    Attributes:
        model: Model size ('tiny', 'small', 'medium', 'large', 'full')
        device: Torch device ('cuda', 'mps', 'cpu')
        hop_length: Hop size in samples (determines time resolution)
        fmin: Minimum frequency to detect (Hz)
        fmax: Maximum frequency to detect (Hz)
    """

    def __init__(
        self,
        model: str = "full",
        device: Optional[str] = None,
        hop_length: int = 160,  # 10ms at 16kHz (crepe native rate)
        fmin: float = 50.0,
        fmax: float = 2000.0,
    ):
        """
        Initialize the pitch detector.

        Args:
            model: Model size - larger = more accurate but slower
                   Options: 'tiny', 'small', 'medium', 'large', 'full'
            device: Compute device. None = auto-detect best available
            hop_length: Analysis hop size in samples
            fmin: Minimum detectable frequency in Hz
            fmax: Maximum detectable frequency in Hz
        """
        self.model = model
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

        # Determine device
        if device is None:
            device = self._auto_detect_device()
        self.device = device

        self._initialized = False

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available compute device."""
        _ensure_imports()
        if _torch.cuda.is_available():
            return "cuda"
        elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_initialized(self) -> None:
        """Ensure torchcrepe model is loaded."""
        if self._initialized:
            return
        _ensure_imports()
        # torchcrepe loads models lazily, but we can warm it up
        self._initialized = True

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        batch_size: int = 512,
        return_confidence: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Detect pitch (f0) from audio.

        Args:
            audio: Mono audio samples as float32 array
            sample_rate: Sample rate of the audio
            batch_size: Batch size for inference (adjust for memory)
            return_confidence: If True, also return confidence values

        Returns:
            If return_confidence is False:
                f0: Array of frequency values in Hz (0 = unvoiced)
            If return_confidence is True:
                Tuple of (f0, confidence) arrays
        """
        self._ensure_initialized()

        # Convert to torch tensor
        audio_tensor = _torch.tensor(audio, dtype=_torch.float32).unsqueeze(0)

        # torchcrepe expects 16kHz audio internally but handles resampling
        # Run prediction
        pitch, periodicity = _torchcrepe.predict(
            audio_tensor,
            sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            model=self.model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )

        # Convert to numpy
        f0 = pitch.squeeze().cpu().numpy()
        confidence = periodicity.squeeze().cpu().numpy()

        # Apply voicing threshold - zero out low confidence regions
        voicing_threshold = 0.5
        f0_voiced = np.where(confidence >= voicing_threshold, f0, 0.0)

        if return_confidence:
            return f0_voiced.astype(np.float32), confidence.astype(np.float32)
        return f0_voiced.astype(np.float32)

    def get_time_axis(
        self, audio_length: int, sample_rate: int
    ) -> np.ndarray:
        """
        Get the time axis corresponding to f0 values.

        Args:
            audio_length: Length of audio in samples
            sample_rate: Sample rate of the audio

        Returns:
            Array of time values in seconds
        """
        # torchcrepe resamples to 16kHz internally
        crepe_sr = 16000
        duration = audio_length / sample_rate
        num_frames = int(duration * crepe_sr / self.hop_length)
        return np.linspace(0, duration, num_frames)

    def samples_to_f0_index(
        self, sample_idx: int, sample_rate: int
    ) -> int:
        """Convert audio sample index to f0 array index."""
        time_seconds = sample_idx / sample_rate
        # torchcrepe uses 16kHz internally
        crepe_sr = 16000
        return int(time_seconds * crepe_sr / self.hop_length)

    def f0_index_to_samples(
        self, f0_idx: int, sample_rate: int
    ) -> int:
        """Convert f0 array index to audio sample index."""
        crepe_sr = 16000
        time_seconds = f0_idx * self.hop_length / crepe_sr
        return int(time_seconds * sample_rate)


class SimplePitchDetector:
    """
    Fallback pitch detector using autocorrelation.

    Used when torchcrepe is not available or for quick previews.
    Less accurate than torchcrepe but faster and has no dependencies.
    """

    def __init__(
        self,
        hop_length: int = 512,
        frame_length: int = 2048,
        fmin: float = 50.0,
        fmax: float = 2000.0,
    ):
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Detect pitch using autocorrelation method.

        Args:
            audio: Mono audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Array of f0 values in Hz (0 = unvoiced)
        """
        # Validate frequency bounds against Nyquist
        nyquist = sample_rate / 2
        fmax = min(self.fmax, nyquist * 0.9)  # Stay below Nyquist
        fmin = max(self.fmin, 20.0)  # Minimum audible frequency

        if fmax <= fmin:
            return np.zeros(1, dtype=np.float32)

        # Calculate lag bounds
        min_lag = max(1, int(sample_rate / fmax))
        max_lag = int(sample_rate / fmin)

        num_frames = 1 + (len(audio) - self.frame_length) // self.hop_length
        f0 = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_length]

            # Apply window
            window = np.hanning(len(frame))
            frame = frame * window

            # Compute autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # Find peak in valid lag range
            search_region = autocorr[min_lag:max_lag]
            if len(search_region) > 0 and np.max(search_region) > 0.3 * autocorr[0]:
                peak_idx = np.argmax(search_region) + min_lag
                f0[i] = sample_rate / peak_idx
            else:
                f0[i] = 0.0

        return f0
