"""Loading and saving audio. Everything inside a GLORE project is stored as
44.1 kHz stereo float WAV so every later step (and the player) sees one format."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_SR = 44100
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"}


def load_stereo(path: str | Path, sr: int = PROJECT_SR) -> np.ndarray:
    """Return audio as float32 array shaped (2, samples) at `sr`."""
    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def load_mono(path: str | Path, sr: int = 22050) -> np.ndarray:
    """Mono copy for analysis (lower sample rate = faster)."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = PROJECT_SR, subtype: str = "FLOAT") -> None:
    """Save (channels, samples) audio. FLOAT internally; PCM_24 for exports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    sf.write(str(path), data, sr, subtype=subtype)


def duration_seconds(audio: np.ndarray, sr: int = PROJECT_SR) -> float:
    return float(audio.shape[-1]) / sr
