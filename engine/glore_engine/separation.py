"""Stem separation with Demucs (HTDemucs FT). Runs on the GPU when CUDA is available.

Used in session 1 to pull the acapella out of a full song.
Later steps reuse it for drums-from-beat and for splitting the final instrumental."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import load_stereo, save_wav, PROJECT_SR

_MODEL_CACHE: dict[str, object] = {}
DEFAULT_MODEL = "htdemucs_ft"  # best open-source 4-stem model (vocals, drums, bass, other)


class SeparationUnavailable(RuntimeError):
    pass


def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:  # pragma: no cover
        raise SeparationUnavailable(
            "PyTorch isn't installed. Run scripts\\setup.ps1 again to install it.") from e


def device_info() -> dict:
    try:
        torch = _torch()
    except SeparationUnavailable:
        return {"torch": False, "cuda": False, "device": "none"}
    cuda = torch.cuda.is_available()
    return {
        "torch": True,
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
    }


def _get_model(name: str):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    try:
        from demucs.pretrained import get_model
    except ImportError as e:
        raise SeparationUnavailable("Demucs isn't installed. Run scripts\\setup.ps1 again.") from e
    model = get_model(name)  # downloads weights on first use (~300MB for htdemucs_ft)
    model.eval()
    _MODEL_CACHE[name] = model
    return model


def separate(path: str | Path, out_dir: str | Path, model_name: str = DEFAULT_MODEL,
             progress: Callable[[float, str], None] | None = None) -> dict[str, Path]:
    """Split a file into stems. Returns {stem_name: wav_path}."""
    torch = _torch()
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    report = progress or (lambda p, m: None)
    report(0.05, "Loading separation model (first run downloads it)")
    model = _get_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report(0.15, "Reading audio")
    audio = load_stereo(path, PROJECT_SR)
    wav = torch.from_numpy(audio)
    wav = convert_audio(wav, PROJECT_SR, model.samplerate, model.audio_channels)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    report(0.25, f"Separating on {device.upper()}")
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, shifts=1, split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * std + mean

    report(0.9, "Saving stems")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    results: dict[str, Path] = {}
    for name, src in zip(model.sources, sources):
        data = convert_audio(src.cpu(), model.samplerate, PROJECT_SR, 2).numpy().astype(np.float32)
        p = out_dir / f"{stem}_{name}.wav"
        save_wav(p, data)
        results[name] = p
    if device == "cuda":
        torch.cuda.empty_cache()
    report(1.0, "Done")
    return results
