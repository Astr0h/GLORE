"""Raw vocal: strip the production off an extracted vocal, one GPU model per stage.

Order matters - each stage feeds the next:
  backing  -> keep the lead, drop doubles / harmonies / ad-lib layers
  echo     -> remove delays and echo throws
  reverb   -> remove room / plate / hall reverb
  noise    -> remove hiss and leftover separation bleed

Every stage is optional. Autotune can't be undone (it changes the notes themselves).
The same lead/backing model is reused to build Gospel/Soul training sets
("keep backing vocals")."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# (model file, stem name to KEEP). Stem names are what the model labels its outputs.
STAGES: dict[str, tuple[str, str]] = {
    "backing": ("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", "vocals"),
    "echo": ("UVR-De-Echo-Normal.pth", "no echo"),
    "reverb": ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "noreverb"),
    "noise": ("denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", "dry"),
}
STAGE_ORDER = ["backing", "echo", "reverb", "noise"]
STAGE_LABELS = {"backing": "Remove backing vocals", "echo": "Remove echo / delay",
                "reverb": "Remove reverb", "noise": "Remove noise"}


class CleanupUnavailable(RuntimeError):
    pass


def models_dir() -> Path:
    import os
    base = os.environ.get("GLORE_HOME") or os.environ.get("APPDATA") or str(Path.home())
    d = Path(base) / "GLORE" / "models" / "separator"
    d.mkdir(parents=True, exist_ok=True)
    return d


_separator = None


def _get_separator(out_dir: Path):
    global _separator
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise CleanupUnavailable("audio-separator isn't installed. Run scripts\\setup.ps1 again.") from e
    if _separator is None:
        _separator = Separator(output_dir=str(out_dir), model_file_dir=str(models_dir()),
                               output_format="WAV", log_level=logging.WARNING)
    _separator.output_dir = str(out_dir)
    if getattr(_separator, "model_instance", None) is not None:
        _separator.model_instance.output_dir = str(out_dir)
    return _separator


def _pick(outputs: list[str], out_dir: Path, stem: str) -> Path:
    """Model outputs are named like 'name_(No Echo)_model.wav' - find the one to keep."""
    want = f"({stem})".lower()
    for o in outputs:
        p = Path(o) if Path(o).is_absolute() else out_dir / o
        if want in p.name.lower():
            return p
    raise RuntimeError(f"Expected a '{stem}' output, got {outputs}")


def run_stage(stage: str, in_path: Path, work_dir: Path, keep: str | None = None) -> Path:
    """Run one model; returns the kept stem. `keep` overrides which stem to keep
    (e.g. keep the backing vocals instead of the lead)."""
    model, default_keep = STAGES[stage]
    sep = _get_separator(work_dir)
    if getattr(sep, "_glore_loaded", None) != model:
        sep.load_model(model_filename=model)  # downloads on first use
        sep._glore_loaded = model
    outputs = sep.separate(str(in_path))
    return _pick(outputs, work_dir, keep or default_keep)


@dataclass
class CleanupResult:
    path: Path
    stages: list[str]


def raw_vocal(in_path: str | Path, out_path: str | Path, stages: list[str],
              progress: Callable[[float, str], None] | None = None) -> CleanupResult:
    stages = [s for s in STAGE_ORDER if s in stages]
    if not stages:
        raise ValueError("Pick at least one clean-up step")
    report = progress or (lambda p, m: None)
    current = Path(in_path)
    with tempfile.TemporaryDirectory(prefix="glore_raw_") as tmp:
        tmp = Path(tmp)
        for i, stage in enumerate(stages):
            report(0.05 + 0.9 * i / len(stages), f"{STAGE_LABELS[stage]} ({i + 1}/{len(stages)})")
            stage_dir = tmp / stage
            stage_dir.mkdir()
            current = run_stage(stage, current, stage_dir)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, out_path)
    report(1.0, "Done")
    return CleanupResult(path=out_path, stages=stages)
