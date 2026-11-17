"""LoRA training sets - one per genre pack, shared across all projects.

Tagging a track copies its full mix in. "Prepare" then strips it for training:
  instrumental  -> all vocals removed (House, R&B / Rap)
  keep_backing  -> lead vocal removed, choirs / hums / backing kept (Gospel / Soul)
Captions (pack + moods + BPM + key) are written next to each file for training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import load_stereo, save_wav, PROJECT_SR
from .project import PACKS

MIN_TRACKS = 30  # rough minimum before a LoRA is worth training


def root() -> Path:
    base = os.environ.get("GLORE_HOME") or os.environ.get("APPDATA") or str(Path.home())
    d = Path(base) / "GLORE" / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pack_dir(pack: str) -> Path:
    if pack not in PACKS:
        raise ValueError(f"Unknown pack {pack}")
    d = root() / pack
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def caption(pack: str, moods: list[str], bpm=None, key=None) -> str:
    parts = [PACKS[pack]["label"].lower()] + [m.lower() for m in moods]
    if bpm:
        parts.append(f"{round(float(bpm))} bpm")
    if key:
        parts.append(str(key))
    return ", ".join(parts)


def _meta_path(pack: str, item_id: str) -> Path:
    return pack_dir(pack) / f"{item_id}.json"


def list_items(pack: str) -> list[dict]:
    items = []
    for p in sorted(pack_dir(pack).glob("*.json")):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(items, key=lambda m: m.get("added", 0))


def summary() -> list[dict]:
    out = []
    for key, p in PACKS.items():
        items = list_items(key)
        used = sorted({m for it in items for m in it.get("moods", [])} | set(p["moods"]), key=str.lower)
        out.append({"id": key, "label": p["label"], "training_audio": p["training_audio"],
                    "bpm": p["bpm"], "count": len(items),
                    "prepared": sum(1 for it in items if it.get("prepared")),
                    "min_tracks": MIN_TRACKS, "moods": used})
    return out


def add_or_update(pack: str, audio_path: Path, name: str, moods: list[str], bpm=None, key=None,
                  source_ref: str | None = None) -> dict:
    item_id = file_hash(audio_path)
    meta_p = _meta_path(pack, item_id)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {
        "id": item_id, "added": time.time(), "prepared": False}
    if not (pack_dir(pack) / f"{item_id}.wav").exists():
        shutil.copyfile(audio_path, pack_dir(pack) / f"{item_id}.wav")
    meta.update(name=name, moods=moods, bpm=bpm, key=key, pack=pack, source_ref=source_ref,
                caption=caption(pack, moods, bpm, key))
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def remove(pack: str, item_id: str) -> None:
    d = pack_dir(pack)
    for suffix in (".json", ".wav", "_train.wav", "_vocals.wav"):
        (d / f"{item_id}{suffix}").unlink(missing_ok=True)


def remove_by_source(source_ref: str) -> None:
    for pack in PACKS:
        for it in list_items(pack):
            if it.get("source_ref") == source_ref:
                remove(pack, it["id"])


def find_by_source(source_ref: str) -> dict | None:
    for pack in PACKS:
        for it in list_items(pack):
            if it.get("source_ref") == source_ref:
                return it
    return None


# ---------------------------------------------------------------- prepare for training

def prepare(pack: str, progress: Callable[[float, str], None] | None = None,
            separate_fn=None, backing_fn=None) -> dict:
    """Strip vocals from every unprepared item. separate_fn / backing_fn are
    injectable for tests; defaults use Demucs and the karaoke model."""
    from . import separation
    report = progress or (lambda p, m: None)
    separate_fn = separate_fn or separation.separate
    mode = PACKS[pack]["training_audio"]
    todo = [it for it in list_items(pack) if not it.get("prepared")]
    if not todo:
        return {"prepared": 0}
    d = pack_dir(pack)
    for i, it in enumerate(todo):
        base = 0.02 + 0.96 * i / len(todo)
        report(base, f"Preparing {i + 1}/{len(todo)}: {it.get('name', it['id'])}")
        with tempfile.TemporaryDirectory(prefix="glore_prep_") as tmp:
            stems = separate_fn(d / f"{it['id']}.wav", tmp)
            if "vocals" in stems:  # kept so each pack can learn its chop style later
                shutil.copyfile(stems["vocals"], d / f"{it['id']}_vocals.wav")
            mix = None
            for name, path in stems.items():
                if name == "vocals":
                    continue
                a = load_stereo(path, PROJECT_SR)
                mix = a if mix is None else mix[:, :a.shape[1]] + a[:, :mix.shape[1]]
            if mode == "keep_backing" and "vocals" in stems:
                if backing_fn is None:
                    from .cleanup import run_stage
                    stage_dir = Path(tmp) / "backing"
                    stage_dir.mkdir()
                    backing_path = run_stage("backing", Path(stems["vocals"]), stage_dir, keep="instrumental")
                else:
                    backing_path = backing_fn(stems["vocals"], tmp)
                b = load_stereo(backing_path, PROJECT_SR)
                n = min(mix.shape[1], b.shape[1])
                mix = mix[:, :n] + b[:, :n]
            peak = float(np.max(np.abs(mix))) or 1.0
            if peak > 0.99:
                mix = mix / peak * 0.99
            save_wav(d / f"{it['id']}_train.wav", mix.astype(np.float32))
        it["prepared"] = True
        it["training_audio"] = mode
        _meta_path(pack, it["id"]).write_text(json.dumps(it, indent=2), encoding="utf-8")
    report(1.0, "Done")
    return {"prepared": len(todo)}
