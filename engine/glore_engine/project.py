"""Project state: the samples you dropped in, their analysis, and the layers
the player can play. Saved as project.json next to the audio files."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

KINDS = ("vocal", "full_song", "beat")

# Genre packs - one LoRA each. training_audio decides what's kept when a
# tagged track is prepared for training.
PACKS = {
    "house": {"label": "House", "training_audio": "instrumental", "bpm": [124, 134],
              "moods": ["soft", "hype", "heavy", "dnb"]},
    "rnb_rap": {"label": "R&B / Rap", "training_audio": "instrumental", "bpm": [130, 160],
                "moods": ["late night", "moody", "melodic", "trap", "R&B", "ambient"]},
    "gospel_soul": {"label": "Gospel / Soul", "training_audio": "keep_backing", "bpm": [70, 100],
                    "moods": ["ethereal", "gospel", "soul sample", "choir", "uplifting", "dark"]},
}
LEGACY_PACKS = {"emotive_house": "house", "melodic_rap": "rnb_rap"}


def default_root() -> Path:
    base = os.environ.get("GLORE_HOME") or os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "GLORE" / "projects"


@dataclass
class Source:
    id: str
    name: str
    kind: str
    file: str  # filename inside sources/
    original_path: str
    duration: float = 0.0
    analysis: dict | None = None
    overrides: dict = field(default_factory=dict)  # bpm / key set by the user
    derived_from: str | None = None  # e.g. vocal extracted from a full song
    pack: str | None = None          # genre pack this track trains (None = not in a training set)
    moods: list = field(default_factory=list)
    origin: str = "file"             # file | link | extracted | raw

    def effective(self) -> dict:
        """Analysis with the user's overrides applied on top."""
        a = dict(self.analysis or {})
        a.update(self.overrides)
        return a


@dataclass
class Layer:
    id: str
    name: str
    file: str  # path relative to the project folder
    role: str  # "vocal_locked", "source", "stem", ...
    source_id: str | None = None


@dataclass
class Project:
    name: str = "Untitled"
    bpm: float | None = None          # project tempo (follows vocal unless changed)
    bpm_locked_by_user: bool = False
    key: str | None = None
    genre_pack: str = "house"
    monitor_mode: str = "headphones"  # playback only - never affects export
    step: str = "import"
    sources: list[Source] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    vocal_lock: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ProjectStore:
    def __init__(self, root: Path | None = None, name: str = "current"):
        self.root = (root or default_root()) / name
        self.sources_dir = self.root / "sources"
        self.renders_dir = self.root / "renders"
        self.stems_dir = self.root / "stems"
        for d in (self.sources_dir, self.renders_dir, self.stems_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.project = self._load()

    # ---------------------------------------------------------- persistence
    @property
    def json_path(self) -> Path:
        return self.root / "project.json"

    def _load(self) -> Project:
        if not self.json_path.exists():
            return Project()
        raw = json.loads(self.json_path.read_text(encoding="utf-8"))
        src_fields = Source.__dataclass_fields__.keys()
        raw["sources"] = [Source(**{k: v for k, v in s.items() if k in src_fields}) for s in raw.get("sources", [])]
        raw["layers"] = [Layer(**l) for l in raw.get("layers", [])]
        known = Project.__dataclass_fields__.keys()
        p = Project(**{k: v for k, v in raw.items() if k in known})
        p.genre_pack = LEGACY_PACKS.get(p.genre_pack, p.genre_pack)
        return p

    def save(self) -> None:
        with self.lock:
            tmp = self.json_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.project.to_dict(), indent=2), encoding="utf-8")
            tmp.replace(self.json_path)

    def reset(self) -> None:
        with self.lock:
            for d in (self.sources_dir, self.renders_dir, self.stems_dir):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
            self.project = Project()
            self.save()

    # ---------------------------------------------------------- helpers
    def path_of(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("Path outside project")
        return p

    def source(self, source_id: str) -> Source:
        for s in self.project.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def add_source(self, src: Source) -> None:
        with self.lock:
            self.project.sources.append(src)
            self.upsert_layer(Layer(id=f"src-{src.id}", name=src.name, file=f"sources/{src.file}",
                                    role="source", source_id=src.id))
            self.save()

    def remove_source(self, source_id: str) -> None:
        with self.lock:
            s = self.source(source_id)
            self.project.sources = [x for x in self.project.sources if x.id != source_id]
            self.project.layers = [l for l in self.project.layers if l.source_id != source_id]
            try:
                (self.sources_dir / s.file).unlink(missing_ok=True)
            except OSError:
                pass
            if self.project.vocal_lock and self.project.vocal_lock.get("source_id") == source_id:
                self.project.vocal_lock = None
            self.save()

    def upsert_layer(self, layer: Layer) -> None:
        with self.lock:
            self.project.layers = [l for l in self.project.layers if l.id != layer.id]
            self.project.layers.append(layer)


def new_id() -> str:
    return uuid.uuid4().hex[:10]


# -------------------------------------------------------------- background jobs

@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | done | error
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)


class JobRunner:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._gpu_lock = threading.Lock()  # one heavy GPU job at a time

    def start(self, kind: str, fn, gpu: bool = False) -> Job:
        job = Job(id=new_id(), kind=kind)
        self.jobs[job.id] = job

        def report(p: float, msg: str = ""):
            job.progress = round(float(p), 3)
            if msg:
                job.message = msg

        def run():
            try:
                if gpu:
                    with self._gpu_lock:
                        job.result = fn(report)
                else:
                    job.result = fn(report)
                job.progress, job.status = 1.0, "done"
            except Exception as e:  # surfaced in the UI
                job.status, job.error = "error", str(e)

        threading.Thread(target=run, daemon=True).start()
        return job
