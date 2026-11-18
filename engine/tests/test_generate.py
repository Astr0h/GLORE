"""Generate: the ACE-Step subprocess is replaced by a tiny script speaking the same protocol."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glore_engine import generate as gen, training
from glore_engine.project import ProjectStore
from glore_engine.server import create_app

FAKE = r"""
import json, sys, numpy as np, soundfile as sf, os
job = json.load(open(sys.argv[1]))
print("GLORE_PROGRESS 0.5 Generating", flush=True)
files = []
for i in range(job["count"]):
    p = os.path.join(job["out_dir"], f"take{i}.wav")
    sf.write(p, np.zeros((44100, 2), dtype="float32"), 44100)
    files.append({"path": p, "seed": 42 + i})
print("GLORE_RESULT " + json.dumps({"files": files, "caption": (job["lora"]["trigger"] + ", " if job["lora"] else "") + job["caption"], "seconds": 3.2, "peak_vram_gb": 6.1, "job": job}), flush=True)
"""


class FakeAce:
    def __init__(self, tmp):
        self.dir = tmp
        self.env = None

    def py(self, script, job):
        return [sys.executable, "-c", FAKE, job]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "ensure_ace", lambda report, variant="turbo": FakeAce(tmp_path))
    d = training.lora_root() / "house" / "run1"
    d.mkdir(parents=True)
    (d / "glore_lora.json").write_text(json.dumps({"pack": "house", "run_id": "run1", "trained_at": 1, "variant": "turbo",
                                                    "trigger": "glore_house", "inference": {"shift": 3.0, "steps": 8}}))
    s = ProjectStore(tmp_path / "projects")
    s.project.bpm, s.project.key = 128.0, "A minor"
    return s


def test_generate_adds_takes_as_layers(store):
    res = gen.generate(store, lambda p, m: None, count=2, caption="house, hype")
    assert [t["seed"] for t in res["takes"]] == [42, 43]
    assert res["caption"] == "glore_house, house, hype" and res["lora"] == "run1"
    layers = [l for l in store.project.layers if l.role == "generated"]
    assert len(layers) == 2 and all((store.root / l.file).exists() for l in layers)
    assert not list((store.root / "generated").glob("_*"))  # temp folder cleaned up


def test_generate_without_lora_and_errors(store, tmp_path):
    res = gen.generate(store, lambda p, m: None, use_lora=False)
    assert res["lora"] is None and res["caption"] == "house"
    with pytest.raises(gen.GenerateError, match="No R&B"):
        gen.generate(store, lambda p, m: None, pack="rnb_rap")


def test_generate_endpoint_validation(tmp_path):
    c = TestClient(create_app(ProjectStore(tmp_path / "p")))
    assert c.post("/generate", json={"duration": 5}).status_code == 400
    assert c.post("/generate", json={"count": 9}).status_code == 400
    assert c.post("/generate", json={"pack": "nope"}).status_code == 404
    assert c.get("/logs/fit-test").status_code == 404


def test_around_vocal_mixes_clean_vocal_over_new_beat(store, tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from glore_engine import separation
    from glore_engine.project import Source

    sr = 44100
    t = np.arange(sr * 2) / sr
    vfile = store.sources_dir / "v1.wav"
    store.sources_dir.mkdir(parents=True, exist_ok=True)
    sf.write(vfile, np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, 1).astype("float32"), sr)
    store.add_source(Source(id="v1", name="Song (vocal)", kind="vocal", file="v1.wav", original_path="x",
                            duration=2.0, analysis={"bpm": 124.0, "key": "F minor"}))

    def fake_sep(path, out_dir, progress=None, **kw):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        stems = {}
        for name, f in (("drums", 60), ("bass", 80), ("other", 300), ("vocals", 440)):
            stems[name] = out / f"{name}.wav"
            sf.write(stems[name], np.stack([0.1 * np.sin(2 * np.pi * f * t)] * 2, 1).astype("float32"), sr)
        return stems

    monkeypatch.setattr(separation, "separate", fake_sep)
    monkeypatch.setattr(gen, "_mp3", lambda wav, mp3: (mp3.parent.mkdir(parents=True, exist_ok=True), mp3.write_bytes(b"mp3")))
    res = gen.generate_around_vocal(store, lambda p, m: None, export_dir=str(tmp_path / "out"))
    assert set(res["files"]) == {"beat", "mix"} and res["vocal"] == "Song (vocal)"
    mix, _ = sf.read(store.root / res["files"]["mix"])
    assert 0.85 < np.max(np.abs(mix)) < 0.9            # normalised to about -1 dBFS
    assert Path(res["mp3"]).name == "Song - GLORE soft house.mp3"
