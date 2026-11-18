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
