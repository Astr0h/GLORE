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
        for i, (name, f) in enumerate((("drums", 60), ("bass", 80), ("other", 300), ("vocals", 440))):
            stems[name] = out / f"{name}.wav"
            tt = t[: len(t) - 57 * i]  # real stems come back a few samples apart
            sf.write(stems[name], np.stack([0.1 * np.sin(2 * np.pi * f * tt)] * 2, 1).astype("float32"), sr)
        return stems

    monkeypatch.setattr(separation, "separate", fake_sep)
    monkeypatch.setattr(gen, "_mp3", lambda wav, mp3: (mp3.parent.mkdir(parents=True, exist_ok=True), mp3.write_bytes(b"mp3")))
    res = gen.generate_around_vocal(store, lambda p, m: None, export_dir=str(tmp_path / "out"))
    assert set(res["files"]) == {"beat", "mix"} and res["vocal"] == "Song (vocal)"
    mix, _ = sf.read(store.root / res["files"]["mix"])
    assert mix.shape == (2 * sr, 2)                     # stereo, as long as the longest part
    assert 0.85 < np.max(np.abs(mix)) < 0.9            # normalised to about -1 dBFS
    assert Path(res["mp3"]).name == "Song - GLORE soft house.mp3"


def test_bar_offset_lines_up_downbeats():
    # song bars at 1.0 + k*1.857 (129.2 BPM); stretched to 129 BPM; beat bars start at 0.30
    f = 129 / 129.2
    bar = 240 / 129
    off = gen.bar_offset([1.0, 2.857], [0.30, 2.16], f, bar)
    assert abs(((1.0 / f) + off - 0.30) % bar) < 1e-6 or abs(((1.0 / f) + off - 0.30) % bar - bar) < 1e-6
    assert 0 <= off < bar
    assert gen.bar_offset([], [0.3], f, bar) == 0.0


def test_vocal_over_beat_end_to_end(store, tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from glore_engine.project import Layer, Source

    sr = 44100
    bpm = 128.0
    n = int(sr * 20)
    t = np.arange(n) / sr
    beat = np.zeros(n, dtype="float32")
    for k in range(int(20 * bpm / 60)):  # clicky four-on-the-floor
        i = int(k * 60 / bpm * sr)
        beat[i:i + 800] += np.sin(2 * np.pi * 60 * t[:800]) * np.linspace(1, 0, 800)
    (store.root / "generated").mkdir(parents=True, exist_ok=True)
    sf.write(store.root / "generated" / "b.wav", np.stack([beat, beat], 1) * 0.5, sr)
    store.upsert_layer(Layer(id="gen-b", name="beat", file="generated/b.wav", role="generated"))
    store.sources_dir.mkdir(parents=True, exist_ok=True)
    sf.write(store.sources_dir / "v.wav", np.stack([0.2 * np.sin(2 * np.pi * 330 * t[: sr * 10])] * 2, 1), sr)
    store.add_source(Source(id="song", name="Song", kind="full_song", file="v.wav", original_path="x", duration=10,
                            analysis={"bpm": 128.5, "downbeats": [0.5, 2.37]}))
    store.add_source(Source(id="v", name="Song (vocal)", kind="vocal", file="v.wav", original_path="x", duration=10,
                            derived_from="song", analysis={"bpm": 128.5}))
    monkeypatch.setattr(gen, "_mp3", lambda wav, mp3: (mp3.parent.mkdir(parents=True, exist_ok=True), mp3.write_bytes(b"x")))
    res = gen.mix_vocal_over_beat(store, lambda p, m: None, beat_layer="gen-b", vocal_source="v",
                                  arrange=False, export_dir=str(tmp_path / "o"))
    assert abs(res["beat_bpm"] - 128) < 3 and res["song_bpm"] == 128.5
    assert 0 <= res["offset_s"] < 240 / res["beat_bpm"]
    assert (store.root / res["file"]).exists() and res["mp3"].endswith("Song (GLORE).mp3")


def test_generate_from_song_covers_the_instrumental_and_lays_the_vocal(store, tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from glore_engine.project import Source

    sr = 44100
    store.sources_dir.mkdir(parents=True, exist_ok=True)
    sf.write(store.sources_dir / "song.wav", np.zeros((sr, 2), "float32"), sr)
    sf.write(store.sources_dir / "v.wav", np.full((sr // 2, 2), 0.1, "float32"), sr)
    store.add_source(Source(id="song", name="Song", kind="full_song", file="song.wav", original_path="x", duration=1,
                            analysis={"bpm": 129.2, "key_name": "E minor"}))
    store.add_source(Source(id="v", name="Song (vocal)", kind="vocal", file="v.wav", original_path="x", duration=0.5,
                            derived_from="song"))
    noise = lambda: (np.random.default_rng(1).standard_normal((2, sr)) * 0.1).astype("float32")
    monkeypatch.setattr(gen, "_stems_of", lambda *a, **k: {"drums": noise(), "bass": noise(), "other": noise(),
                                                           "vocals": noise()})
    seen = {}
    real = gen.run_generator
    monkeypatch.setattr(gen, "run_generator", lambda ace, job, *a, **k: (seen.update(job), real(ace, job, *a, **k))[1])
    monkeypatch.setattr(gen, "_mp3", lambda wav, mp3: (mp3.parent.mkdir(parents=True, exist_ok=True), mp3.write_bytes(b"x")))

    res = gen.generate_from_song(store, lambda p, m: None, vocal_source="v", caption="heavy house",
                                 strength=0.6, export_dir=str(tmp_path / "o"))
    assert seen["mode"] == "cover" and seen["strength"] == 0.6 and seen["lm"] is None
    assert seen["bpm"] == 129.2 and seen["key"] == "E minor" and abs(seen["duration"] - 1.0) < 0.01
    assert seen["lora"]["trigger"] == "glore_house"
    src = sf.read(seen["src_audio"])[0] if Path(seen["src_audio"]).exists() else None
    assert src is None  # temp folder is cleaned up afterwards
    ids = {l.id for l in store.project.layers}
    assert res["beat_layer"] in ids and res["song_layer"] in ids
    assert res["mp3"].endswith("Song (GLORE).mp3") and res["beat_mp3"].endswith("Song (GLORE) beat.mp3")


def test_generate_from_song_needs_the_original(store):
    from glore_engine.project import Source
    store.add_source(Source(id="v", name="My vocal", kind="vocal", file="v.wav", original_path="x", duration=1))
    with pytest.raises(gen.GenerateError, match="full song"):
        gen.generate_from_song(store, lambda p, m: None, vocal_source="v")
