"""Session 2: link import parsing, raw vocal chain, genre tags + training sets.
Heavy models are swapped for fakes so tests run fast and offline."""

import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from glore_engine import cleanup, datasets, linkimport
from glore_engine.project import ProjectStore
from glore_engine.server import create_app

from test_engine import make_vocal, make_beat


# ---------------------------------------------------------------- link import

def test_detect_service():
    assert linkimport.detect_service("https://www.youtube.com/watch?v=abc") == "youtube"
    assert linkimport.detect_service("https://youtu.be/abc") == "youtube"
    assert linkimport.detect_service("https://soundcloud.com/a/b") == "soundcloud"
    assert linkimport.detect_service("https://open.spotify.com/track/123") == "spotify"
    assert linkimport.detect_service("https://music.apple.com/ie/album/x/1?i=2") == "apple"
    with pytest.raises(linkimport.LinkError):
        linkimport.detect_service("https://example.com/song.mp3")


def test_parse_spotify():
    page = ('<meta property="og:title" content="Delilah (pull me out of this)"/>'
            '<meta property="og:description" content="Fred again.. · Delilah (pull me out of this) · Song · 2023"/>')
    assert linkimport.parse_spotify(page) == ("Delilah (pull me out of this)", "Fred again..")


def test_parse_apple():
    page = '<meta property="og:title" content="Marea (we&#39;ve lost dancing) - Single by Fred again.. on Apple Music">'
    assert linkimport.parse_apple(page) == ("Marea (we've lost dancing)", "Fred again..")
    page2 = '<meta content="Song Name by Some Artist on Apple Music" property="og:title">'
    assert linkimport.parse_apple(page2) == ("Song Name", "Some Artist")


def test_safe_name():
    assert linkimport.safe_name('a/b:c*?"d') == "abcd"
    assert linkimport.safe_name("") == "Imported"


# ---------------------------------------------------------------- fakes

def fake_run_stage(stage, in_path, work_dir, keep=None):
    """Pretend model: halves the volume and returns the kept stem."""
    a, sr = sf.read(in_path)
    out = Path(work_dir) / f"x_({keep or cleanup.STAGES[stage][1]})_{stage}.wav"
    sf.write(out, a * 0.5, sr)
    return out


def fake_separate(path, out_dir, **_):
    a, sr = sf.read(path)
    out = {}
    for name in ("vocals", "drums", "bass", "other"):
        p = Path(out_dir) / f"{name}.wav"
        sf.write(p, a * 0.25, sr)
        out[name] = p
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GLORE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cleanup, "run_stage", fake_run_stage)
    vocal, beat = tmp_path / "vocal.wav", tmp_path / "beat.wav"
    make_vocal(vocal)
    make_beat(beat)
    client = TestClient(create_app(ProjectStore(tmp_path / "projects")))
    return {"client": client, "vocal": vocal, "beat": beat, "tmp": tmp_path}


def wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] != "running":
            assert j["status"] == "done", j
            return j["result"]
        time.sleep(0.2)
    raise TimeoutError


def wait_analysed(client, sid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = next(x for x in client.get("/project").json()["sources"] if x["id"] == sid)
        if s["analysis"]:
            return s
        time.sleep(0.2)
    raise TimeoutError


# ---------------------------------------------------------------- raw vocal

def test_raw_vocal_chain_order(tmp_path, monkeypatch):
    calls = []

    def spy(stage, in_path, work_dir, keep=None):
        calls.append(stage)
        return fake_run_stage(stage, in_path, work_dir, keep)

    monkeypatch.setattr(cleanup, "run_stage", spy)
    src = tmp_path / "v.wav"
    make_vocal(src)
    res = cleanup.raw_vocal(src, tmp_path / "out.wav", ["noise", "backing", "reverb"])
    assert calls == ["backing", "reverb", "noise"]  # always run in the right order
    assert res.path.exists()
    with pytest.raises(ValueError):
        cleanup.raw_vocal(src, tmp_path / "o2.wav", [])


def test_raw_vocal_api(env):
    c = env["client"]
    r = c.post("/sources/import", json={"path": str(env["vocal"]), "kind": "vocal"}).json()
    wait(c, r["job_id"])
    sid = r["source"]["id"]
    c.patch(f"/sources/{sid}/analysis", json={"bpm": 100})
    res = wait(c, c.post(f"/sources/{sid}/raw-vocal", json={"stages": ["reverb", "echo"]}).json()["job_id"])
    raw = wait_analysed(c, res["source_id"])
    assert raw["name"].endswith("(raw)") and raw["kind"] == "vocal" and raw["origin"] == "raw"
    assert raw["overrides"]["bpm"] == 100
    assert c.post(f"/sources/{sid}/raw-vocal", json={"stages": ["autotune"]}).status_code == 400


# ---------------------------------------------------------------- tags + datasets

def test_tagging_and_prepare(env, monkeypatch):
    c = env["client"]
    r = c.post("/sources/import", json={"path": str(env["beat"]), "kind": "full_song"}).json()
    wait(c, r["job_id"])
    sid = r["source"]["id"]

    out = c.post(f"/sources/{sid}/tag", json={"pack": "gospel_soul", "moods": ["choir", "ethereal", " "]}).json()
    assert out["source"]["pack"] == "gospel_soul" and out["source"]["moods"] == ["choir", "ethereal"]
    gospel = next(p for p in out["packs"] if p["id"] == "gospel_soul")
    assert gospel["count"] == 1 and gospel["prepared"] == 0
    item = c.get("/datasets/gospel_soul").json()[0]
    assert item["caption"].startswith("gospel / soul, choir, ethereal")

    # moving packs moves the track, no duplicates
    c.post(f"/sources/{sid}/tag", json={"pack": "house", "moods": ["hype"]})
    counts = {p["id"]: p["count"] for p in c.get("/packs").json()}
    assert counts == {"house": 1, "rnb_rap": 0, "gospel_soul": 0}

    # prepare (instrumental mode) with fake separation
    monkeypatch.setattr("glore_engine.separation.separate", fake_separate)
    wait(c, c.post("/datasets/house/prepare").json()["job_id"])
    item = c.get("/datasets/house").json()[0]
    assert item["prepared"] and item["training_audio"] == "instrumental"
    train = datasets.pack_dir("house") / f"{item['id']}_train.wav"
    assert train.exists()

    # removing from the dataset clears the tag on the sample
    c.delete(f"/datasets/house/{item['id']}")
    s = next(x for x in c.get("/project").json()["sources"] if x["id"] == sid)
    assert s["pack"] is None
    assert c.post(f"/sources/{sid}/tag", json={"pack": "polka"}).status_code == 400


def test_prepare_keep_backing(env):
    path = env["tmp"] / "song.wav"
    shutil.copy(env["beat"], path)
    datasets.add_or_update("gospel_soul", path, "song", ["choir"])
    used = []

    def backing(vocal_path, tmp):
        used.append(vocal_path)
        return vocal_path  # pretend the whole vocal stem is backing

    datasets.prepare("gospel_soul", separate_fn=fake_separate, backing_fn=backing)
    assert used, "keep_backing mode must run the lead/backing split"
    it = datasets.list_items("gospel_soul")[0]
    assert it["prepared"] and it["training_audio"] == "keep_backing"


def test_legacy_pack_names_migrate(tmp_path):
    store = ProjectStore(tmp_path / "p")
    store.project.genre_pack = "melodic_rap"
    store.save()
    assert ProjectStore(tmp_path / "p").project.genre_pack == "rnb_rap"


def test_link_import_rejects_unknown(env):
    assert env["client"].post("/sources/import-link", json={"url": "https://example.com/x"}).status_code == 400


def test_friendly_link_errors():
    import urllib.error
    assert "internet" in linkimport.friendly_error(urllib.error.URLError("Tunnel connection failed"), "spotify")
    assert "private" in linkimport.friendly_error(Exception("ERROR: Private video"), "youtube")
