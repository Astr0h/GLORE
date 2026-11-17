"""Engine tests using synthetic audio (no real songs needed).
Run:  python -m pytest engine/tests -q   (from the repo root, with the venv active)"""

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from glore_engine.analysis import analyse, parse_key
from glore_engine.project import ProjectStore
from glore_engine.server import create_app
from glore_engine.vocal_lock import lock_vocal

SR = 44100


def make_beat(path: Path, bpm=124.0, bars=8):
    """Four-on-the-floor kick + offbeat hats."""
    beat = 60.0 / bpm
    n = int(SR * beat * 4 * bars)
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(int(0.25 * SR)) / SR
    kick = (np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 12)).astype(np.float32)
    rng = np.random.default_rng(0)
    hat = (rng.standard_normal(int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / 200) * 0.3).astype(np.float32)
    for i in range(bars * 4):
        s = int(i * beat * SR)
        y[s:s + len(kick)] += kick[: n - s]
        h = int((i + 0.5) * beat * SR)
        if h < n:
            y[h:h + len(hat)] += hat[: n - h]
    sf.write(path, np.stack([y, y]).T * 0.8, SR)


def make_vocal(path: Path, bpm=100.0):
    """'Sung' phrases: A-minor notes with vibrato, separated by silence."""
    beat = 60.0 / bpm
    notes = [57, 60, 64, 62, 60, 57, 64, 69]  # A3 C4 E4 D4 C4 A3 E4 A4
    out = []
    for i, midi in enumerate(notes):
        dur = beat * 2
        t = np.arange(int(dur * SR)) / SR
        f = 440 * 2 ** ((midi - 69) / 12) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
        tone = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3))
        env = np.minimum(1, t / 0.05) * np.minimum(1, (dur - t) / 0.1)
        out.append(tone * env * 0.3)
        out.append(np.zeros(int(beat * 2 * SR)))  # rest between phrases
    y = np.concatenate(out).astype(np.float32)
    sf.write(path, np.stack([y, y]).T, SR)


@pytest.fixture
def files(tmp_path):
    beat, vocal = tmp_path / "beat.wav", tmp_path / "vocal.wav"
    make_beat(beat)
    make_vocal(vocal)
    return {"beat": beat, "vocal": vocal, "tmp": tmp_path}


def test_beat_tempo_and_kind(files):
    r = analyse(str(files["beat"]))
    assert abs(r.bpm - 124) < 2.5, r.bpm
    assert r.kind_guess == "beat"
    assert len(r.downbeats) >= 6


def test_vocal_key_phrases_kind(files):
    r = analyse(str(files["vocal"]))
    assert r.kind_guess == "vocal"
    assert r.key in ("A minor", "C major"), r.key  # relative keys share notes
    r2 = analyse(str(files["vocal"]), kind="vocal")
    assert 6 <= len(r2.phrases) <= 10, len(r2.phrases)


def test_parse_key():
    assert parse_key("Am") == (9, "minor")
    assert parse_key("F# major") == (6, "major")
    assert parse_key("Bb minor") == (10, "minor")
    assert parse_key("C") == (0, "major")


def test_vocal_lock_keeps_pitch_changes_length(files):
    out = files["tmp"] / "locked.wav"
    res = lock_vocal(files["vocal"], out, source_bpm=100, target_bpm=125)
    src_len = sf.info(files["vocal"]).duration
    assert abs(res.duration - src_len * 100 / 125) < 0.1
    # pitch preserved: key estimate unchanged
    assert analyse(str(out)).key == analyse(str(files["vocal"])).key


def test_vocal_lock_refuses_extreme_ratio(files):
    with pytest.raises(ValueError):
        lock_vocal(files["vocal"], files["tmp"] / "x.wav", source_bpm=60, target_bpm=170)


def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] != "running":
            assert j["status"] == "done", j
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_api_flow(files):
    client = TestClient(create_app(ProjectStore(files["tmp"] / "projects")))
    assert client.get("/health").json()["ok"]

    r = client.post("/sources/import", json={"path": str(files["vocal"])}).json()
    job = _wait(client, r["job_id"])
    assert job["result"]["kind"] == "vocal"
    sid = r["source"]["id"]

    proj = client.get("/project").json()
    assert proj["bpm"] is not None  # project follows the vocal's tempo

    # user corrects the BPM
    client.patch(f"/sources/{sid}/analysis", json={"bpm": 100})
    assert client.get("/project").json()["bpm"] == 100

    r = client.post("/vocal-lock", json={"source_id": sid, "target_bpm": 124}).json()
    lock = _wait(client, r["job_id"])["result"]
    assert lock["target_bpm"] == 124
    audio = client.get("/audio/" + lock["file"])
    assert audio.status_code == 200 and len(audio.content) > 1000

    # path traversal blocked
    assert client.get("/audio/../../etc/passwd").status_code in (403, 404)

    # bad requests
    assert client.post("/sources/import", json={"path": "nope.wav"}).status_code == 400
    assert client.post("/vocal-lock", json={"source_id": sid, "target_bpm": 300}).status_code == 400
    assert client.patch("/project", json={"monitor_mode": "loud"}).status_code == 400
