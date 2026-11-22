"""Chops, vocal FX and the mix, on synthetic audio (a sung line with gaps, a four-on-the-floor beat)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from glore_engine import chops, fx, mixing

SR, BPM = 44100, 128
BEAT = 60 / BPM


def vocal(bars=8):
    t = np.arange(int(bars * 4 * BEAT * SR)) / SR
    v = np.zeros_like(t)
    for i, f in enumerate([220, 261.6, 293.7, 329.6, 261.6, 220, 196, 220][:bars]):
        s, e = int(i * 4 * BEAT * SR), int((i * 4 + 3) * BEAT * SR)  # 3 beats sung, 1 beat gap
        tt = t[s:e] - t[s]
        ph = 2 * np.pi * f * tt + 0.3 * np.sin(2 * np.pi * 5.5 * tt)
        v[s:e] = sum(np.sin(k * ph) / k for k in range(1, 8)) * 0.3 * np.minimum(1, tt * 20) * np.minimum(1, (tt[-1] - tt) * 20)
    return np.stack([v, v]).astype(np.float32)


def vocal_then_break(bars=8):
    """Sung for the first half, then an instrumental break: where chops belong."""
    v = vocal(bars)
    v[:, int(bars / 2 * 4 * BEAT * SR):] = 0
    return v


def drums(n):
    k = np.zeros(n)
    for b in range(int(n / SR / BEAT)):
        s, L = int(b * BEAT * SR), int(0.25 * SR)
        tt = np.arange(min(L, n - s)) / SR
        k[s:s + len(tt)] += np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
    return np.stack([k, k]).astype(np.float32)


SONG = Path(__file__).with_name("song_fixture.py")


def load_song():
    import importlib.util
    spec = importlib.util.spec_from_file_location("song_fixture", SONG)
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s


@pytest.mark.parametrize("style", ["house", "rnb", "gospel"])
def test_chops_render_in_every_style(style):
    s = load_song()
    brk = np.zeros(int(8 * s.BAR * s.SR), np.float32)
    v1 = np.concatenate([s.voc, brk])          # the song, then an 8-bar instrumental break for the riff
    v = np.stack([v1, v1])
    layer, info = chops.make_chops(v, s.SR, s.BPM, "A minor", style=style, seed=1)
    assert layer.shape == v.shape and np.isfinite(layer).all()
    assert info.hits > 0 and 0 < float(np.abs(layer).max()) < 1 and info.in_song == 1.0
    again, _ = chops.make_chops(v, s.SR, s.BPM, "A minor", style=style, seed=1)
    assert np.allclose(layer, again)  # a seed gives the same chop; reshuffle = new seed
    other, _ = chops.make_chops(v, s.SR, s.BPM, "A minor", style=style, seed=2)
    assert not np.allclose(layer, other)


def test_chops_answer_the_vocal():
    v = vocal()
    layer, _ = chops.make_chops(v, SR, BPM, "A minor", style="house", seed=2, density=1.5)
    inside = np.mean([np.abs(layer[:, int(i * 4 * BEAT * SR) + SR // 10:int((i * 4 + 3) * BEAT * SR)]).mean() for i in range(8)])
    gaps = np.mean([np.abs(layer[:, int((i * 4 + 3) * BEAT * SR) + SR // 10:int((i * 4 + 4) * BEAT * SR)]).mean() for i in range(8)])
    assert gaps > 2 * inside


def test_house_chops_sit_on_the_16th_grid():
    v = vocal_then_break()
    _, info = chops.make_chops(v, SR, BPM, "A minor", style="house", seed=4)
    assert info.pattern and all(0 <= h["step"] < 16 for h in info.pattern)
    assert info.in_song == 1.0                                       # every chop note is a sung note


def test_chops_stay_out_of_the_verses():
    v = vocal_then_break(16)
    layer, info = chops.make_chops(v, SR, BPM, "A minor", style="house", seed=1)
    bar = int(4 * BEAT * SR)
    sung = np.abs(layer[:, : 8 * bar]).mean()
    brk = np.abs(layer[:, 8 * bar:]).mean()
    assert brk > 4 * sung                         # the riff lives in the break, not under the vocal
    assert info.roles[:8].count("riff") == 0 and info.roles[8:].count("riff") >= 6





def test_finds_the_chorus_reads_the_chords_and_plays_in_tune():
    import importlib.util
    import librosa
    spec = importlib.util.spec_from_file_location("song_fixture", SONG)
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    v, h = np.stack([s.voc, s.voc]), np.stack([s.harm, s.harm])
    layer, info = chops.make_chops(v, s.SR, s.BPM, "A minor", style="house", seed=3, harmony=h, structure=v + h)
    found = {b for a, e in info.chorus for b in range(int(round(a / s.BAR)), int(round(e / s.BAR)))}
    true = set(s.chorus_bars)
    assert len(found & true) >= 0.8 * len(true) and len(found - true) <= 2
    assert info.chords[:4] == ["Am", "F", "C", "G"]
    assert info.melody and {m["midi"] for m in info.melody} <= {64, 62, 60, 69, 67, 72, 71}  # the sung line
    assert info.notes and all(n["chorus"] for n in info.notes)          # held notes come from the chorus
    hs, he = info.hook
    assert any(a <= (hs + he) / 2 < e for a, e in info.chorus)           # so does the hook
    y = librosa.resample(layer.mean(0), orig_sr=s.SR, target_sr=22050)
    f0, vo, _ = librosa.pyin(y, fmin=100, fmax=2000, sr=22050, hop_length=512)
    pcs = np.round(librosa.hz_to_midi(f0[vo])) % 12
    chord_notes = {9, 0, 4, 5, 7, 11, 2}                                  # Am F C G
    assert np.isin(pcs, list(chord_notes)).mean() > 0.8


def test_silence_makes_no_chops():
    layer, info = chops.make_chops(np.zeros((2, SR * 4), np.float32), SR, BPM, style="house")
    assert info.hits == 0 and not layer.any()


@pytest.mark.parametrize("preset", fx.PRESETS)
def test_vocal_fx_presets(preset):
    v = vocal(2)
    out = fx.vocal_fx(v, SR, preset, BPM)
    assert out.shape == v.shape and np.isfinite(out).all()
    if preset == "none":
        assert np.allclose(out, v)


def test_more_reverb_means_a_longer_tail():
    v = vocal(2)
    tail = slice(int(7 * BEAT * SR) + 2000, int(8 * BEAT * SR))  # the gap after the last note
    dry = np.abs(fx.vocal_fx(v, SR, "airy", BPM, reverb=0.0, delay=0.0)[:, tail]).mean()
    wet = np.abs(fx.vocal_fx(v, SR, "airy", BPM, reverb=1.0, delay=0.0)[:, tail]).mean()
    assert wet > 3 * dry


def test_mix_sidechains_and_limits():
    v = vocal()
    n = v.shape[1]
    d = drums(n)
    bass = np.stack([np.sin(2 * np.pi * 55 * np.arange(n) / SR)] * 2).astype(np.float32) * 0.5
    song, parts = mixing.mix_song({"drums": d, "bass": bass}, v, SR, BPM)
    assert song.shape == (2, n) and float(np.abs(song).max()) <= 0.981
    assert sorted(parts) == ["bass", "drums", "vocal"]
    on = np.mean([np.abs(parts["bass"][:, int(b * BEAT * SR) + 300:int(b * BEAT * SR) + 1500]).mean() for b in range(4, 28)])
    off = np.mean([np.abs(parts["bass"][:, int((b + 0.85) * BEAT * SR):int((b + 0.95) * BEAT * SR)]).mean() for b in range(4, 28)])
    assert on < 0.7 * off  # the bass ducks on each kick


def test_pocket_only_dips_while_singing():
    v = vocal()
    n = v.shape[1]
    noise = (np.random.default_rng(0).standard_normal((2, n)) * 0.1).astype(np.float32)
    out = mixing.vocal_pocket(noise, v, SR, depth_db=6)
    sung = slice(int(1 * BEAT * SR), int(2.5 * BEAT * SR))
    gap = slice(int(3.6 * BEAT * SR), int(3.95 * BEAT * SR))
    lost = lambda s: 1 - np.abs(out[:, s]).mean() / np.abs(noise[:, s]).mean()
    assert lost(sung) > 2 * lost(gap)


# ------------------------------------------------- the cloud-side tool runner, for real (no GPU models needed)

TOOLS = Path(__file__).parents[1] / "glore_engine" / "pod" / "glore_tools.py"


def run_tool(tmp_path, op, item, params):
    # glore_tools.py imports GLORE's engine from ./engine next to it; on the pod that's synced, here point at the repo
    pod = tmp_path / "pod"
    pod.mkdir(exist_ok=True)
    (pod / "glore_tools.py").write_text(TOOLS.read_text())
    if not (pod / "engine").exists():
        (pod / "engine").symlink_to(Path(__file__).parents[1])
    jf = tmp_path / f"{op}.json"
    jf.write_text(json.dumps({"root": str(tmp_path / "vol"), "work": str(tmp_path / "work" / op), "item": item,
                              "params": params}))
    r = subprocess.run([sys.executable, str(pod / "glore_tools.py"), op, str(jf)], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith(("GLORE_RESULT", "GLORE_ERROR"))]
    assert line, r.stdout + r.stderr
    return json.loads(line[-1].split(" ", 1)[1])


def test_tools_analyse_lock_and_chop_for_real(tmp_path):
    root = tmp_path / "vol"
    d = root / "library" / "abc"
    d.mkdir(parents=True)
    sf.write(d / "v.flac", vocal().T, SR)
    item = {"id": "abc", "name": "My vocal", "kind": "vocal", "file": "library/abc/v.flac"}
    (d / "item.json").write_text(json.dumps(item))
    res = run_tool(tmp_path, "analyse", item, {})
    upd = res["updated"]
    assert upd["key"] in ("A minor", "C major") and upd["bpm"] > 0
    assert json.loads((d / "item.json").read_text())["key"] == upd["key"]
    res = run_tool(tmp_path, "lock", upd, {"source_bpm": 128, "target_bpm": 140, "pitch": 2})
    new = res["items"][0]
    assert new["bpm"] == 140 and new["parent"] == "abc" and "140 BPM, +2 st" in new["name"]
    assert abs(new["duration"] - upd["duration"] * 128 / 140) < 0.2
    assert (root / new["file"]).exists()
    res = run_tool(tmp_path, "chops", upd, {"style": "house", "seed": 7, "bpm": 128})
    ch = res["items"][0]
    assert ch["kind"] == "chops" and ch["hits"] > 0 and (root / ch["file"]).exists()
    err = run_tool(tmp_path, "lock", {**item, "bpm": None}, {"target_bpm": 140})
    assert "Analyse it first" in err["error"]


# ------------------------------------------------- structure, reference mastering, AI textures

from glore_engine import structure


def test_structure_adds_intro_drops_and_outro_on_the_bar_grid():
    s = load_song()
    v = np.stack([s.voc, s.voc])
    n_bars = int(np.ceil(len(s.voc) / s.SR / s.BAR))
    act = chops.vocal_activity(s.voc, s.SR, s.BPM, 0, n_bars)
    regions, _ = chops.find_chorus(chops._mono(v, s.SR), s.BPM, 0, n_bars, act)
    plan = structure.plan_structure(v, s.SR, s.BPM, 0.0, regions, "house")
    kinds = [i.kind for i in plan.inserts]
    assert kinds[0] == "intro" and kinds[-1] == "outro" and kinds.count("drop") == 2
    bar = plan.bar_samples
    assert all(i.at % bar == 0 and i.bars in (8,) for i in plan.inserts)  # whole bars, on bar lines
    arranged = structure.apply_plan(v, plan)
    assert arranged.shape[1] == v.shape[1] + sum(i.bars for i in plan.inserts) * bar
    # the vocal is all still there, in order, nothing cut
    assert np.allclose(np.abs(arranged).sum(), np.abs(v).sum(), rtol=1e-6)
    assert np.allclose(arranged[:, 8 * bar:8 * bar + 1000], v[:, :1000])
    secs = [x["part"] for x in plan.sections]
    assert secs == ["intro", "vocal", "drop", "vocal", "drop", "vocal", "outro"]
    assert "drops" in structure.caption_hint(plan)


def test_structure_keeps_an_existing_intro_and_stays_under_the_length_cap():
    s = load_song()
    pad = np.zeros(int(8 * s.BAR * s.SR), np.float32)
    v1 = np.concatenate([pad, s.voc])
    v = np.stack([v1, v1])
    plan = structure.plan_structure(v, s.SR, s.BPM, 0.0, [], "house")
    assert "intro" not in [i.kind for i in plan.inserts]
    long = structure.plan_structure(v, s.SR, s.BPM, 0.0, [(4, 8), (12, 16)], "house", max_seconds=v1.size / s.SR + 20)
    assert v1.size / s.SR + sum(i.bars for i in long.inserts) * long.bar_samples / s.SR <= v1.size / s.SR + 20


def test_arrange_tool_for_real(tmp_path):
    s = load_song()
    sf.write(tmp_path / "v.wav", np.stack([s.voc, s.voc]).T, s.SR)
    res = run_tool(tmp_path, "arrange", {}, {"vocal": str(tmp_path / "v.wav"), "bpm": s.BPM, "style": "house"})
    out, _ = sf.read(res["vocal"])
    assert abs(out.shape[0] / s.SR - res["duration"]) < 0.05 and res["duration"] > len(s.voc) / s.SR + 30
    assert res["plan"]["sections"][0]["part"] == "intro" and "intro" in res["hint"]


def test_reference_master_matches_the_reference(tmp_path):
    pytest.importorskip("matchering")
    import importlib.util
    spec = importlib.util.spec_from_file_location("glore_tools", TOOLS)
    gt = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(__file__).parents[1]))
    spec.loader.exec_module(gt)
    root = tmp_path / "vol"
    ds = root / "datasets" / "house"
    ds.mkdir(parents=True)
    rng = np.random.default_rng(1)
    t = np.arange(SR * 40) / SR
    for i in range(3):  # loud, bright "training tracks"
        x = 0.35 * np.sin(2 * np.pi * (100 + 30 * i) * t) * np.sign(np.sin(2 * np.pi * 2 * t)) + 0.08 * rng.standard_normal(len(t))
        sf.write(ds / f"t{i}_full.flac", np.stack([x, x], 1).clip(-1, 1), SR)
    quiet = (0.05 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    song = np.stack([quiet, quiet])
    out, label = gt.master_to_reference(song, root, tmp_path, "pack", "house")
    assert label == "house pack" and (root / "references" / "house.wav").exists()
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)))
    ref, _ = sf.read(root / "references" / "house.wav")
    assert abs(rms(out) - rms(ref)) < 1.5 and rms(out) > rms(song) + 10
    # no reference audio: GLORE's own master, never a crash
    out2, label2 = gt.master_to_reference(song, tmp_path / "empty", tmp_path, "pack", "rnb")
    assert label2 is None and np.isfinite(out2).all()


def test_ai_textures_only_fill_the_gaps():
    v = vocal_then_break()
    noise = (np.random.default_rng(0).standard_normal(v.shape) * 0.2).astype(np.float32)
    tex = mixing.ai_textures(noise, v, SR, level_db=-9)
    bar = int(4 * BEAT * SR)
    sung = np.mean([np.abs(tex[:, b * bar + int(0.5 * BEAT * SR):b * bar + int(2.7 * BEAT * SR)]).mean() for b in range(4)])
    gap = np.abs(tex[:, 5 * bar:7 * bar]).mean()
    assert gap > 5 * sung
