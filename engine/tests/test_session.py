"""The multitrack session engine and the effects rack."""

import numpy as np
import pytest

from glore_engine import fxrack, session
from glore_engine.session import Source

SR, BPM = 22050, 120
BAR = int(240 / BPM * SR)


def tone(f, bars, amp=0.3):
    t = np.arange(int(bars * BAR)) / SR
    x = (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)
    return np.stack([x, x])


def kicks(bars):
    x = np.zeros(int(bars * BAR), np.float32)
    for b in range(int(bars * 4)):
        s = b * BAR // 4
        t = np.arange(BAR // 8) / SR
        x[s:s + len(t)] += (np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 12)).astype(np.float32)
    return np.stack([x, x])


SOURCES = {"vocal": Source(np.concatenate([np.zeros((2, SR // 2), np.float32), tone(440, 8)], 1), offset=0.5),
           "bed.drums": Source(kicks(16)), "bed.bass": Source(tone(55, 16, 0.4))}


def plan(**over):
    p = {"bpm": BPM, "bars": 16, "tracks": [
        {"id": "lead", "group": "vocal", "guide": True, "clips": [{"at": 4, "src": "vocal", "from": 0, "bars": 8}]},
        {"id": "drums", "role": "drums", "clips": [{"at": 0, "src": "bed.drums", "from": 0, "bars": 16}]},
        {"id": "bass", "role": "bass", "sidechain_db": 9, "clips": [{"at": 8, "src": "bed.bass", "from": 8, "bars": 8}]}]}
    p.update(over)
    return p


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_clips_land_where_the_plan_says():
    p, w = session.clean_plan(plan(), {k: v.bars(SR, BPM) for k, v in SOURCES.items()})
    out = session.render(p, SOURCES, SR)
    assert not w and not out["warnings"] and out["mix"].shape == (2, 16 * BAR)
    lead = out["tracks"]["lead"]
    assert rms(lead[:, :4 * BAR - 200]) < 1e-5 and rms(lead[:, 4 * BAR + 500:12 * BAR - 500]) > 0.05   # bar 4..12
    assert rms(lead[:, 12 * BAR + 200:]) < 1e-5
    assert rms(out["tracks"]["bass"][:, :8 * BAR - 200]) < 1e-5
    assert float(np.abs(out["mix"]).max()) <= 0.72           # -3 dB headroom for mastering
    assert set(out["groups"]) == {"vocal", "drums", "bass"}


def test_bass_ducks_on_the_kick():
    p, _ = session.clean_plan(plan())
    b = session.render(p, SOURCES, SR)["tracks"]["bass"]
    on = np.mean([rms(b[:, 8 * BAR + k * BAR // 4 + 200:8 * BAR + k * BAR // 4 + 900]) for k in range(4, 28)])
    off = np.mean([rms(b[:, 8 * BAR + k * BAR // 4 + int(0.85 * BAR / 4):8 * BAR + (k + 1) * BAR // 4 - 50]) for k in range(4, 28)])
    assert on < 0.75 * off


def test_guide_is_only_the_marked_tracks_dry():
    p, _ = session.clean_plan(plan())
    p["tracks"][0]["fx"] = [{"type": "reverb", "mix": 0.9}]
    g = session.render(p, SOURCES, SR, guide_only=True)
    assert list(g["tracks"]) == ["lead"] and rms(g["mix"][:, 13 * BAR:]) < 1e-5   # no reverb tail in the guide


def test_loops_stutters_and_synths():
    p, w = session.clean_plan({"bpm": BPM, "bars": 8, "tracks": [
        {"id": "loop", "clips": [{"at": 0, "src": "bed.drums", "from": 0, "bars": 8, "loop": 1}]},
        {"id": "stut", "group": "vocal", "clips": [{"at": 2, "src": "vocal", "from": 1, "bars": 1,
                                                    "fx": [{"type": "stutter", "slice": 0.25}]}]},
        {"id": "fx", "group": "fx", "clips": [{"at": 6, "src": "synth:riser", "bars": 2},
                                               {"at": 0, "src": "synth:laser", "bars": 1}]}]},
        {k: v.bars(SR, BPM) for k, v in SOURCES.items()})
    assert w == ["track fx: no synth called laser"]
    out = session.render(p, SOURCES, SR)
    r = out["tracks"]["fx"]
    assert rms(r[:, 7 * BAR + BAR // 2:]) > 3 * rms(r[:, 6 * BAR:6 * BAR + BAR // 2])   # the riser rises
    s = out["tracks"]["stut"][:, 2 * BAR:3 * BAR]
    q = BAR // 16
    assert np.allclose(s[:, 300:q - 300], s[:, q + 300:2 * q - 300], atol=1e-3)           # the slice repeats


def test_sends_make_a_shared_reverb_return():
    p, _ = session.clean_plan(plan())
    p["tracks"][0]["sends"] = {"reverb": 0.5}
    out = session.render(p, SOURCES, SR)
    assert rms(out["tracks"]["fx_returns"][:, 12 * BAR + 500:13 * BAR]) > 1e-4   # the tail rings after the vocal stops


def test_a_broken_plan_still_renders():
    bad = {"bpm": "fast", "bars": 9999, "tracks": [
        {"id": "a", "clips": [{"at": "x", "src": "vocal", "bars": -3, "fx": [{"type": "nonsense"}, "junk", {"type": "eq", "hp": "abc"}]}]},
        {"id": "a", "clips": [{"at": 2, "src": "ghost", "bars": 2}]}, "junk", {"id": "empty"}],
        "buses": "no", "mix": 7, "master_fx": [{"type": "filter", "cutoff": [[0, 300], [8, 18000]]}]}
    p, w = session.clean_plan(bad, {k: v.bars(SR, BPM) for k, v in SOURCES.items()})
    assert p["bpm"] == 120 and len(p["tracks"]) == 1 and any("ghost" in x for x in w)
    out = session.render(p, SOURCES, SR)
    assert np.isfinite(out["mix"]).all() and len(out["warnings"]) == 2


@pytest.mark.parametrize("name", sorted(fxrack.FX))
def test_every_effect_runs(name):
    x = tone(220, 2) * (np.sin(np.arange(2 * BAR) / SR * 12) > 0)
    extra = {"filter": {"cutoff": [[0, 200], [1, 8000]]}, "gain": {"db": [[0, -20], [1, 0]]}, "fade": {"in": 1, "out": 1},
             "pitch": {"semis": 3}, "vocoder": {"per_bar": [[45, 57, 60], [43, 55, 59]]},
             "delay": {"send": [[0, 0], [0.7, 0], [0.75, 1]]}, "eq": {"hp": 200, "high_db": 3}, "width": {"amount": 1.5},
             "pan": {"pos": 0.5}}.get(name, {})
    w = []
    y = fxrack.apply_fx(x.astype(np.float32), SR, [{"type": name, **extra}], {"bpm": BPM, "unit": "frac"}, w)
    assert not w and y.shape == x.shape and np.isfinite(y).all()
    assert not np.allclose(y, x) or name == "width"   # mono in = nothing to widen


def test_automation_sweeps():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, 4 * BAR)) * 0.2).astype(np.float32)
    y = fxrack.apply_fx(x, SR, [{"type": "filter", "mode": "lp", "cutoff": [[0, 150], [4, 9000]]}],
                        {"bpm": BPM, "pos0": 0, "pos1": 4})
    assert rms(y[:, 3 * BAR:]) > 2 * rms(y[:, :BAR])   # opens up over the 4 bars
