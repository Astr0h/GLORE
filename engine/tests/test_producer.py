"""Song map -> plan (AI or rules) -> render -> measurements, on the fixture song."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import song_fixture as s  # noqa: E402

from glore_engine import producer, session, songmap  # noqa: E402
from glore_engine.session import Source  # noqa: E402

V = np.stack([s.voc, s.voc])
H = np.stack([s.harm, s.harm])


@pytest.fixture(scope="module")
def world():
    m = songmap.build(V, s.SR, s.BPM, "A minor", 0.0)
    riff = songmap.riff_loop(V, s.SR, s.BPM, "A minor")
    n = int(120 * s.BAR * s.SR)
    t = np.arange(n) / s.SR
    kick = np.zeros(n, np.float32)
    for b in range(int(n / s.SR / s.BEAT)):
        i = int(b * s.BEAT * s.SR)
        tt = np.arange(min(4000, n - i)) / s.SR
        kick[i:i + len(tt)] += np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
    src = {"vocal": Source(V), "chopriff": Source(riff), "bed.drums": Source(np.stack([kick, kick])),
           "bed.bass": Source(np.stack([0.3 * np.sin(2 * np.pi * 55 * t)] * 2).astype(np.float32)),
           "bed.other": Source(np.tile(H, 6)[:, :n])}
    bars = {k: v.bars(s.SR, s.BPM) for k, v in src.items()}
    return m, src, bars


def test_song_map_reads_the_song(world):
    m, _, _ = world
    kinds = [x["kind"] for x in m["sections"]]
    assert "chorus" in kinds and "verse" in kinds and m["delivery"]["style"] == "sung"
    assert m["hook"] and m["held_notes"] and set(m["sung_notes"]) <= {0, 2, 4, 7, 9, 11}
    assert len(json.dumps(m)) < 12000          # small enough to hand to the producer


def test_rule_plan_renders_as_an_arranged_song(world):
    m, src, bars = world
    plan, how = producer.make_plan(m, "warm and emotional", bars, "house", api_key="")
    assert how["by"] == "rules" and not how["warnings"]
    names = [x["name"] for x in plan["sections"]]
    assert names[0] == "intro" and names[-1] == "outro" and sum(n.startswith("drop") for n in names) == 2
    out = session.render(plan, src, s.SR)
    assert not out["warnings"]
    met = producer.measure(out, plan, s.SR)
    assert not met["vocal_clips_off_grid"] and not met["silent_bars"]
    by = {x["name"]: x for x in met["sections"]}
    assert by["drop 1"]["loudness_db"] > by["intro"]["loudness_db"] + 3      # the drop is louder than the intro
    assert met["energy_follows_plan"] > 0.5
    # the guide is the lead vocal alone, and the whole vocal is in it
    g = session.render(plan, src, s.SR, guide_only=True)
    assert list(g["tracks"]) == ["lead"]
    assert abs(float(np.abs(g["mix"]).sum()) / float(np.abs(V).sum()) - 1) < 0.02


def test_claude_plan_is_used_and_bad_replies_fall_back(world, monkeypatch):
    m, src, bars = world
    good = {"title": "T", "notes": "n", "bpm": 124, "bars": 24, "bed": {"caption": "garage"},
            "sections": [{"name": "a", "start": 0, "bars": 24, "energy": 0.5}],
            "tracks": [{"id": "lead", "group": "vocal", "guide": True, "clips": [{"at": 4, "src": "vocal", "from": 0, "bars": 8}]},
                       {"id": "d", "role": "drums", "clips": [{"at": 0, "src": "bed.drums", "from": 0, "bars": 24},
                                                             {"at": 0, "src": "made.up", "bars": 2}]}]}
    seen = {}

    def fake(system, user, model, key, **k):
        seen.update(system=system, user=user, model=model)
        return good, {"input_tokens": 10, "output_tokens": 20}
    monkeypatch.setattr(producer, "ask", fake)
    plan, how = producer.make_plan(m, "dark garage", bars, "house", model="deep", api_key="k")
    assert how["by"] == "claude" and how["model"] == "claude-fable-5-2" and plan["bpm"] == 124
    assert any("made.up" in w for w in how["warnings"])
    assert "dark garage" in seen["user"] and '"chopriff"' in seen["system"] and "bed.drums" in seen["system"]
    assert producer.guide_signature(plan) == producer.guide_signature(json.loads(json.dumps(plan)))
    # no lead vocal in the reply, or the API failing: the rules take over, with the reason kept
    monkeypatch.setattr(producer, "ask", lambda *a, **k: ({"tracks": [{"id": "d", "clips": [{"src": "bed.drums", "bars": 4}]}]}, {}))
    plan, how = producer.make_plan(m, "x", bars, api_key="k")
    assert how["by"] == "rules" and "lead vocal" in how["error"]

    def boom(*a, **k):
        raise RuntimeError("Anthropic API 529")
    monkeypatch.setattr(producer, "ask", boom)
    plan, how = producer.make_plan(m, "x", bars, api_key="k")
    assert how["by"] == "rules" and "529" in how["error"] and plan["tracks"]


def test_revise(world, monkeypatch):
    m, src, bars = world
    plan, _ = producer.make_plan(m, "", bars, api_key="")
    assert producer.revise(plan, {}, bars, api_key="")[0] is None
    monkeypatch.setattr(producer, "ask", lambda *a, **k: ({"ok": True}, {}))
    assert producer.revise(plan, {}, bars, api_key="k") == (None, {"by": "claude", "model": "claude-sonnet-5-5", "usage": {}, "verdict": "ok"})
    louder = json.loads(json.dumps(plan))
    louder["tracks"][0]["gain_db"] = 3
    monkeypatch.setattr(producer, "ask", lambda *a, **k: (louder, {}))
    new, how = producer.revise(plan, {}, bars, api_key="k")
    assert how["verdict"] == "revised" and new["tracks"][0]["gain_db"] == 3
    assert producer.guide_signature(new) == producer.guide_signature(plan)   # the bed can be reused


def test_extract_json():
    assert producer._extract_json('Here you go:\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        producer._extract_json("no json")
