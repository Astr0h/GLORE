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
    from glore_engine import fxrack
    for part, x in fxrack.make_kit("house", 120, s.SR, s.BPM).items():
        src[f"kit.{part}"] = Source(x)
    session.calibrate(src, s.SR)
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
    assert by["drop 1"]["drums_minus_mix_db"] > -9 and not met["problems"]     # and it has real drums
    assert plan["bed"]["caption"].startswith("warm and emotional") and plan["style"] == "house"
    assert met["energy_follows_plan"] > 0.5
    # the guide is the lead vocal alone, and the whole vocal is in it
    g = session.render(plan, src, s.SR, guide_only=True)
    assert list(g["tracks"]) == ["lead", "kit"]               # the beat is built around the vocal and the drums
    assert abs(float(np.abs(g["tracks"]["lead"]).sum()) / float(np.abs(src["vocal"].audio).sum()) - 1) < 0.02


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
    moved = json.loads(json.dumps(plan))
    kit = next(t for t in moved["tracks"] if t["id"] == "kit")
    kit["clips"][0]["bars"] += 1
    assert producer.guide_signature(moved) != producer.guide_signature(plan)   # drums changed: rebuild it


def test_extract_json():
    assert producer._extract_json('Here you go:\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        producer._extract_json("no json")


def test_cloud_tools_map_guide_and_render_for_real(tmp_path, monkeypatch):
    """The three producer steps as they run on the GPU, on real audio (stem separation faked)."""
    import importlib.util
    import soundfile as sf
    import zipfile
    from glore_engine import separation
    spec = importlib.util.spec_from_file_location("glore_tools", Path(__file__).parents[1] / "glore_engine" / "pod" / "glore_tools.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    SR = gt.SR
    root, work = tmp_path / "vol", tmp_path / "work"
    work.mkdir()
    sf.write(tmp_path / "v.wav", V.T, s.SR)

    def fake_separate(src, out_dir, model_name="htdemucs"):
        x, sr = sf.read(str(src))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out = {}
        for i, k in enumerate(("drums", "bass", "other", "vocals")):
            sf.write(Path(out_dir) / f"{k}.wav", x * (0.5 if k != "vocals" else 0.1), sr)
            out[k] = Path(out_dir) / f"{k}.wav"
        return out
    monkeypatch.setattr(separation, "separate", fake_separate)
    r = gt.op_pmap({}, root, work, {}, {"sid": "abc", "vocal": str(tmp_path / "v.wav"), "bpm": s.BPM, "key": "A minor"})
    assert r["source_bars"]["vocal"] == r["songmap"]["bars"] and r["source_bars"]["chopriff"] == 4
    plan, how = producer.make_plan(r["songmap"], "", r["source_bars"], "house", api_key="")
    plan["bpm"] = 126.0                                   # the producer nudges the tempo: sources get conformed
    g = gt.op_pguide({}, root, work, {}, {"sid": "abc", "plan": plan})
    guide, _ = sf.read(g["guide"])
    assert abs(g["duration"] - plan["bars"] * 240 / 126) < 0.1 and not g["warnings"]
    # a "generated" instrumental as long as the guide
    t = np.arange(len(guide)) / SR
    bed = tmp_path / "bed.wav"
    sf.write(bed, np.stack([0.3 * np.sin(2 * np.pi * 110 * t)] * 2).T, SR)
    out = gt.op_prender({}, root, work, {}, {"sid": "abc", "plan": plan, "bed": str(bed), "out_base": str(tmp_path / "out" / "Song"),
                                              "master": "none"})
    assert Path(out["mp3"]).stat().st_size > 10000 and Path(out["beat_mp3"]).exists()
    assert abs(out["duration"] - g["duration"]) < 0.1 and not out["metrics"]["engine_warnings"]
    names = zipfile.ZipFile(out["zip"]).namelist()
    assert "Song - lead.wav" in names and "Song - plan.json" in names and "Song - master.wav" in names
    assert (root / "glore" / "produce" / "abc" / "conf_126_0" / "vocal.wav").exists()


def test_read_stream():
    ev = lambda d: ("data: " + json.dumps(d)).encode()
    lines = [b"event: message_start", ev({"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
             ev({"type": "content_block_delta", "delta": {"type": "text_delta", "text": '{"a"'}}), b"", b"data: [junk",
             ev({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
             ev({"type": "content_block_delta", "delta": {"type": "text_delta", "text": ": 1}"}}),
             ev({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}})]
    assert producer.read_stream(lines) == ('{"a": 1}', "end_turn", {"input_tokens": 9, "output_tokens": 5})
    with pytest.raises(RuntimeError, match="overloaded"):
        producer.read_stream([ev({"type": "error", "error": {"message": "overloaded"}})])


def test_bed_clips_default_to_aligned():
    p, _ = session.clean_plan({"bpm": 120, "tracks": [{"id": "d", "clips": [{"at": 8, "src": "bed.drums", "bars": 4},
                                                                           {"at": 8, "src": "vocal", "bars": 4}]}]})
    assert [c["from"] for c in p["tracks"][0]["clips"]] == [8, 0]


def test_a_drop_without_drums_gets_a_kit_and_weak_mixes_are_flagged(world):
    m, src, bars = world
    raw = {"bpm": s.BPM, "bars": 16, "bed": {"caption": "deep house"},
           "sections": [{"name": "verse", "start": 0, "bars": 8, "energy": 0.4}, {"name": "drop", "start": 8, "bars": 8, "energy": 1}],
           "tracks": [{"id": "lead", "group": "vocal", "guide": True, "clips": [{"at": 0, "src": "vocal", "bars": 16}]},
                      {"id": "bass", "role": "bass", "clips": [{"at": 0, "src": "bed.bass", "bars": 16}]},
                      {"id": "fx", "group": "fx", "clips": [{"at": 6, "src": "synth:riser", "bars": 2, "gain_db": 12}]}]}
    plan, _ = session.clean_plan(json.loads(json.dumps(raw)), bars)
    assert plan["bed"] == {"caption": "deep house"}                       # the producer's caption reaches the generator
    met = producer.measure(session.render(plan, src, s.SR), plan, s.SR)
    assert any("drop: high energy but the drums are weak" in p for p in met["problems"])
    assert any("no drum tracks" in p for p in met["problems"])
    assert session.ensure_backbone(plan) == ["drop"]
    out = session.render(plan, src, s.SR)
    met = producer.measure(out, plan, s.SR)
    assert not any("drums" in p for p in met["problems"]) and met["sections"][1]["drums_minus_mix_db"] > -9
    assert session.ensure_backbone(plan) == []                             # already covered: nothing more added


def test_sources_are_level_matched():
    from glore_engine import fxrack
    t = np.arange(s.SR * 8) / s.SR
    quiet = np.stack([0.01 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    loud = np.stack([0.8 * np.sin(2 * np.pi * 110 * t)] * 2).astype(np.float32)
    src = {"vocal": Source(quiet), "bed.bass": Source(loud), "bed.other": Source(loud * 0.5), "bed.vox": Source(loud)}
    gains = session.calibrate(src, s.SR)
    db = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)))
    assert abs(db(src["vocal"].audio) + 20) < 0.5 and abs(db(src["bed.bass"].audio + src["bed.other"].audio) + 20) < 0.5
    assert gains["vocal"] > 20 and gains["bed.bass"] == gains["bed.other"] == gains["bed.vox"] < 0
    kit = fxrack.make_kit("house", 4, s.SR, 124)
    bar = int(round(240 / 124 * s.SR))
    assert set(kit) == {"kick", "clap", "hats", "perc", "full"} and kit["full"].shape == (2, 4 * bar)
    k = np.abs(kit["kick"][0])
    assert all(k[int(b * bar / 4):int(b * bar / 4) + 400].max() > 0.3 for b in range(16))   # four on the floor
    assert -24 < db(kit["full"]) < -13                                                       # sits with a -20 dB vocal
