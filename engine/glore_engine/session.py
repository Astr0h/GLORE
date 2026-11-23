"""GLORE's multitrack session: a song as a plan, rendered like a DAW would.

A plan is plain JSON (the producer brain writes it; a person could too). Time is in bars.

    {"bpm": 126, "bars": 96,
     "sections": [{"name": "intro", "start": 0, "bars": 8, "energy": 0.3}, ...],
     "tracks": [
       {"id": "lead", "group": "vocal", "guide": true, "gain_db": 0, "sends": {"reverb": 0.2},
        "fx": [{"type": "compress"}],
        "clips": [{"at": 8, "src": "vocal", "from": 0, "bars": 16},
                  {"at": 24, "src": "vocal", "from": 12, "bars": 0.5, "fx": [{"type": "stutter", "slice": 0.25}]}]},
       {"id": "drums", "group": "music", "role": "drums",
        "clips": [{"at": 8, "src": "bed.drums", "from": 8, "bars": 88}]},
       {"id": "risers", "group": "fx", "clips": [{"at": 22, "src": "synth:riser", "bars": 2}]}],
     "buses": {"reverb": {"size": 0.75}, "delay": {"time": 0.75, "feedback": 0.45}},
     "mix": {"sidechain_db": 5, "pocket_db": 3},
     "master_fx": [{"type": "filter", "mode": "lp", "cutoff": [[0, 400], [8, 18000]]}]}

A clip plays `bars` of source `src` starting at source bar `from`, placed at output bar `at`.
`loop` (bars) repeats that much of the source to fill the clip. Clip fx run on the clip alone
(automation positions 0..1 across the clip); track fx run on the whole track (positions in bars).
Sources are audio buffers already at the session tempo, each with the time of its bar 0.

    out = render(plan, {"vocal": Source(audio, offset_s), "bed.drums": Source(audio)}, sr)
    out["mix"], out["tracks"]["lead"], out["warnings"]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fxrack
from .fxrack import apply_fx

MAX_SECONDS = 330.0
DEFAULT_BUSES = {"reverb": {"size": 0.75, "damping": 0.5}, "delay": {"time": 0.75, "feedback": 0.42}}


@dataclass
class Source:
    audio: np.ndarray          # (2, n) float32 at the session's sample rate and tempo
    offset: float = 0.0        # seconds: where the source's bar 0 starts

    def bars(self, sr: int, bpm: float) -> float:
        return max(0.0, (self.audio.shape[1] / sr - self.offset) / (240.0 / bpm))


def conform(audio: np.ndarray, sr: int, src_bpm: float, bpm: float, transpose: float = 0.0) -> np.ndarray:
    """Bring a source to the session's tempo (and key), keeping the voice's character."""
    if abs(src_bpm - bpm) < 0.01 and abs(transpose) < 0.01:
        return audio.astype(np.float32)
    import pedalboard as pb
    return pb.time_stretch(audio.astype(np.float32), sr, stretch_factor=bpm / src_bpm,
                           pitch_shift_in_semitones=float(transpose), high_quality=True,
                           preserve_formants=True).astype(np.float32)


def _num(v, default, lo=None, hi=None) -> float:
    try:
        v = float(v)
        if not np.isfinite(v):
            raise ValueError
    except (TypeError, ValueError):
        v = float(default)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def clean_plan(plan: dict, source_bars: dict[str, float] | None = None) -> tuple[dict, list[str]]:
    """Make any plan safe to render: clamp numbers, drop what can't work, say what was dropped."""
    w: list[str] = []
    bpm = _num(plan.get("bpm"), 120, 60, 200)
    bar_s = 240.0 / bpm
    max_bars = int(MAX_SECONDS / bar_s)
    tracks = []
    end = 0.0
    for ti, t in enumerate((plan.get("tracks") or [])[:32]):
        if not isinstance(t, dict):
            continue
        clips = []
        for c in (t.get("clips") or [])[:600]:
            if not isinstance(c, dict) or not c.get("src"):
                continue
            src = str(c["src"])
            if src.startswith("synth:"):
                if src[6:] not in fxrack.SYNTHS:
                    w.append(f"track {t.get('id', ti)}: no synth called {src[6:]}")
                    continue
            elif source_bars is not None and src not in source_bars and not src.startswith("kit."):
                w.append(f"track {t.get('id', ti)}: no source called {src}")
                continue
            at, bars = _num(c.get("at"), 0, 0, max_bars), _num(c.get("bars"), 1, 1 / 64, max_bars)
            if at >= max_bars:
                continue
            frm = _num(c.get("from"), at if src.startswith(("bed.", "kit.")) else 0, -8)   # aligned bar for bar
            if source_bars is not None and src in source_bars and frm >= source_bars[src]:
                w.append(f"track {t.get('id', ti)}: clip at bar {at:g} starts past the end of {src}")
                continue
            clip = {"at": at, "src": src, "from": frm, "bars": min(bars, max_bars - at),
                    "gain_db": _num(c.get("gain_db"), 0, -60, 12),
                    "fx": [f for f in (c.get("fx") or []) if isinstance(f, dict)][:12]}
            if c.get("loop"):
                clip["loop"] = _num(c["loop"], 1, 1 / 16, 64)
            for k in ("fade_in", "fade_out", "tail"):
                if c.get(k) is not None:
                    clip[k] = _num(c[k], 0, 0, 64)
            clips.append(clip)
            end = max(end, clip["at"] + clip["bars"])
        if not clips:
            continue
        tracks.append({"id": str(t.get("id") or f"track{ti}"), "group": str(t.get("group") or "music"),
                       "role": str(t.get("role") or ""), "guide": bool(t.get("guide")), "mute": bool(t.get("mute")),
                       "gain_db": _num(t.get("gain_db"), 0, -60, 12), "pan": _num(t.get("pan"), 0, -1, 1),
                       "sidechain_db": _num(t["sidechain_db"], 0, 0, 18) if t.get("sidechain_db") is not None else None,
                       "pocket": t.get("pocket"),
                       "sends": {str(k): _num(v, 0, 0, 1) for k, v in (t.get("sends") or {}).items()} if isinstance(t.get("sends"), dict) else {},
                       "fx": [f for f in (t.get("fx") or []) if isinstance(f, dict)][:16], "clips": clips})
    ids = set()
    for t in tracks:  # unique ids
        while t["id"] in ids:
            t["id"] += "_"
        ids.add(t["id"])
    bars = int(np.ceil(_num(plan.get("bars"), end, 1, max_bars)))
    bars = max(bars, int(np.ceil(min(end, max_bars))))
    sections = []
    for s in plan.get("sections") or []:
        if isinstance(s, dict):
            st = _num(s.get("start"), 0, 0, bars)
            sections.append({"name": str(s.get("name") or "part"), "start": st,
                             "bars": _num(s.get("bars"), 1, 0.25, max(0.25, bars - st)),
                             "energy": _num(s.get("energy"), 0.5, 0, 1)})
    given = plan.get("buses") if isinstance(plan.get("buses"), dict) else {}
    buses = {**DEFAULT_BUSES, **{str(k): v for k, v in given.items() if isinstance(v, dict)}}
    mix = plan.get("mix") if isinstance(plan.get("mix"), dict) else {}
    out = {"bpm": bpm, "bars": bars, "sections": sorted(sections, key=lambda s: s["start"]), "tracks": tracks,
           "buses": buses, "mix": {"sidechain_db": _num(mix.get("sidechain_db"), 5, 0, 18),
                                   "pocket_db": _num(mix.get("pocket_db"), 3, 0, 9)},
           "master_fx": [f for f in (plan.get("master_fx") or []) if isinstance(f, dict)][:8]}
    for k in ("title", "notes", "key", "generate", "transpose", "caption", "bed", "style"):
        if plan.get(k) is not None:
            out[k] = plan[k]
    if not tracks:
        w.append("the plan has no playable tracks")
    return out, w


def _clip_audio(c: dict, sources: dict[str, Source], sr: int, bpm: float, w: list[str]) -> np.ndarray | None:
    bar = 240.0 / bpm * sr
    n = int(round(c["bars"] * bar))
    if c["src"].startswith("synth:"):
        x = fxrack.make_synth(c["src"][6:], c["bars"], sr, bpm, seed=int(c["at"] * 16))[:, :n]
        return np.pad(x, ((0, 0), (0, n - x.shape[1])))
    src = sources.get(c["src"])
    if src is None:
        w.append(f"missing source {c['src']}")
        return None
    take = int(round(min(c.get("loop") or c["bars"], c["bars"]) * bar)) if c.get("loop") else n
    s = int(round(src.offset * sr + c["from"] * bar))
    seg = np.zeros((2, take), dtype=np.float32)
    a, b = max(0, s), min(src.audio.shape[1], s + take)
    if b > a:
        seg[:, a - s:b - s] = src.audio[:, a:b]
    if c.get("loop") and take < n:
        k = min(256, take // 8)  # click-free loop point
        if k > 1:
            seg[:, :k] *= np.linspace(0, 1, k, dtype=np.float32)
            seg[:, -k:] *= np.linspace(1, 0, k, dtype=np.float32)
        seg = np.tile(seg, int(np.ceil(n / take)))[:, :n]
    return seg


def render(plan: dict, sources: dict[str, Source], sr: int, guide_only: bool = False) -> dict:
    """Render a cleaned plan. guide_only: the tracks marked guide plus every drum-machine clip, dry -
    what the instrumental is built around."""
    from .mixing import kick_envelope, sidechain, vocal_pocket
    w: list[str] = []
    bpm = plan["bpm"]
    bar = 240.0 / bpm * sr
    total = int(round(plan["bars"] * bar))
    tctx = {"bpm": bpm, "pos0": 0.0, "pos1": float(plan["bars"])}
    rendered: dict[str, np.ndarray] = {}
    for t in plan["tracks"]:
        if t["mute"]:
            continue
        clips = t["clips"]
        if guide_only and not t["guide"]:   # the instrumental is built around the lead AND the drum machine
            clips = [c for c in clips if c["src"].startswith("kit.")]
            if not clips:
                continue
        buf = np.zeros((2, total), dtype=np.float32)
        for c in clips:
            x = _clip_audio(c, sources, sr, bpm, w)
            if x is None:
                continue
            fx = c["fx"] if not guide_only else [f for f in c["fx"] if f.get("type") not in fxrack.TAIL_FX]
            tail = c.get("tail", 1.0 if any(f.get("type") in fxrack.TAIL_FX for f in fx) else 0.0)
            if tail:
                x = np.pad(x, ((0, 0), (0, int(tail * bar))))
            cctx = {"bpm": bpm, "unit": "frac"}
            x = apply_fx(x, sr, fx, cctx, w)
            x = fxrack.fx_fade(x, sr, {"in": c.get("fade_in", 0), "out": c.get("fade_out", 0)}, cctx)
            k = min(x.shape[1] // 2, int(0.003 * sr))   # declick every edit
            if k > 1:
                x[:, :k] *= np.linspace(0, 1, k, dtype=np.float32)
                x[:, -k:] *= np.linspace(1, 0, k, dtype=np.float32)
            s = int(round(c["at"] * bar))
            e = min(total, s + x.shape[1])
            if e > s:
                buf[:, s:e] += x[:, :e - s] * 10 ** (c["gain_db"] / 20)
        if not guide_only:
            buf = apply_fx(buf, sr, t["fx"], tctx, w)
            if t["pan"]:
                buf = fxrack.fx_pan(buf, sr, {"pos": t["pan"]}, tctx)
        rendered[t["id"]] = buf * 10 ** (t["gain_db"] / 20)
    by_id = {t["id"]: t for t in plan["tracks"]}
    if guide_only:
        mix = sum(rendered.values()) if rendered else np.zeros((2, total), dtype=np.float32)
        return {"mix": mix.astype(np.float32), "tracks": rendered, "warnings": w}

    # the kick drives the ducking; the vocal carves its pocket
    drums = [rendered[i] for i in rendered if by_id[i]["role"] == "drums"]
    vocal = [rendered[i] for i in rendered if by_id[i]["group"] == "vocal"]
    kick = kick_envelope(sum(drums), sr, bpm) if drums and float(np.max(np.abs(sum(drums)))) > 1e-4 else None
    vox = sum(vocal) if vocal else None
    for i, x in rendered.items():
        t = by_id[i]
        sc = t["sidechain_db"]
        if sc is None:
            sc = plan["mix"]["sidechain_db"] if (t["group"] == "music" and t["role"] != "drums") else 0.0
        if kick is not None and sc:
            x = sidechain(x, kick, sc)
        pocket = t["pocket"] if t["pocket"] is not None else (t["group"] == "music" and t["role"] not in ("drums", "bass"))
        if vox is not None and pocket and plan["mix"]["pocket_db"]:
            x = vocal_pocket(x, vox, sr, plan["mix"]["pocket_db"])
        rendered[i] = x.astype(np.float32)

    # sends: shared reverb and delay, like aux returns on a desk
    returns = np.zeros((2, total), dtype=np.float32)
    for name, params in plan["buses"].items():
        feed = sum((rendered[i] * by_id[i]["sends"][name] for i in rendered if by_id[i]["sends"].get(name)),
                   np.zeros((2, total), dtype=np.float32))
        if not np.any(feed):
            continue
        if name == "delay" or params.get("type") == "delay":
            wet = fxrack.fx_delay(feed, sr, {**params, "mix": 1.0}, tctx) - feed
        else:
            wet = fxrack.fx_reverb(feed, sr, {**params, "mix": 1.0, "add": False}, tctx)
        if kick is not None:
            wet = sidechain(wet, kick, 3.0)
        returns += wet.astype(np.float32)
    if np.any(returns):
        rendered["fx_returns"] = returns
    mix = sum(rendered.values()) if rendered else np.zeros((2, total), dtype=np.float32)
    mix = apply_fx(mix.astype(np.float32), sr, plan["master_fx"], tctx, w)
    peak = float(np.max(np.abs(mix))) or 1.0
    scale = 10 ** (-3 / 20) / peak
    groups: dict[str, np.ndarray] = {}
    for i, x in rendered.items():
        g = by_id[i]["group"] if i in by_id else "fx"
        g = "drums" if i in by_id and by_id[i]["role"] == "drums" else "bass" if i in by_id and by_id[i]["role"] == "bass" else g
        groups[g] = groups.get(g, 0) + x * scale
    return {"mix": (mix * scale).astype(np.float32), "tracks": {i: x * scale for i, x in rendered.items()},
            "groups": groups, "warnings": w}


def ensure_backbone(plan: dict, style: str = "house") -> list[str]:
    """A safety net under any plan: every high-energy section gets a full drum kit if the plan left it
    without one. A drop without drums is not a drop."""
    added = []
    drum_clips = [c for t in plan["tracks"] if t["role"] == "drums" and not t["mute"] for c in t["clips"]
                  if c["src"].startswith("kit.") and c["gain_db"] > -9
                  and not any(f.get("type") in ("filter", "eq", "telephone") for f in c["fx"])]
    clips = []
    for sec in plan.get("sections", []):
        if sec["energy"] < 0.7:
            continue
        a, b = sec["start"], sec["start"] + sec["bars"]
        covered = sum(max(0.0, min(b, c["at"] + c["bars"]) - max(a, c["at"])) for c in drum_clips
                      if c["src"] in ("kit.kick", "kit.full"))
        if covered < 0.7 * sec["bars"]:
            clips.append({"at": a, "src": "kit.full", "from": a, "bars": sec["bars"], "gain_db": 0.0, "fx": []})
            added.append(sec["name"])
    bass_clips = [c for t in plan["tracks"] if t["role"] == "bass" and not t["mute"] for c in t["clips"]
                  if c["src"] == "kit.bass" and c["gain_db"] > -9
                  and not any(f.get("type") in ("filter", "eq", "telephone") for f in c["fx"])]
    bclips = []
    for sec in plan.get("sections", []):
        if sec["energy"] < 0.7:
            continue
        a, b = sec["start"], sec["start"] + sec["bars"]
        covered = sum(max(0.0, min(b, c["at"] + c["bars"]) - max(a, c["at"])) for c in bass_clips)
        if covered < 0.7 * sec["bars"]:
            bclips.append({"at": a, "src": "kit.bass", "from": a, "bars": sec["bars"], "gain_db": 0.0, "fx": []})
            added.append(f"{sec['name']} (bass)")
    if bclips:
        plan["tracks"].append({"id": "bass_backbone", "group": "music", "role": "bass", "guide": False, "mute": False,
                               "gain_db": 0.0, "pan": 0.0, "sidechain_db": 6.0, "pocket": None, "sends": {}, "fx": [],
                               "clips": bclips})
    if clips:
        plan["tracks"].append({"id": "kit_backbone", "group": "music", "role": "drums", "guide": False, "mute": False,
                               "gain_db": 0.0, "pan": 0.0, "sidechain_db": None, "pocket": None, "sends": {}, "fx": [],
                               "clips": clips})
    return added


def ride(x: np.ndarray, sr: int, max_db: float = 6.0) -> np.ndarray:
    """Even out phrase-to-phrase level (slowly, at most +-max_db), leaving the dynamics inside each phrase."""
    hop = int(0.05 * sr)
    m = x.mean(0)
    frames = len(m) // hop
    if frames < 20:
        return x
    rms = np.sqrt(np.mean(m[:frames * hop].reshape(frames, hop) ** 2, axis=1))
    win = 16                                             # 0.8 s: a phrase, not a syllable
    env = np.sqrt(np.convolve(rms ** 2, np.ones(win) / win, mode="same"))
    active = env > 0.1 * (np.percentile(env, 95) + 1e-9)
    if active.sum() < 10:
        return x
    target = np.median(env[active])
    lim = 10 ** (max_db / 20)
    g = np.where(active, np.clip(target / (env + 1e-9), 1 / lim, lim), np.nan)
    idx = np.arange(frames)                              # hold the last gain through the gaps
    good = ~np.isnan(g)
    g = np.interp(idx, idx[good], g[good])
    g = np.convolve(g, np.ones(win) / win, mode="same")
    gain = np.interp(np.arange(x.shape[1]), idx * hop + hop / 2, g).astype(np.float32)
    return (x * gain).astype(np.float32)


def calibrate(sources: dict[str, Source], sr: int) -> dict[str, float]:
    """Put every source at a known level so gains in a plan mean something: the vocal and the new
    instrumental (as a whole) both land at -20 dB while they play; the original's stems likewise."""
    def active_rms(x):
        m = np.abs(x.mean(0))
        thr = 0.1 * (np.percentile(m, 99) + 1e-9)
        sel = x[:, m > thr]
        return float(np.sqrt(np.mean(sel ** 2))) if sel.size else 0.0
    if "vocal" in sources:  # ride the vocal like an engineer's hand on the fader: quiet lines up, loud lines down
        sources["vocal"] = Source(ride(sources["vocal"].audio, sr), sources["vocal"].offset)
    target, gains = 10 ** (-20 / 20), {}
    groups = {"vocal": ["vocal"], "bed": ["bed.drums", "bed.bass", "bed.other"],
              "orig": ["orig.drums", "orig.bass", "orig.other"], "chopriff": ["chopriff"]}
    for g, names in groups.items():
        have = [n for n in names if n in sources]
        if not have:
            continue
        n = min(sources[k].audio.shape[1] for k in have)
        level = active_rms(sum(sources[k].audio[:, :n] for k in have))
        if level < 1e-5:
            continue
        gain = float(np.clip(target / level, 0.1, 20.0)) * (0.7 if g == "chopriff" else 1.0)
        for k in have + (["bed.vox"] if g == "bed" and "bed.vox" in sources else []):
            sources[k] = Source((sources[k].audio * gain).astype(np.float32), sources[k].offset)
            gains[k] = round(20 * np.log10(gain), 1)
    return gains
