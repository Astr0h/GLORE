"""The producer brain: a song map and a few words from the user in, a full production plan out.

Claude (through Anthropic's API) makes the creative decisions a producer makes - what to keep, repeat,
chop, filter, throw into a delay, where the build and the drop go - and writes them as a session plan
(see session.py). GLORE then renders the plan, measures the result, and lets the producer revise once.

    plan, how = make_plan(songmap, "dark garage remix, keep the hook, huge second drop", sources, style="house")

Without an API key (or if the call fails) a rule-based plan is used instead, so a song always comes out.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from . import fxrack, session

MODELS = {"standard": "claude-sonnet-5-5", "deep": "claude-fable-5-2"}
API = "https://api.anthropic.com/v1/messages"

SYSTEM = """You are GLORE's producer: a studio-level electronic music producer and remixer. You are given a map of a \
vocal (or of a full song split into stems) and a short brief. You decide the whole production and write it as a session \
plan in JSON, which a multitrack engine renders exactly as written. Think like a producer finishing a record: clear \
sections, contrast between them, tension before every release, the hook used with intent, space where the vocal needs \
it, ear candy in the gaps, nothing random.

# How the session works
Time is in bars (4 beats; fractions allowed: 0.25 = one beat, 0.0625 = a 16th). The output timeline starts at bar 0.
A track holds clips. A clip plays `bars` of a source, starting at source bar `from`, placed at output bar `at`.
`loop: n` repeats the first n bars of that source region to fill the clip. Clip `fx` process that clip alone; track \
`fx` process the whole track. Automatable parameters take a number or points [[pos, value], ...]: for clip fx pos runs \
0..1 across the clip, for track fx and master_fx pos is the output bar.

# Sources you can place
{sources}
- "synth:riser", "synth:snare_roll" (builds: end them exactly where the next section starts), "synth:noise_sweep", \
"synth:downlifter", "synth:impact", "synth:sub_drop" (start them on a downbeat): transition sounds of any length.
The "bed.*" stems are the new instrumental (bass, chords and synths - no drums: the drums are your kit), generated \
to fit your arrangement: the model hears the lead vocal (tracks with "guide": true) and your kit clips exactly as you \
arranged them, and plays bass and chords around them over the whole output timeline, aligned bar for bar (bed bar N = \
output bar N; normally at == from). Where your kit drops out, the bed will tend to thin out too. You shape the bed with \
"bed": {{"caption": "..."}}: comma-separated tags for genre, mood, drums, bass, instruments, texture. It is \
generated once, with one sound all the way through - the contrast between sections is YOUR job: mute stems, filter \
them, bring the drums in late, strip to bass and vocal, loop a bar. Changing the lead's or the kit's clips in a \
revision rebuilds the bed.

The "kit.*" sources are a drum machine locked to the grid over the whole timeline (kit.kick, kit.clap, kit.hats, \
kit.perc, or kit.full for all four), in the pack's style. They are the only drums: build the whole groove from them, part by part. Put kit tracks in group "music" with \
role "drums". "kit.bass" is the bassline (role "bass"): it follows the song's harmony and ducks to the kick. The \
generated bed.bass is usually faint - use kit.bass as THE bass and bed.bass, if at all, as a layer. "kit.keys" (piano \
stabs) and "kit.pad" (sustained pad) play the song's own chords: the generated bed.other is usually quiet, so the \
chords carry the harmony - pad under intros, verses and breakdowns, keys (with or without the pad) in the drops.
All sources are level-matched (vocal, bed and kit sit together at gain 0), so gain_db is a real mix decision: 0 is \
"at full level", -6 clearly back, -12 background.

# What makes it sound like a record in this style
{style_guide}
- Every section with energy 0.7 or more has the full kit (kick, clap, hats) and unfiltered kit.bass, start to finish. \
A drop is the fullest, loudest part of the song: all drums, bass, music wide open, and the hook or chops on top.
- Quieter sections keep a pulse (hats or a filtered kick) unless it is a deliberate break of 1-4 bars. Do not \
high-pass or bury the drums for whole verses; thin them instead (drop the clap, or the kick, not everything).
- Transition sounds are seasoning: at most one riser or snare_roll into a section and one impact on its downbeat, \
at their default level or lower. Never leave them as the loudest thing.
- Do not layer pitch-shifted copies of the lead for whole sections (octave_up / pitch on long clips sounds synthetic). \
Pitch and vocoder effects are for short moments. Vocoder notes must come from the map's sung_notes pitch classes.
- "bed": {{"caption": ...}} is required: describe the instrumental richly (genre, drums, bass, chords, mood).

# Rules that keep it musical
- Vocal clips of a bar or more must keep (at - from) a whole number of bars, so words stay on the beat. Cut at phrase \
boundaries from the map (phrases never start exactly on the cut: leave the pickup with its phrase).
- Never play two different lead-vocal regions on top of each other unless one is clearly an effect (filtered, pitched, \
low in level).
- The song must read as sections with different energy. Give every section in "sections" with an energy 0..1 and make \
the arrangement match it: low energy = fewer stems, filtered, drier; high = everything, wide, loud.
- Before a drop or chorus: build (riser / snare_roll / high-pass sweep on the music / drums drop out for the last bar \
/ a vocal throw or stutter into it). On the downbeat: impact, full drums and bass.
- Chops: use "chopriff" (the song's own melody as a sampler riff, a 4-bar loop) or cut your own from "vocal" at \
held_notes or the hook (short clips with pitch, reverse, stutter, gate, delay). Chops belong where the lead is not \
singing, and should sound like part of the record, not on top of it. Don't spam.
- Use effects with intent and automation: filter sweeps, delay/reverb throws on last words (automate `send`), \
saturation, phaser, wobble, tremolo/gate, vocoder (give `per_bar` chord notes in the song's key), telephone, tape_stop, \
reverse_reverb into a phrase, octave layers. Subtle on the lead, bolder on doubles, chops and transitions.
- Length: usually 2.5 to 4 minutes at the session tempo; never over 5.
- You may change tempo ("bpm", within about 8% of the original keeps the voice natural) and key ("transpose" in \
semitones, small).

# Effects (type and parameters; (a) = automatable)
gain db(a) | fade in,out (beats) | pan pos(a) | width amount | eq hp,lp,low_db,high_db,bells[[hz,db,q]] |
filter mode(lp|hp|bp),cutoff(a),resonance,drive | telephone | wobble rate(beats),min_hz,max_hz | saturate drive_db,mix |
bitcrush bits | compress threshold_db,ratio | reverb size,mix(a),send(a) | delay time(beats),feedback,mix(a),send(a),ping_pong |
reverse_reverb size,mix | chorus | phaser rate_hz,depth | tremolo rate(beats),depth,shape(sine|square|saw) |
gate pattern("x.x.x..x........" 16ths) | autopan rate(beats),depth | pump depth_db,rate(beats) | pitch semis,formants |
vocoder notes[midi]|per_bar[[midi]],mix | reverse | stutter slice(beats),length(beats) | tape_stop beats |
octave_up mix | octave_down mix

# Output: ONE JSON object, nothing else - no commentary before or after
Write it compactly: no indentation, leave out any field that is at its default (from 0 - or from = at for "bed.*" clips - gain_db 0, empty fx, fades 0). One clip per continuous region: a stem that plays unchanged for 32 bars is one clip, not eight. Spend clips where the ideas are. A full song is typically 60-200 clips.
{{"title": "short name", "notes": "2-4 plain sentences for the artist: what you did and why",
 "bpm": 126, "transpose": 0, "bars": 96,
 "bed": {{"caption": "tags for the instrumental"}},
 "sections": [{{"name": "intro", "start": 0, "bars": 8, "energy": 0.25}}, ...],
 "tracks": [
  {{"id": "lead", "group": "vocal", "guide": true, "gain_db": 0, "sends": {{"reverb": 0.15, "delay": 0.05}},
    "fx": [...], "clips": [{{"at": 8, "src": "vocal", "from": 0, "bars": 16, "fx": [...], "gain_db": 0,
                              "fade_in": 0, "fade_out": 0}}]}},
  {{"id": "kick", "group": "music", "role": "drums", "clips": [{{"at": 4, "src": "kit.kick", "bars": 92}}]}},
  {{"id": "clap", "group": "music", "role": "drums", "clips": [...]}}, {{"id": "hats", "group": "music", "role": "drums", "clips": [...]}},
  {{"id": "bass", "group": "music", "role": "bass", "clips": [...]}},
  {{"id": "music", "group": "music", "clips": [...]}},
  {{"id": "chops", "group": "chops", "sends": {{"delay": 0.2}}, "clips": [...]}},
  {{"id": "fx", "group": "fx", "clips": [...]}}],
 "buses": {{"reverb": {{"size": 0.75}}, "delay": {{"time": 0.75, "feedback": 0.45}}}},
 "mix": {{"sidechain_db": 5, "pocket_db": 3}},
 "master_fx": []}}
group is one of vocal, music, chops, fx. role "drums" / "bass" on the tracks carrying those (the kick ducks the rest). \
Tracks in group "music" duck to the kick and make room for the vocal automatically."""

REVISE = """Here is the plan you wrote, and measurements of how it rendered. "problems" lists what the engine \
found wrong: fix every one. Then fix other real problems only: a section \
whose loudness doesn't match its energy, a lead that is buried (vocal well below music) or far too loud, dead air, \
a muddy low end (low share too high) or a thin one, vocal clips flagged off the grid, things the engine skipped. \
Changing the lead vocal's clips forces the whole instrumental to be rebuilt, so only do that if it is the problem.
If it is good, reply {"ok": true}. Otherwise reply with the full corrected plan as one JSON object."""


def describe_sources(source_bars: dict[str, float]) -> str:
    notes = {"vocal": "the lead vocal, dry, on its own bar grid (see the map)",
             "chopriff": "a ready-made vocal-chop riff from the song's own melody, loopable",
             "bed.bass": "new instrumental: bass",
             "bed.other": "new instrumental: chords, synths, keys", "bed.vox": "the model's own vocal-like textures (use low, in gaps)",
             "kit.keys": "chord machine: warm electric-piano stabs on the song's own chords, in the style's rhythm",
             "kit.pad": "chord machine: a sustained, filtered pad on the song's own chords",
             "kit.bass": "bass machine: a sub bassline on the grid that follows the song's chord roots under your arrangement",
             "kit.full": "drum machine: kick, clap, hats and perc together, on the grid",
             "kit.kick": "drum machine: kick", "kit.clap": "drum machine: clap / snare",
             "kit.hats": "drum machine: hats", "kit.perc": "drum machine: shaker / percussion",
             "orig.drums": "the ORIGINAL song's drums (same bar grid as vocal)", "orig.bass": "the original's bass",
             "orig.other": "the original's instruments"}
    return "\n".join(f'- "{k}": {v:g} bars. {notes.get(k, "")}'.rstrip() for k, v in source_bars.items())


STYLE_GUIDE = {
    "house": "- House: four-on-the-floor kick on every beat, clap on 2 and 4, open hats on the offbeats, a rolling bass "
             "that ducks to the kick, chords or piano stabs, 120-130 BPM feel. Long builds, clear drops, DJ-friendly "
             "intro and outro with drums.",
    "rnb": "- R&B / rap: half-time feel, heavy kick and 808 bass, crisp snare on 3, rolling hats, lots of space for the "
           "vocal, lush chords. No four-on-the-floor.",
    "gospel": "- Gospel / soul: live-feeling drums, warm bass, organ and piano carrying the chords, the vocal and "
              "stacked harmonies in front, lifts into each chorus rather than drops.",
}


def system_prompt(source_bars: dict[str, float], style: str) -> str:
    return SYSTEM.format(sources=describe_sources(source_bars), style_guide=STYLE_GUIDE.get(style, STYLE_GUIDE["house"]))


def _extract_json(text: str) -> dict:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON in the reply")
    return json.loads(text[a:b + 1])


def read_stream(lines) -> tuple[str, str | None, dict]:
    """Collect the text of a streamed Messages reply. Returns (text, stop_reason, usage)."""
    text, stop, usage = [], None, {}
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
            text.append(ev["delta"].get("text", ""))
        elif kind == "message_start":
            usage.update(ev.get("message", {}).get("usage") or {})
        elif kind == "message_delta":
            stop = ev.get("delta", {}).get("stop_reason") or stop
            usage.update(ev.get("usage") or {})
        elif kind == "error":
            raise RuntimeError(f"Anthropic API: {ev.get('error', {}).get('message', ev)}")
    return "".join(text), stop, usage


def ask(system: str, user: str, model: str, api_key: str, max_tokens: int = 48000, timeout: float = 900) -> tuple[dict, dict]:
    """One streamed call (plans are long; streaming keeps the connection alive)."""
    body = {"model": model, "max_tokens": max_tokens, "system": system, "stream": True,
            "messages": [{"role": "user", "content": user}]}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text, stop, usage = read_stream(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Anthropic API {e.code}: {e.read()[:300].decode(errors='replace')}") from e
    if stop == "max_tokens":
        raise RuntimeError(f"the plan was cut off (over {max_tokens} tokens)")
    return _extract_json(text), usage


def make_plan(songmap: dict, brief: str, source_bars: dict[str, float], style: str = "house",
              model: str = "standard", api_key: str | None = None, seed: int = 0) -> tuple[dict, dict]:
    """Returns (cleaned plan, how) where how = {"by": "claude"|"rules", "model", "usage", "warnings", "error"}."""
    api_key = api_key if api_key is not None else os.environ.get("GLORE_ANTHROPIC_KEY", "")
    how: dict = {"by": "rules", "model": None, "warnings": []}
    plan = None
    if api_key:
        mid = MODELS.get(model, model)
        user = (f"Brief from the artist: {brief or 'make it sound like a finished record in this style'}\n"
                f"Style pack: {style}\n\nSong map (bars on the vocal's grid):\n{json.dumps(songmap)}\n")
        try:
            raw, usage = ask(system_prompt(source_bars, style), user, mid, api_key)
            plan, w = session.clean_plan(raw, source_bars)
            if not plan["tracks"] or not any(t["guide"] for t in plan["tracks"]):
                raise RuntimeError("the plan had no lead vocal track to build around")
            added = session.ensure_backbone(plan, style)
            w += [f"added a full drum kit to {n} (it had none)" for n in added]
            how = {"by": "claude", "model": mid, "usage": usage, "warnings": w}
        except Exception as e:
            how["error"] = str(e)
            plan = None
    if plan is None:
        plan, w = session.clean_plan(default_plan(songmap, brief, style, source_bars, seed), source_bars)
        how["warnings"] = w
    plan["style"] = style
    if not (plan.get("bed") or {}).get("caption"):
        plan["bed"] = {"caption": ", ".join(x for x in (brief.strip(), CAPTIONS.get(style, CAPTIONS["house"])) if x)}
    return plan, how


def revise(plan: dict, metrics: dict, source_bars: dict[str, float], model: str = "standard",
           api_key: str | None = None, style: str = "house") -> tuple[dict | None, dict]:
    """One critique pass. Returns (new cleaned plan or None if it's fine / unavailable, how)."""
    api_key = api_key if api_key is not None else os.environ.get("GLORE_ANTHROPIC_KEY", "")
    if not api_key:
        return None, {"by": "none"}
    mid = MODELS.get(model, model)
    user = f"{REVISE}\n\nPlan:\n{json.dumps(plan)}\n\nMeasurements:\n{json.dumps(metrics)}"
    try:
        raw, usage = ask(system_prompt(source_bars, style), user, mid, api_key)
    except Exception as e:
        return None, {"by": "claude", "model": mid, "error": str(e)}
    if raw.get("ok") is True or not raw.get("tracks"):
        return None, {"by": "claude", "model": mid, "usage": usage, "verdict": "ok"}
    new, w = session.clean_plan(raw, source_bars)
    if not new["tracks"] or not any(t["guide"] for t in new["tracks"]):
        return None, {"by": "claude", "model": mid, "usage": usage, "error": "revised plan unusable"}
    session.ensure_backbone(new, style)
    new["style"] = style
    new.setdefault("bed", plan.get("bed"))
    return new, {"by": "claude", "model": mid, "usage": usage, "verdict": "revised", "warnings": w}


def guide_signature(plan: dict) -> str:
    """Changes only when what the instrumental is built around changes (so a revision can reuse the bed)."""
    g = [[(c["at"], c["src"], c["from"], c["bars"], c.get("loop"), c["gain_db"] if not t["guide"] else 0,
           json.dumps(c["fx"], sort_keys=True) if not t["guide"] else "") for c in t["clips"]
          if t["guide"] or c["src"].startswith("kit.")] for t in plan["tracks"] if not t["mute"]]
    g = [x for x in g if x]
    return json.dumps([g, plan["bpm"], plan.get("transpose", 0), (plan.get("bed") or {}).get("caption", "")], sort_keys=True)


# ---------------------------------------------------------------- the rule-based producer (fallback)

CAPTIONS = {"house": "house, four on the floor, punchy kick, crisp hats, deep rolling sub bass, warm piano chords, atmospheric pads",
            "rnb": "r&b, laid back drums, deep 808 bass, lush electric piano, smooth pads",
            "gospel": "gospel soul, live drums, warm bass, organ, piano, uplifting"}


def default_plan(m: dict, brief: str, style: str, source_bars: dict[str, float], seed: int = 0) -> dict:
    """A solid, conventional arrangement from the song map: intro, the song with builds, a drop after
    each of the first two choruses, outro. Used when the AI producer isn't available."""
    drop = 8 if style == "house" else 4
    has = lambda s: s in source_bars
    secs, tracks = [], {k: [] for k in ("lead", "kit", "bass", "keys", "music", "chops", "fx", "vox")}
    pos = 0.0

    def bed(at, bars, drums=True, bass=True, music=True, music_fx=None, drum_fx=None, kit="kit.full"):
        if drums:
            tracks["kit"].append({"at": at, "src": kit, "from": at, "bars": bars, "fx": drum_fx or []})
        if bass:
            tracks["bass"].append({"at": at, "src": "kit.bass", "from": at, "bars": bars})
        if music:
            tracks["keys"].append({"at": at, "src": "kit.keys" if drums and bass else "kit.pad", "from": at, "bars": bars})
        if music:
            tracks["music"].append({"at": at, "src": "bed.other", "from": at, "bars": bars, "fx": music_fx or []})
        if has("bed.vox"):
            tracks["vox"].append({"at": at, "src": "bed.vox", "from": at, "bars": bars})

    def build_into(at, bars=2):
        tracks["fx"].append({"at": max(0, at - bars), "src": "synth:riser", "bars": min(bars, at), "gain_db": -8})
        tracks["fx"].append({"at": at, "src": "synth:impact", "bars": 2, "gain_db": -7})

    hook = m.get("hook")
    intro = 8
    secs.append({"name": "intro", "start": 0, "bars": intro, "energy": 0.3})
    bed(0, intro, drums=False, bass=False, music_fx=[{"type": "filter", "mode": "lp", "cutoff": [[0, 350], [1, 9000]], "resonance": 0.25}])
    tracks["kit"].append({"at": 0, "src": "kit.hats", "from": 0, "bars": intro})
    tracks["kit"].append({"at": intro / 2, "src": "kit.kick", "from": intro / 2, "bars": intro / 2, "gain_db": -6,
                          "fx": [{"type": "filter", "mode": "lp", "cutoff": [[0, 300], [1, 6000]]}]})
    if hook:
        tracks["chops"].append({"at": intro - 1, "src": "vocal", "from": hook["bar"], "bars": min(1.0, hook["bars"]),
                                "fx": [{"type": "filter", "mode": "hp", "cutoff": 500}, {"type": "delay", "time": 0.75, "mix": 0.4}],
                                "gain_db": -4})
    pos = float(intro)
    drops = 0
    sections = m.get("sections") or [{"kind": "verse", "start": 0, "bars": m["bars"], "energy": 0.6}]
    for i, s in enumerate(sections):
        st, n, kind = float(s["start"]), float(s["bars"]), s.get("kind", "verse")
        energy = {"chorus": 0.85, "verse": 0.55, "instrumental": 0.6}.get(kind, 0.6)
        secs.append({"name": s.get("name", kind), "start": pos, "bars": n, "energy": energy})
        if kind != "instrumental":
            tracks["lead"].append({"at": pos, "src": "vocal", "from": st, "bars": n})
        if kind == "chorus":
            build_into(pos)
            bed(pos, n)
        elif kind == "verse":
            bed(pos, n, music_fx=[{"type": "eq", "lp": 9000}], kit="kit.kick")
            tracks["kit"].append({"at": pos, "src": "kit.hats", "from": pos, "bars": n})
        else:
            bed(pos, n)
            if has("chopriff"):
                tracks["chops"].append({"at": pos, "src": "chopriff", "from": 0, "bars": n, "loop": 4})
        pos += n
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        if kind == "chorus" and drops < 2 and (nxt is None or nxt.get("kind") != "instrumental"):
            secs.append({"name": f"drop {drops + 1}", "start": pos, "bars": drop, "energy": 1.0})
            build_into(pos, 1)
            if hook:  # stutter the hook into the drop
                tracks["chops"].append({"at": pos - 0.25, "src": "vocal", "from": hook["bar"], "bars": 0.25,
                                        "fx": [{"type": "stutter", "slice": 0.0625}], "gain_db": -2})
            bed(pos, drop)
            if has("chopriff"):
                tracks["chops"].append({"at": pos, "src": "chopriff", "from": 0, "bars": drop, "loop": 4})
            pos += drop
            drops += 1
    outro = 8
    secs.append({"name": "outro", "start": pos, "bars": outro, "energy": 0.3})
    bed(pos, outro, bass=False, music_fx=[{"type": "filter", "mode": "lp", "cutoff": [[0, 9000], [1, 300]]}],
        drum_fx=[{"type": "gain", "db": [[0, 0], [1, -18]]}])
    tracks["fx"].append({"at": pos, "src": "synth:downlifter", "bars": 2, "gain_db": -9})
    pos += outro
    meta = {"lead": {"group": "vocal", "guide": True, "sends": {"reverb": 0.14, "delay": 0.05},
                     "fx": [{"type": "eq", "hp": 90}, {"type": "compress", "threshold_db": -20, "ratio": 3}]},
            "kit": {"group": "music", "role": "drums"}, "keys": {"group": "music"}, "bass": {"group": "music", "role": "bass"},
            "music": {"group": "music"}, "chops": {"group": "chops", "sends": {"delay": 0.2, "reverb": 0.15}, "gain_db": -5},
            "fx": {"group": "fx", "sends": {"reverb": 0.2}},
            "vox": {"group": "music", "gain_db": -11, "fx": [{"type": "eq", "hp": 250, "lp": 8500}]}}
    cap = ", ".join(x for x in (brief.strip(), CAPTIONS.get(style, CAPTIONS["house"])) if x)
    return {"title": "Rule-based arrangement", "bpm": m["bpm"], "bars": pos, "bed": {"caption": cap}, "sections": secs,
            "notes": "Arranged by GLORE's built-in rules (the AI producer wasn't available): intro, the song with "
                     "builds into each chorus, a drop after the first two choruses, outro.",
            "tracks": [{"id": k, **meta[k], "clips": v} for k, v in tracks.items() if v]}


# ---------------------------------------------------------------- measuring the render (for the critique)

def _db(x) -> float:
    return round(float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)), 1)


def measure(out: dict, plan: dict, sr: int) -> dict:
    """What the producer needs to judge its own mix without hearing it."""
    mix, groups = out["mix"], out.get("groups", {})
    bar = 240.0 / plan["bpm"] * sr
    vocal = groups.get("vocal")
    music = sum((v for k, v in groups.items() if k in ("music", "drums", "bass")), np.zeros_like(mix))
    spec = np.abs(np.fft.rfft(mix.mean(0)[: min(mix.shape[1], sr * 240)]))
    freqs = np.fft.rfftfreq(min(mix.shape[1], sr * 240), 1 / sr)
    tot = float(np.sum(spec ** 2)) + 1e-9
    share = lambda lo, hi: round(float(np.sum(spec[(freqs >= lo) & (freqs < hi)] ** 2)) / tot, 3)
    secs = []
    for s in plan.get("sections", []):
        a, b = int(s["start"] * bar), int((s["start"] + s["bars"]) * bar)
        if b <= a:
            continue
        row = {"name": s["name"], "planned_energy": s["energy"], "loudness_db": _db(mix[:, a:b])}
        if vocal is not None and float(np.max(np.abs(vocal[:, a:b]))) > 1e-4:
            row["vocal_minus_music_db"] = round(_db(vocal[:, a:b]) - _db(music[:, a:b]), 1)
        secs.append(row)
    planned = [s["planned_energy"] for s in secs]
    loud = [s["loudness_db"] for s in secs]
    corr = round(float(np.corrcoef(planned, loud)[0, 1]), 2) if len(secs) > 2 and np.std(planned) > 0 and np.std(loud) > 0 else None
    hop = int(bar)
    quiet = [round(i * hop / bar, 1) for i in range(int(mix.shape[1] / hop)) if _db(mix[:, i * hop:(i + 1) * hop]) < -50]
    offgrid = [f"{t['id']} clip at bar {c['at']:g}" for t in plan["tracks"] if t["group"] == "vocal" and t["guide"]
               for c in t["clips"] if c["bars"] >= 1 and abs((c["at"] - c["from"]) - round(c["at"] - c["from"])) > 0.01]
    drums, bassg = groups.get("drums"), groups.get("bass")
    problems = []
    real = [x for x in plan.get("sections", []) if int((x["start"] + x["bars"]) * bar) > int(x["start"] * bar)]
    for s_, row in zip(real, secs):
        a, b = int(s_["start"] * bar), int((s_["start"] + s_["bars"]) * bar)
        if drums is not None:
            row["drums_minus_mix_db"] = round(_db(drums[:, a:b]) - _db(mix[:, a:b]), 1)
        if s_["energy"] >= 0.7 and (drums is None or row.get("drums_minus_mix_db", -99) < -12):
            problems.append(f"{row['name']}: high energy but the drums are weak or missing")
        if bassg is not None:
            row["bass_minus_mix_db"] = round(_db(bassg[:, a:b]) - _db(mix[:, a:b]), 1)
        if s_["energy"] >= 0.7 and (bassg is None or row.get("bass_minus_mix_db", -99) < -16):
            problems.append(f"{row['name']}: high energy but the bass is weak or missing (use kit.bass)")
        if row.get("vocal_minus_music_db") is not None and row["vocal_minus_music_db"] < -10:
            problems.append(f"{row['name']}: the lead vocal is buried")
    hi = [r["loudness_db"] for r in secs if r["planned_energy"] >= 0.8]
    lo = [r["loudness_db"] for r in secs if r["planned_energy"] <= 0.45]
    if hi and lo and max(hi) - float(np.mean(lo)) < 3:
        problems.append("the drops are not clearly louder and fuller than the quiet sections (under 3 dB apart)")
    if "fx" in groups and drums is not None and _db(groups["fx"]) > _db(drums) - 3:
        problems.append("transition sounds (fx) are as loud as the drums: turn them down or use fewer")
    if "fx" in groups and drums is None:
        problems.append("there are transition sounds but no drum tracks (role drums) at all")
    problems += [f"vocal off the grid: {x}" for x in offgrid]
    return {"problems": problems, "sections": secs, "energy_follows_plan": corr, "silent_bars": quiet[:20],
            "spectrum_share": {"low_under_150hz": share(0, 150), "mid_150_4k": share(150, 4000), "high_over_4k": share(4000, 20000)},
            "stem_levels_db": {k: _db(v) for k, v in groups.items()}, "vocal_clips_off_grid": offgrid,
            "length_seconds": round(mix.shape[1] / sr, 1), "engine_warnings": (out.get("warnings") or [])[:20]}
