"""GLORE tools on the cloud GPU - run inside ACE-Step's venv, with GLORE's engine code synced
next to this file (glore/engine/glore_engine on the volume).

    python glore_tools.py <op> job.json

job: {"root": "/workspace", "item": {library item}, "params": {...}, "work": dir}
ops:
  stems    4 stems (vocals, drums, bass, other), htdemucs_ft        -> new library items
  clean    raw-vocal chain: backing / echo / reverb / noise         -> new library item
  analyse  BPM, key, Camelot, kind guess                            -> updates the item
  lock     stretch to a BPM, optional pitch shift (formants kept)   -> new library item
  chops    vocal chops in a pack style (house / rnb / gospel)       -> new library item
  mixsong  the Song mix: take's own vocal out, real vocal + FX + chops on top, sidechain,
           pocket EQ, limiter; exports MP3 320 + WAV 24-bit + stems ZIP
Prints GLORE_RESULT <json> / GLORE_ERROR <json>.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

SR = 44100


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def to_wav(src: str | Path, dst: str | Path) -> Path:
    ffmpeg("-i", str(src), "-ac", "2", "-ar", str(SR), str(dst))
    return Path(dst)


def duration(path: Path) -> float:
    import soundfile as sf
    return round(sf.info(str(path)).duration, 2)


# ---------------------------------------------------------------- library on the mounted volume

def lib_add(root: Path, wav: Path, name: str, kind: str, origin: str, parent: dict | None = None, **extra) -> dict:
    item_id = uuid.uuid4().hex[:10]
    d = root / "library" / item_id
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in " ._-()&',+").strip()[:80] or "audio"
    flac = d / f"{safe}.flac"
    ffmpeg("-i", str(wav), "-ac", "2", "-ar", str(SR), "-sample_fmt", "s16", str(flac))
    item = {"id": item_id, "name": safe, "kind": kind, "file": f"library/{item_id}/{flac.name}",
            "duration": duration(flac), "origin": origin, "created": time.time(), **extra}
    if parent:
        item["parent"] = parent["id"]
        for k in ("bpm", "key", "camelot"):
            if parent.get(k) and k not in item:
                item[k] = parent[k]
    (d / "item.json").write_text(json.dumps(item), encoding="utf-8")
    return item


def lib_update(root: Path, item: dict) -> dict:
    (root / "library" / item["id"] / "item.json").write_text(json.dumps(item), encoding="utf-8")
    return item


def load(path: Path):
    from glore_engine.audio_io import load_stereo
    return load_stereo(path, SR)


def save(path: Path, x, subtype: str = "FLOAT") -> Path:
    import soundfile as sf
    sf.write(str(path), x.T, SR, subtype=subtype)
    return path


# ---------------------------------------------------------------- ops

def op_stems(job, root, work, item, p):
    from glore_engine.separation import separate
    src = to_wav(root / item["file"], work / "in.wav")
    stems = separate(src, work / "stems", model_name=p.get("model", "htdemucs_ft"))
    made = []
    for name in ("vocals", "drums", "bass", "other"):
        if name in stems:
            made.append(lib_add(root, stems[name], f"{item['name']} ({name})", "vocal" if name == "vocals" else "stem",
                                "stems", parent=item, stem=name))
    return {"items": made}


def op_clean(job, root, work, item, p):
    os.environ.setdefault("GLORE_HOME", str(root / "models"))  # separator models stay on the volume
    from glore_engine.cleanup import raw_vocal
    stages = p.get("stages") or ["backing", "echo", "reverb", "noise"]
    src = to_wav(root / item["file"], work / "in.wav")
    out = raw_vocal(src, work / "clean.wav", stages)
    return {"items": [lib_add(root, out.path, f"{item['name']} (clean)", "vocal", "clean", parent=item,
                              stages=out.stages)]}


def op_analyse(job, root, work, item, p):
    from glore_engine.analysis import analyse
    src = to_wav(root / item["file"], work / "in.wav")
    kind = {"vocal": "vocal", "beat": "beat", "song": "full_song"}.get(item.get("kind"))
    a = analyse(str(src), kind)
    item.update({"bpm": a.bpm, "key": a.key, "camelot": a.camelot, "bpm_candidates": a.bpm_candidates,
                 "kind_guess": a.kind_guess, "duration": a.duration,
                 "phrases": [vars(ph) if hasattr(ph, "__dict__") else ph for ph in a.phrases][:400]})
    return {"updated": lib_update(root, item)}


def op_lock(job, root, work, item, p):
    from glore_engine.vocal_lock import lock_vocal
    src_bpm = float(p.get("source_bpm") or item.get("bpm") or 0)
    if not src_bpm:
        raise ValueError("Analyse it first (or give its BPM) so GLORE knows how far to stretch")
    target, pitch = float(p["target_bpm"]), float(p.get("pitch", 0))
    src = to_wav(root / item["file"], work / "in.wav")
    res = lock_vocal(src, work / "locked.wav", src_bpm, target, pitch)
    tag = f"{target:g} BPM" + (f", {pitch:+g} st" if pitch else "")
    new = lib_add(root, Path(res.path), f"{item['name']} ({tag})", item.get("kind", "vocal"), "lock", parent=item)
    new["bpm"] = target
    if pitch and item.get("key"):
        from glore_engine.analysis import key_name, parse_key, camelot
        tonic, mode = parse_key(item["key"])
        t2 = (tonic + int(round(pitch))) % 12
        new["key"], new["camelot"] = key_name(t2, mode), camelot(t2, mode)
    return {"items": [lib_update(root, new)]}


def op_chops(job, root, work, item, p):
    from glore_engine.chops import make_chops
    bpm = float(p.get("bpm") or item.get("bpm") or 0)
    if not bpm:
        raise ValueError("Chops need a BPM: analyse the vocal first or type one in")
    voc = load(to_wav(root / item["file"], work / "in.wav"))
    style, seed = p.get("style", "house"), int(p.get("seed", 0))
    layer, info = make_chops(voc, SR, bpm, p.get("key") or item.get("key"), style=style, seed=seed,
                             density=float(p.get("density", 1)), length=float(p.get("length", 1)),
                             pitch_spread=float(p.get("pitch_spread", 1)), offset=float(p.get("offset", 0)),
                             duck_under_vocal=float(p.get("duck", 0)))
    out = save(work / "chops.wav", layer)
    return {"items": [lib_add(root, out, f"{item['name']} chops ({style} {seed})", "chops", "chops", parent=item,
                              bpm=bpm, style=style, seed=seed, hits=info.hits)]}


def drum_beats(drums, bpm: float):
    """Beat times of the generated drums (so chops sit on the real kicks, even if the tempo breathes)."""
    import librosa
    import numpy as np
    y = librosa.resample(drums.mean(0), orig_sr=SR, target_sr=22050)
    _, beats = librosa.beat.beat_track(y=y, sr=22050, start_bpm=bpm, tightness=400, units="time")
    return np.asarray(beats, dtype=float)


def beat_offset(beat, bpm: float) -> float:
    """Where the beat grid starts (first beat, folded into one beat length)."""
    import librosa
    y = librosa.resample(beat.mean(0), orig_sr=SR, target_sr=22050)
    _, beats = librosa.beat.beat_track(y=y, sr=22050, start_bpm=bpm, units="time")
    return float(beats[0] % (60.0 / bpm)) if len(beats) else 0.0


def op_arrange(job, root, work, item, p):
    """p: vocal, music?, bpm, style. Plans intro / drops / outro around the vocal and opens that space
    in its timeline (and the original music's, so the two stay aligned)."""
    import numpy as np
    from glore_engine import chops, structure
    bpm = float(p["bpm"])
    dry = load(to_wav(p["vocal"], work / "vocal.wav"))
    music = load(to_wav(p["music"], work / "music.wav")) if p.get("music") else None
    grid_src = music if music is not None else dry
    offset = beat_offset(grid_src, bpm)
    bar_s = 4 * 60.0 / bpm
    n_bars = int(np.ceil((dry.shape[1] / SR - offset) / bar_s))
    act = chops.vocal_activity(dry.mean(0), SR, bpm, offset, n_bars)
    struct = chops._mono(dry if music is None else dry[:, :music.shape[1]] + music[:, :dry.shape[1]], SR)
    regions, _ = chops.find_chorus(struct, bpm, offset, n_bars, act)
    plan = structure.plan_structure(dry, SR, bpm, offset, regions, p.get("style", "house"))
    out = {"vocal": str(save(work / "vocal_arranged.wav", structure.apply_plan(dry, plan))),
           "plan": plan.to_dict(), "hint": structure.caption_hint(plan), "chorus": regions, "offset": offset}
    if music is not None:
        out["music"] = str(save(work / "music_arranged.wav", structure.apply_plan(music, plan)))
    out["duration"] = round(dry.shape[1] / SR + sum(i.bars for i in plan.inserts) * bar_s, 2)
    return out


def pack_reference(root: Path, pack: str) -> Path | None:
    """A mastering reference that sounds like the pack: the loudest 30 s of 8 of its training tracks,
    stitched together (built once, kept on the volume)."""
    import numpy as np
    ref = root / "references" / f"{pack}.wav"
    if ref.exists():
        return ref
    tracks = sorted((root / "datasets" / pack).glob("*_full.flac")) or sorted((root / "datasets" / pack).glob("*.flac"))
    if not tracks:
        return None
    pick = [tracks[int(i)] for i in np.linspace(0, len(tracks) - 1, min(8, len(tracks)))]
    pieces = []
    for t in pick:
        x = load(t)
        sec = SR
        frames = x.shape[1] // sec
        if frames < 30:
            pieces.append(x)
            continue
        e = np.array([np.mean(x[:, i * sec:(i + 1) * sec] ** 2) for i in range(frames)])
        win = np.convolve(e, np.ones(30), mode="valid")
        s = int(np.argmax(win)) * sec
        pieces.append(x[:, s:s + 30 * sec])
    ref.parent.mkdir(parents=True, exist_ok=True)
    save(ref, np.concatenate(pieces, 1), "PCM_16")
    return ref


def master_to_reference(song, root: Path, work: Path, master: str, pack: str):
    """Match the song's loudness, tone and width to a reference (Matchering). Falls back to GLORE's own master."""
    from glore_engine.mixing import master as basic_master
    try:
        import matchering as mg
        if master == "pack":
            ref, label = pack_reference(root, pack), f"{pack} pack"
        else:
            f = root / "library" / master / "item.json"
            it = json.loads(f.read_text(encoding="utf-8"))
            ref, label = to_wav(root / it["file"], work / "reference.wav"), it["name"]
        if ref is None:
            raise FileNotFoundError("no reference audio")
        tgt = save(work / "premaster.wav", song, "FLOAT")
        out = work / "mastered.wav"
        mg.process(target=str(tgt), reference=str(ref), results=[mg.pcm24(str(out))])
        return load(out), label
    except Exception as e:
        print(f"reference master failed ({e}); using GLORE's own", flush=True)
        return basic_master(song, SR), None


# ---------------------------------------------------------------- the producer (session plans)

BED_BARS = 400.0   # "the whole timeline": bed stems exist wherever the plan plays them


def _pdir(root: Path, sid: str) -> Path:
    d = root / "glore" / "produce" / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def op_pmap(job, root, work, item, p):
    """p: sid, vocal, music?, bpm, key, style, seed. Listens to the song: saves the dry sources and the song map."""
    import numpy as np
    from glore_engine import songmap
    from glore_engine.separation import separate
    d = _pdir(root, p["sid"])
    bpm = float(p["bpm"])
    dry = load(to_wav(p["vocal"], work / "vocal.wav"))
    stems = {}
    if p.get("music"):
        music_wav = to_wav(p["music"], work / "music.wav")
        parts = separate(music_wav, work / "orig_stems", model_name="htdemucs")
        stems = {k: load(v) for k, v in parts.items() if k in ("drums", "bass", "other")}
    grid = stems.get("drums") if stems else dry
    bt = drum_beats(grid if grid is not None else dry, bpm)
    offset = float(bt[0] % (60.0 / bpm)) if len(bt) else 0.0
    key = p.get("key")
    if stems:  # the instruments say the key more reliably than the voice alone
        try:
            from glore_engine.analysis import analyse
            harm = sum(v for k, v in stems.items() if k != "drums")
            key = analyse(str(save(work / "harm.wav", harm)), "beat").key or key
        except Exception as e:
            print(f"key from the music failed ({e}); keeping {key}", flush=True)
    p = {**p, "key": key}
    m = songmap.build(dry, SR, bpm, p.get("key"), offset, music=stems or None)
    save(d / "vocal.wav", dry)
    bars = {"vocal": m["bars"]}
    riff = songmap.riff_loop(dry, SR, bpm, p.get("key"), offset, style=p.get("style", "house"), seed=int(p.get("seed", 0)))
    if riff is not None:
        save(d / "chopriff.wav", riff)
        bars["chopriff"] = 4
    for k in ("bed.bass", "bed.other", "bed.vox", "kit.bass", "kit.keys", "kit.pad", "kit.full", "kit.kick", "kit.clap", "kit.hats", "kit.perc"):
        bars[k] = BED_BARS
    for k, v in stems.items():
        save(d / f"orig.{k}.wav", v)
        bars[f"orig.{k}"] = m["bars"]
    meta = {"bpm": bpm, "key": p.get("key"), "offset": offset, "source_bars": bars, "roots": m["roots"],
            "root_cycle": m["root_cycle"], "chords": m.pop("_chords"), "chord_cycle": m.pop("_chord_cycle")}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "songmap.json").write_text(json.dumps(m), encoding="utf-8")
    return {"songmap": m, "source_bars": bars, "sid": p["sid"]}


def _psources(d: Path, plan: dict, work: Path) -> dict:
    """The session's sources at the plan's tempo and key (conformed copies are kept for the next pass)."""
    from glore_engine.session import Source, conform
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    bpm, tr = float(plan["bpm"]), float(plan.get("transpose") or 0)
    same = abs(bpm - meta["bpm"]) < 0.01 and abs(tr) < 0.01
    cdir = d if same else d / f"conf_{bpm:g}_{tr:g}"
    cdir.mkdir(exist_ok=True)
    out = {}
    for f in sorted(d.glob("*.wav")):
        name = f.stem
        if name.startswith(("bed", "guide")):
            continue
        g = cdir / f.name
        if not g.exists():
            save(g, conform(load(f), SR, meta["bpm"], bpm, 0.0 if name == "orig.drums" else tr))
        off = 0.0 if name == "chopriff" else meta["offset"] * meta["bpm"] / bpm
        out[name] = Source(load(g), off)
    from glore_engine import fxrack
    kit = fxrack.make_kit(plan.get("style") or "house", int(plan["bars"]) + 1, SR, bpm)
    for part, x in kit.items():
        out[f"kit.{part}"] = Source(x, 0.0)
    if meta.get("roots"):
        from glore_engine.songmap import plan_roots
        tr = int(round(float(plan.get("transpose") or 0)))
        roots = [(r + tr) % 12 for r in plan_roots(plan, meta["roots"], meta.get("root_cycle"))]
        out["kit.bass"] = Source(fxrack.make_bass(roots, plan.get("style") or "house", SR, bpm), 0.0)
    if meta.get("chords"):
        from glore_engine.songmap import plan_roots
        tr = int(round(float(plan.get("transpose") or 0)))
        ch = [(c % 12 + tr) % 12 + (12 if c >= 12 else 0) for c in plan_roots(plan, meta["chords"], meta.get("chord_cycle"))]
        for kind in ("keys", "pad"):
            out[f"kit.{kind}"] = Source(fxrack.make_chords(ch, plan.get("style") or "house", SR, bpm, kind), 0.0)
    return out


def op_pguide(job, root, work, item, p):
    """p: sid, plan. Renders what the instrumental will be built around (the lead vocal as arranged)."""
    from glore_engine import session
    d = _pdir(root, p["sid"])
    plan = p["plan"]
    sources = _psources(d, plan, work)
    session.calibrate(sources, SR)
    g = session.render(plan, sources, SR, guide_only=True)
    path = save(d / "guide.wav", g["mix"])
    return {"guide": str(path), "duration": round(g["mix"].shape[1] / SR, 2), "warnings": g["warnings"]}


def op_prender(job, root, work, item, p):
    """p: sid, plan, bed (the generated take), out_base, master, pack. Renders the session, measures it, masters it."""
    import numpy as np
    from glore_engine import producer, session
    from glore_engine.separation import separate
    from glore_engine.session import Source
    from glore_engine.mixing import master as basic_master
    d = _pdir(root, p["sid"])
    plan = p["plan"]
    sources = _psources(d, plan, work)
    bed_dir = Path(p["bed"]).with_suffix("")          # stems are kept beside the bed for a second pass
    if not (bed_dir / "drums.wav").exists():
        parts = separate(to_wav(p["bed"], work / "bed.wav"), work / "bed_stems", model_name="htdemucs")
        bed_dir.mkdir(parents=True, exist_ok=True)
        for k, v in parts.items():
            save(bed_dir / f"{k}.wav", load(v))
    # the bed's drum stem is the model's copy of our own kit (it was in the guide): the real kit replaces it
    for k, name in (("bass", "bed.bass"), ("other", "bed.other"), ("vocals", "bed.vox")):
        if (bed_dir / f"{k}.wav").exists():
            sources[name] = Source(load(bed_dir / f"{k}.wav"), 0.0)
    levels = session.calibrate(sources, SR)
    out = session.render(plan, sources, SR)
    metrics = producer.measure(out, plan, SR)
    metrics["source_gains_db"] = levels
    sh = metrics["spectrum_share"]   # a boomy, dull mix gets a gentle tilt before mastering
    tilt = []
    if sh["low_under_150hz"] > 0.4:
        tilt.append({"type": "eq", "low_db": -min(4.0, 10 * (sh["low_under_150hz"] - 0.35)), "low_hz": 140})
    if sh["high_over_4k"] < 0.08:
        tilt.append({"type": "eq", "high_db": min(4.0, 40 * (0.08 - sh["high_over_4k"]) + 1.5), "high_hz": 5000})
    if tilt:
        from glore_engine.fxrack import apply_fx
        out["mix"] = apply_fx(out["mix"], SR, tilt, {"bpm": plan["bpm"]})
        metrics["tone_fix"] = tilt
    song = out["mix"]
    master, mastered_to = p.get("master", "pack"), None
    if master not in ("none", None):
        song, mastered_to = master_to_reference(song, root, work, master, p.get("pack", "house"))
    else:
        song = basic_master(song, SR)
    base = Path(p["out_base"])
    base.parent.mkdir(parents=True, exist_ok=True)
    wav = save(work / "song.wav", song, "PCM_24")
    ffmpeg("-i", str(wav), "-c:a", "libmp3lame", "-b:a", "320k", f"{base}.mp3")
    groups = out.get("groups", {})
    beat = sum((v for k, v in groups.items() if k != "vocal"), np.zeros_like(out["mix"]))
    beat = beat * (0.89 / max(1e-6, float(np.max(np.abs(beat)))))
    ffmpeg("-i", str(save(work / "beat.wav", beat)), "-c:a", "libmp3lame", "-b:a", "320k", f"{base} (beat).mp3")
    z = Path(f"{base} (stems).zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        zf.write(wav, f"{base.name} - master.wav")
        for k, v in out["tracks"].items():
            zf.write(save(work / f"t_{k}.wav", v, "PCM_24"), f"{base.name} - {k}.wav")
        zf.writestr(f"{base.name} - plan.json", json.dumps(plan, indent=1))
    (d / "last_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return {"mp3": f"{base}.mp3", "beat_mp3": f"{base} (beat).mp3", "zip": str(z), "metrics": metrics,
            "mastered_to": mastered_to, "duration": metrics["length_seconds"]}


def op_mixsong(job, root, work, item, p):
    """p: take, vocal, bpm, key, fx, reverb, delay, chops (style or ''), chops_seed, vocal_db, beat_db,
    chops_db, out_base (path without extension), stems (bool)."""
    import numpy as np
    from glore_engine.separation import separate
    from glore_engine.fx import vocal_fx
    from glore_engine.chops import make_chops
    from glore_engine.mixing import mix_song
    bpm = float(p["bpm"])
    take = to_wav(p["take"], work / "take.wav")
    parts = separate(take, work / "take_stems", model_name="htdemucs")  # the model sings its own vocal too
    music = {k: load(v) for k, v in parts.items() if k != "vocals"}
    dry = load(to_wav(p["vocal"], work / "vocal.wav"))
    if p.get("ai_vox_db") is not None and "vocals" in parts:  # ...keep its chops / ad-libs as texture, in the gaps
        from glore_engine.mixing import ai_textures
        music["ai_vox"] = ai_textures(load(parts["vocals"]), dry, SR, float(p["ai_vox_db"]))
    wet = vocal_fx(dry, SR, p.get("fx", "clean"), bpm, float(p.get("reverb", 0.5)), float(p.get("delay", 0.5)))
    chops, chop_info = None, None
    if p.get("chops"):
        beat_mix = sum(music.values())
        structure = None
        if p.get("music"):  # a full song: its own arrangement shows where the chorus is
            m = load(to_wav(p["music"], work / "orig_music.wav"))
            n = min(m.shape[1], dry.shape[1])
            structure = m[:, :n] + dry[:, :n]
        bt = drum_beats(music.get("drums", beat_mix), bpm)
        chops, info = make_chops(dry, SR, bpm, p.get("key"), style=p["chops"], seed=int(p.get("chops_seed", 0)),
                                 density=float(p.get("chops_density", 1)), offset=float(bt[0] % (60.0 / bpm)) if len(bt) else 0.0,
                                 harmony=beat_mix, structure=structure, beat_times=bt)
        chop_info = {"chorus": info.chorus, "hook": info.hook, "melody": info.melody[:16], "hits": info.hits,
                     "in_song": info.in_song}
    master = p.get("master", "pack")   # "pack" | a library item id | "none"
    song, out = mix_song(music, wet, SR, bpm, chops=chops, vocal_db=float(p.get("vocal_db", 0)),
                         beat_db=float(p.get("beat_db", 0)), chops_db=float(p.get("chops_db", 0)),
                         finish=master in ("none", None))
    mastered_to = None
    if master not in ("none", None):
        song, mastered_to = master_to_reference(song, root, work, master, p.get("pack", "house"))
    base = Path(p["out_base"])
    base.parent.mkdir(parents=True, exist_ok=True)
    wav = save(work / "song.wav", song, "PCM_24")
    ffmpeg("-i", str(wav), "-c:a", "libmp3lame", "-b:a", "320k", f"{base}.mp3")
    beat = sum(v for k, v in out.items() if k not in ("vocal", "ai_vox"))
    beat = beat * (0.89 / max(1e-6, float(np.max(np.abs(beat)))))
    ffmpeg("-i", str(save(work / "beat.wav", beat)), "-c:a", "libmp3lame", "-b:a", "320k", f"{base} (beat).mp3")
    res = {"mp3": f"{base}.mp3", "beat_mp3": f"{base} (beat).mp3", "chops": chop_info, "mastered_to": mastered_to}
    if p.get("stems", True):
        z = Path(f"{base} (stems).zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            zf.write(wav, f"{base.name} - master.wav")
            for k, v in out.items():
                zf.write(save(work / f"{k}.wav", v, "PCM_24"), f"{base.name} - {k}.wav")
            zf.write(save(work / "vocal_dry.wav", dry, "PCM_24"), f"{base.name} - vocal (dry).wav")
        res["zip"] = str(z)
    return res


OPS = {"pmap": op_pmap, "pguide": op_pguide, "prender": op_prender, "arrange": op_arrange, "stems": op_stems, "clean": op_clean, "analyse": op_analyse, "lock": op_lock, "chops": op_chops,
       "mixsong": op_mixsong}


if __name__ == "__main__":
    try:
        op, path = sys.argv[1], sys.argv[2]
        job = json.load(open(path, encoding="utf-8"))
        root, work = Path(job["root"]), Path(job["work"])
        work.mkdir(parents=True, exist_ok=True)
        res = OPS[op](job, root, work, job.get("item") or {}, job.get("params") or {})
        print("GLORE_RESULT " + json.dumps(res), flush=True)
    except Exception as e:
        traceback.print_exc()
        print("GLORE_ERROR " + json.dumps({"error": str(e)}), flush=True)
        sys.exit(1)
