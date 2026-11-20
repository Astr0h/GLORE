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


def beat_offset(beat, bpm: float) -> float:
    """Where the beat grid starts (first beat, folded into one beat length)."""
    import librosa
    y = librosa.resample(beat.mean(0), orig_sr=SR, target_sr=22050)
    _, beats = librosa.beat.beat_track(y=y, sr=22050, start_bpm=bpm, units="time")
    return float(beats[0] % (60.0 / bpm)) if len(beats) else 0.0


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
    parts = separate(take, work / "take_stems", model_name="htdemucs")  # the model sings its own vocal: drop it
    music = {k: load(v) for k, v in parts.items() if k != "vocals"}
    dry = load(to_wav(p["vocal"], work / "vocal.wav"))
    wet = vocal_fx(dry, SR, p.get("fx", "clean"), bpm, float(p.get("reverb", 0.5)), float(p.get("delay", 0.5)))
    chops, chop_info = None, None
    if p.get("chops"):
        beat_mix = sum(music.values())
        structure = None
        if p.get("music"):  # a full song: its own arrangement shows where the chorus is
            m = load(to_wav(p["music"], work / "orig_music.wav"))
            n = min(m.shape[1], dry.shape[1])
            structure = m[:, :n] + dry[:, :n]
        chops, info = make_chops(dry, SR, bpm, p.get("key"), style=p["chops"], seed=int(p.get("chops_seed", 0)),
                                 density=float(p.get("chops_density", 1)), offset=beat_offset(beat_mix, bpm),
                                 harmony=beat_mix, structure=structure)
        chop_info = {"chorus": info.chorus, "hook": info.hook, "chords": info.chords[:16], "hits": info.hits}
    song, out = mix_song(music, wet, SR, bpm, chops=chops, vocal_db=float(p.get("vocal_db", 0)),
                         beat_db=float(p.get("beat_db", 0)), chops_db=float(p.get("chops_db", 0)))
    base = Path(p["out_base"])
    base.parent.mkdir(parents=True, exist_ok=True)
    wav = save(work / "song.wav", song, "PCM_24")
    ffmpeg("-i", str(wav), "-c:a", "libmp3lame", "-b:a", "320k", f"{base}.mp3")
    beat = sum(v for k, v in out.items() if k not in ("vocal",))
    beat = beat * (0.89 / max(1e-6, float(np.max(np.abs(beat)))))
    ffmpeg("-i", str(save(work / "beat.wav", beat)), "-c:a", "libmp3lame", "-b:a", "320k", f"{base} (beat).mp3")
    res = {"mp3": f"{base}.mp3", "beat_mp3": f"{base} (beat).mp3", "chops": chop_info}
    if p.get("stems", True):
        z = Path(f"{base} (stems).zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            zf.write(wav, f"{base.name} - master.wav")
            for k, v in out.items():
                zf.write(save(work / f"{k}.wav", v, "PCM_24"), f"{base.name} - {k}.wav")
            zf.write(save(work / "vocal_dry.wav", dry, "PCM_24"), f"{base.name} - vocal (dry).wav")
        res["zip"] = str(z)
    return res


OPS = {"stems": op_stems, "clean": op_clean, "analyse": op_analyse, "lock": op_lock, "chops": op_chops,
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
