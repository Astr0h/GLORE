"""Generate: ACE-Step 1.5 on this PC, with one of your LoRAs.

ACE-Step lives in its own venv (%APPDATA%/GLORE/ace, installed by the 8 GB test or
the first generation), so generation runs pod/glore_generate.py as a subprocess in
that venv and reads its progress lines. Each take lands in the project as a layer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from . import training
from .pod import glore_pod
from .project import PACKS, Layer, ProjectStore, new_id

SCRIPT = Path(__file__).parent / "pod" / "glore_generate.py"
Report = Callable[[float, str], None]


class GenerateError(RuntimeError):
    pass


def default_caption(pack: str, moods: list[str] | None = None) -> str:
    label = PACKS[pack]["label"].lower()
    return ", ".join([label, *(moods or [])])


def find_lora(run_id: str | None, pack: str | None) -> dict | None:
    loras = training.list_loras(pack)
    if run_id:
        loras = [l for l in loras if l.get("run_id") == run_id]
    return loras[0] if loras else None


def ace_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return env


def ensure_ace(report: Report, variant: str = "turbo") -> glore_pod.Ace:
    ace = glore_pod.Ace(training.glore_home() / "ace", [sys.executable, "-m", "uv"], ace_env())
    status = training._ReportStatus(training.glore_home() / "ace-status.json", report, 0.0, 0.0)
    with open(training.glore_home() / "ace-install.log", "a", encoding="utf-8") as log:
        ace.ensure(variant, log, status)
    return ace


def run_generator(ace, job: dict, tmp: Path, log_path: Path, report: Report, lo: float = 0.05, span: float = 0.9) -> dict:
    """Run pod/glore_generate.py in ACE-Step's venv; returns its GLORE_RESULT."""
    (tmp / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    cmd = ace.py(str(SCRIPT), str(tmp / "job.json"))
    result, error = None, None
    win = os.name == "nt"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=str(ace.dir), env=ace.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                creationflags=0x08000000 if win else 0)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            line = line.rstrip("\n")
            if line.startswith("GLORE_PROGRESS "):
                _, frac, msg = line.split(" ", 2)
                report(lo + span * float(frac), msg)
            elif line.startswith("GLORE_RESULT "):
                result = json.loads(line[len("GLORE_RESULT "):])
            elif line.startswith("GLORE_ERROR "):
                error = json.loads(line[len("GLORE_ERROR "):]).get("error")
        proc.wait()
    if proc.returncode or not result:
        err = error or "Generation failed"
        if "out of memory" in log_path.read_text(encoding="utf-8", errors="replace").lower():
            err = "Ran out of GPU memory. Try a shorter length or one take at a time."
        raise GenerateError(f"{err} (log: {log_path})")
    return result


def generate(store: ProjectStore, report: Report, *, pack: str = "house", lora_run: str | None = None,
             use_lora: bool = True, caption: str | None = None, bpm: float | None = None, key: str | None = None,
             duration: float = 60.0, seed: int = -1, count: int = 1, scale: float = 1.0,
             offload: bool = True, lm: str | None = "acestep-5Hz-lm-1.7B",
             rewrite_caption: bool = True, velocity_norm: float = 0.0, model: str | None = None,
             export_dir: str | None = None) -> dict:
    """model = turbo | base | sft (default: the LoRA's own model, else turbo). SFT/base run 50 steps with CFG."""
    lora = find_lora(lora_run, pack) if use_lora else None
    if use_lora and not lora:
        raise GenerateError(f"No {PACKS[pack]['label']} LoRA yet. Train one on the LoRA Packs page, or turn the LoRA off.")
    if model and model not in glore_pod.VARIANT_DIRS:
        raise GenerateError("model must be turbo, base or sft")
    if lora and model and model != lora.get("variant", "turbo"):
        raise GenerateError(f"This LoRA was trained on {lora.get('variant', 'turbo')}, not {model}")
    variant = (lora or {}).get("variant") or model or "turbo"
    inference = (lora or {}).get("inference") or glore_pod.VARIANT_INFERENCE[variant]
    p = store.project
    bpm = bpm or p.bpm
    key = key or p.key

    report(0.01, "Checking ACE-Step")
    ace = ensure_ace(lambda pr, m: report(0.02, m), variant)

    gid = new_id()
    gen_dir = store.root / "generated"
    tmp = gen_dir / f"_{gid}"
    tmp.mkdir(parents=True, exist_ok=True)
    job = {"ace_dir": str(ace.dir), "out_dir": str(tmp), "variant": variant,
           "caption": caption if caption is not None else default_caption(pack),
           "lora": {"path": lora["path"], "trigger": lora.get("trigger"), "scale": scale} if lora else None,
           "bpm": bpm, "key": key, "duration": duration, "seed": seed, "count": count,
           "steps": inference.get("steps"), "shift": inference.get("shift"), "offload": offload, "lm": lm,
           "rewrite_caption": rewrite_caption, "velocity_norm": velocity_norm}
    (tmp / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")

    result = run_generator(ace, job, tmp, gen_dir / f"{gid}.log", report)

    report(0.97, "Adding to the project")
    label = PACKS[pack]["label"] if lora else "No LoRA"
    takes = []
    for i, f in enumerate(result["files"]):
        src = Path(f["path"])
        name = f"{gid}-{i + 1}{src.suffix or '.wav'}"
        shutil.move(str(src), gen_dir / name)
        layer = Layer(id=f"gen-{gid}-{i + 1}", name=f"Generated · {label} · seed {f.get('seed')}",
                      file=f"generated/{name}", role="generated")
        store.upsert_layer(layer)
        take = {"layer_id": layer.id, "file": layer.file, "seed": f.get("seed")}
        if export_dir:
            tag = "LoRA" if lora else "no LoRA"
            mp3 = Path(export_dir) / f"GLORE {short_name(caption or default_caption(pack))} ({tag}, seed {f.get('seed')}).mp3"
            _mp3(gen_dir / name, mp3)
            take["mp3"] = str(mp3)
        takes.append(take)
    store.save()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"takes": takes, "caption": result.get("caption"), "planned_caption": result.get("planned_caption"),
            "seconds": result.get("seconds"),
            "peak_vram_gb": result.get("peak_vram_gb"), "lora": (lora or {}).get("run_id")}


# ---------------------------------------------------------------- from the song a vocal came from

def generate_from_song(store: ProjectStore, report: Report, *, vocal_source: str, grid_source: str | None = None,
                       source: str = "vocal", pack: str = "house", lora_run: str | None = None,
                       use_lora: bool = True, caption: str | None = None, strength: float = 0.5, seed: int = -1,
                       scale: float = 1.0, vocal_gain_db: float = -3.0, beat_gain_db: float = 0.0,
                       export_dir: str | None = None, name: str | None = None) -> dict:
    """A new beat that ACE-Step builds while hearing the vocal (ACE-Step cover, turbo + LoRA).

    source="vocal" (default): the a cappella goes in on its own (the original beat is gone), so the
    model writes a fresh beat around the vocal's own rhythm and phrasing. The model sings its own
    copy of the vocal back; that copy is separated out and your clean vocal goes on top.
    source="instrumental": the original song minus the vocal goes in instead (restyles the old beat)."""
    import numpy as np
    from . import audio_io, separation

    vocal = store.source(vocal_source)
    lora = find_lora(lora_run, pack) if use_lora else None
    if use_lora and not lora:
        raise GenerateError(f"No {PACKS[pack]['label']} LoRA yet. Train one, or turn the LoRA off.")
    variant = (lora or {}).get("variant", "turbo")
    inference = (lora or {}).get("inference") or glore_pod.VARIANT_INFERENCE[variant]
    meta_src = store.source(grid_source) if grid_source else (
        store.source(vocal.derived_from) if vocal.derived_from else vocal)
    meta = {**vocal.effective(), **{k: v for k, v in meta_src.effective().items() if v}}
    sr = audio_io.PROJECT_SR
    if source == "instrumental" and meta_src is vocal:
        raise GenerateError("This needs the full song the vocal came from")

    report(0.01, "Checking ACE-Step")
    ace = ensure_ace(lambda pr, m: report(0.02, m), variant)
    gid = new_id()
    gen_dir = store.root / "generated"
    tmp = gen_dir / f"_{gid}"
    tmp.mkdir(parents=True, exist_ok=True)

    voc = audio_io.load_stereo(store.sources_dir / vocal.file, sr)
    if source == "instrumental":
        report(0.03, "Taking the vocal out of the original")
        stems = _stems_of(store, report, meta_src.id, store.sources_dir / meta_src.file, 0.03, 0.1)
        n = max(stems[k].shape[1] for k in ("drums", "bass", "other"))
        audio = sum(np.pad(stems[k], ((0, 0), (0, n - stems[k].shape[1]))) for k in ("drums", "bass", "other"))
    else:
        audio = voc
    src = tmp / "source.wav"
    audio_io.save_wav(src, audio, sr, subtype="PCM_16")

    caption = caption if caption is not None else default_caption(pack)
    job = {"mode": "cover", "src_audio": str(src), "strength": strength,
           "ace_dir": str(ace.dir), "out_dir": str(tmp), "variant": variant, "caption": caption,
           "lora": {"path": lora["path"], "trigger": lora.get("trigger"), "scale": scale} if lora else None,
           "bpm": meta.get("bpm"), "key": meta.get("key_name") or meta.get("key"), "duration": audio.shape[1] / sr,
           "seed": seed, "count": 1, "steps": inference.get("steps"), "shift": inference.get("shift"),
           "offload": True, "lm": None}
    (tmp / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    result = run_generator(ace, job, tmp, gen_dir / f"{gid}.log", report, 0.14, 0.6)
    take = result["files"][0]

    if source == "instrumental":
        beat = audio_io.load_stereo(take["path"], sr)
    else:
        report(0.76, "Removing the model's copy of the vocal")
        parts = separation.separate(take["path"], tmp / "stems", progress=lambda pr, m: report(0.76 + 0.14 * pr, m))
        beats = [audio_io.load_stereo(parts[k], sr) for k in parts if k != "vocals"]
        n = max(x.shape[1] for x in beats)
        beat = sum(np.pad(x, ((0, 0), (0, n - x.shape[1]))) for x in beats)

    report(0.92, "Laying your vocal on it")
    beat_rel = f"generated/{gid}-beat.wav"
    audio_io.save_wav(store.root / beat_rel, beat, sr)
    n = max(beat.shape[1], voc.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, n - x.shape[1])))
    mix = pad(voc) * 10 ** (vocal_gain_db / 20) + pad(beat) * 10 ** (beat_gain_db / 20)
    mix *= 0.89 / max(1e-6, float(np.max(np.abs(mix))))
    song_rel = f"generated/{gid}-song.wav"
    audio_io.save_wav(store.root / song_rel, mix, sr)

    title = name or vocal.name.replace(" (vocal)", "")
    label = PACKS[pack]["label"] if lora else "No LoRA"
    store.upsert_layer(Layer(id=f"gen-{gid}-beat", name=f"Beat · {label} · built on {vocal.name}", file=beat_rel,
                             role="generated"))
    store.upsert_layer(Layer(id=f"gen-{gid}-song", name=f"Song · {title}", file=song_rel, role="generated"))
    store.save()
    shutil.rmtree(tmp, ignore_errors=True)
    out = {"beat_layer": f"gen-{gid}-beat", "song_layer": f"gen-{gid}-song", "seed": take.get("seed"),
           "source": source, "strength": strength, "seconds": result.get("seconds"),
           "peak_vram_gb": result.get("peak_vram_gb"), "lora": (lora or {}).get("run_id")}
    if export_dir:
        safe = "".join(c for c in f"{title} (GLORE)" if c not in '<>:"/\\|?*').strip()
        for kind, rel, fname in (("mp3", song_rel, safe), ("beat_mp3", beat_rel, f"{safe} beat")):
            mp3 = Path(export_dir) / f"{fname}.mp3"
            _mp3(store.root / rel, mp3)
            out[kind] = str(mp3)
    report(1.0, "Done")
    return out


# ---------------------------------------------------------------- around a vocal

def short_name(text: str, limit: int = 48) -> str:
    """A prompt as a safe, short file-name part (Windows paths max out at 260 characters)."""
    t = "".join(c for c in " ".join(str(text).split()) if c not in '<>:"/\\|?*').strip(" .")
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0].rstrip(" ,")
    return (cut or t[:limit]) + "…"


def _mp3(wav: Path, mp3: Path) -> None:
    import imageio_ffmpeg
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(wav), "-c:a", "libmp3lame",
                    "-b:a", "256k", str(mp3)], check=True, creationflags=0x08000000 if os.name == "nt" else 0)


def generate_around_vocal(store: ProjectStore, report: Report, *, source_id: str | None = None,
                          caption: str = "soft house", tracks: list[str] | None = None, seed: int = -1,
                          steps: int = 50, guidance: float = 7.0, beat_gain_db: float = -3.0,
                          export_dir: str | None = None) -> dict:
    """ACE-Step 'complete' (base model): write drums/bass/keys/synths around your vocal,
    then strip whatever vocal the model echoed back and lay your clean vocal on top."""
    import numpy as np
    from . import audio_io, separation

    p = store.project
    if source_id:
        vocal = store.source(source_id)
    else:
        lock = p.vocal_lock or {}
        vocal = store.source(lock["source_id"]) if lock.get("source_id") else None
        vocal = vocal or next((s for s in reversed(p.sources) if s.kind == "vocal"), None)
    if not vocal:
        raise GenerateError("Add or extract a vocal first")
    vocal_path = store.sources_dir / vocal.file
    a = vocal.effective()

    report(0.01, "Checking ACE-Step (base model)")
    ace = ensure_ace(lambda pr, m: report(0.02, m), "base")

    gid = new_id()
    gen_dir = store.root / "generated"
    tmp = gen_dir / f"_{gid}"
    tmp.mkdir(parents=True, exist_ok=True)
    job = {"mode": "complete", "ace_dir": str(ace.dir), "out_dir": str(tmp), "variant": "base",
           "src_audio": str(vocal_path), "caption": caption, "tracks": tracks, "lora": None,
           "bpm": a.get("bpm"), "key": a.get("key_name") or a.get("key"), "duration": vocal.duration or None,
           "seed": seed, "count": 1, "steps": steps, "shift": 1.0, "guidance": guidance, "offload": True}
    result = run_generator(ace, job, tmp, gen_dir / f"{gid}.log", report, 0.04, 0.66)
    full = Path(result["files"][0]["path"])

    report(0.72, "Separating the new beat from the model's copy of the vocal")
    stems = separation.separate(full, tmp / "stems", progress=lambda pr, m: report(0.72 + 0.18 * pr, m))
    sr = audio_io.PROJECT_SR
    parts = [audio_io.load_stereo(stems[k], sr) for k in stems if k != "vocals"]   # each (2, samples)
    voc = audio_io.load_stereo(vocal_path, sr)
    n = max([voc.shape[1], *[x.shape[1] for x in parts]])
    pad = lambda x: np.pad(x, ((0, 0), (0, n - x.shape[1])))  # resampling leaves lengths a few samples apart
    beat = sum(pad(x) for x in parts)
    voc = pad(voc)
    mix = voc + beat * (10 ** (beat_gain_db / 20))
    mix *= 0.89 / max(1e-6, float(np.max(np.abs(mix))))   # peak at about -1 dBFS

    report(0.93, "Saving")
    name = f"{vocal.name.replace(' (vocal)', '')}"
    files = {}
    for kind, audio in (("beat", beat), ("mix", mix)):
        rel = f"generated/{gid}-{kind}.wav"
        audio_io.save_wav(store.root / rel, audio, sr)
        layer = Layer(id=f"gen-{gid}-{kind}", name=f"{'Beat' if kind == 'beat' else 'Mix'} · {caption} · {name}",
                      file=rel, role="generated")
        store.upsert_layer(layer)
        files[kind] = rel
    store.save()
    out = {"files": files, "seed": result["files"][0].get("seed"), "seconds": result.get("seconds"),
           "peak_vram_gb": result.get("peak_vram_gb"), "vocal": vocal.name}
    if export_dir:
        safe = "".join(c for c in f"{name} - GLORE {caption}" if c not in '<>:"/\\|?*').strip()
        mp3 = Path(export_dir) / f"{safe}.mp3"
        _mp3(store.root / files["mix"], mp3)
        out["mp3"] = str(mp3)
    shutil.rmtree(tmp, ignore_errors=True)
    report(1.0, "Done")
    return out


# ---------------------------------------------------------------- vocal over a generated beat

def bar_offset(song_downbeats: list[float], beat_downbeats: list[float], factor: float, bar: float) -> float:
    """Seconds to delay the (stretched) vocal so the song's bar lines land on the beat's.
    `factor` = target_bpm / source_bpm (times in the song shrink by 1/factor)."""
    if not song_downbeats or not beat_downbeats:
        return 0.0
    d0 = song_downbeats[0] / factor
    return float((beat_downbeats[0] - d0) % bar)


def _stems_of(store: ProjectStore, report: Report, owner_id: str, path: Path, lo: float, span: float) -> dict:
    """The 4 Demucs stems of a file, separated once and kept in stems/<owner_id>/."""
    from . import audio_io, separation

    d = store.stems_dir / owner_id
    names = ("drums", "bass", "other", "vocals")
    found = {n: next(iter(d.glob(f"*_{n}.wav")), None) if d.exists() else None for n in names}
    if not all(found.values()):
        found = separation.separate(path, d, progress=lambda p, m: report(lo + p * span, m))
    return {n: audio_io.load_stereo(found[n], audio_io.PROJECT_SR) for n in names}


def _arranged_beat(store, report, layer, grid, ga, song_bpm, factor, offset, sr):
    """Read the original song's arrangement and replay it on the beat's stems."""
    from . import arrange

    report(0.45, "Reading the original song's arrangement")
    song = _stems_of(store, report, grid.id, store.sources_dir / grid.file, 0.45, 0.15)
    d0, song_bar = float(ga["downbeats"][0]), 240.0 / song_bpm
    lv = {k: arrange.bar_levels(song[k], sr, d0, song_bar) for k in ("drums", "bass", "other")}
    bars = arrange.section_map(lv["drums"], lv["bass"], lv["other"])
    report(0.6, "Splitting the beat into stems")
    beat_stems = _stems_of(store, report, layer.id, store.root / layer.file, 0.6, 0.25)
    report(0.86, "Arranging the beat around the vocal")
    starts = [(d0 + i * song_bar) / factor + offset for i in range(len(bars))]
    beat = arrange.arrange_stems(beat_stems, sr, bars, starts)
    secs = arrange.sections(bars)
    for s in secs:
        s["start_s"] = round(starts[s["start_bar"]], 2)
    return beat, secs


def mix_vocal_over_beat(store: ProjectStore, report: Report, *, beat_layer: str, vocal_source: str,
                        grid_source: str | None = None, beat_bpm: float | None = None,
                        vocal_gain_db: float = -2.0, beat_gain_db: float = 0.0, arrange: bool = True,
                        export_dir: str | None = None, name: str | None = None) -> dict:
    """Lay a vocal on a generated beat: match tempo (Rubber Band), line up bar lines, mix.
    grid_source = the full song the vocal came from (its drums give far better downbeats).
    arrange = replay the original song's arrangement (breakdowns, drum cuts, drops) on the beat."""
    import numpy as np
    from . import analysis, audio_io, vocal_lock

    p = store.project
    layer = next((l for l in p.layers if l.id == beat_layer), None)
    if not layer:
        raise GenerateError("That beat isn't in the project")
    vocal = store.source(vocal_source)
    grid = store.source(grid_source) if grid_source else (
        store.source(vocal.derived_from) if vocal.derived_from else vocal)
    ga = grid.effective()
    song_bpm = float(ga.get("bpm") or vocal.effective().get("bpm") or 0)
    if not song_bpm:
        raise GenerateError("Analyse the vocal first (no BPM yet)")
    sr = audio_io.PROJECT_SR

    report(0.1, "Reading the beat's tempo and bar lines")
    beat = audio_io.load_stereo(store.root / layer.file, sr)
    mono = audio_io.load_mono(store.root / layer.file)
    found_bpm, _, _, beats = analysis.estimate_tempo(mono, 22050)
    target = float(beat_bpm or found_bpm)
    if abs(target * 2 - song_bpm) < abs(target - song_bpm):
        target *= 2
    elif abs(target / 2 - song_bpm) < abs(target - song_bpm):
        target /= 2
    beat_db = analysis.estimate_downbeats(mono, 22050, beats)

    report(0.35, f"Fitting the vocal to {target:.1f} BPM")
    gid = new_id()
    tmp = store.root / "generated" / f"_{gid}-vocal.wav"
    vocal_lock.lock_vocal(store.sources_dir / vocal.file, tmp, song_bpm, target)
    voc = audio_io.load_stereo(tmp, sr)
    tmp.unlink(missing_ok=True)
    factor = target / song_bpm
    bar = 240.0 / target
    offset = bar_offset(ga.get("downbeats") or [], beat_db, factor, bar)

    sections = None
    if arrange and ga.get("downbeats"):
        beat, sections = _arranged_beat(store, report, layer, grid, ga, song_bpm, factor, offset, sr)

    report(0.9, "Mixing")
    shift = int(round(offset * sr))
    voc = np.pad(voc, ((0, 0), (shift, 0)))
    n = max(voc.shape[1], beat.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, n - x.shape[1])))
    mix = pad(voc) * 10 ** (vocal_gain_db / 20) + pad(beat) * 10 ** (beat_gain_db / 20)
    mix = mix[:, : min(n, voc.shape[1] + int(2 * bar * sr))]          # end a couple of bars after the vocal
    fade = int(min(2.0, bar) * sr)
    mix[:, -fade:] *= np.linspace(1, 0, fade)
    mix *= 0.89 / max(1e-6, float(np.max(np.abs(mix))))

    rel = f"generated/{gid}-song.wav"
    audio_io.save_wav(store.root / rel, mix, sr)
    title = name or vocal.name.replace(" (vocal)", "")
    store.upsert_layer(Layer(id=f"gen-{gid}-song", name=f"Song · {title}", file=rel, role="generated"))
    store.save()
    out = {"file": rel, "beat_bpm": round(target, 2), "song_bpm": song_bpm, "stretch": round(factor, 5),
           "offset_s": round(offset, 3), "duration": round(mix.shape[1] / sr, 1)}
    if sections is not None:
        out["sections"] = sections
    if export_dir:
        safe = "".join(c for c in f"{title} (GLORE)" if c not in '<>:"/\\|?*').strip()
        mp3 = Path(export_dir) / f"{safe}.mp3"
        _mp3(store.root / rel, mp3)
        out["mp3"] = str(mp3)
    report(1.0, "Done")
    return out
