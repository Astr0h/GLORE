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
             offload: bool = True) -> dict:
    lora = find_lora(lora_run, pack) if use_lora else None
    if use_lora and not lora:
        raise GenerateError(f"No {PACKS[pack]['label']} LoRA yet. Train one on the LoRA Packs page, or turn the LoRA off.")
    variant = (lora or {}).get("variant", "turbo")
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
           "steps": inference.get("steps"), "shift": inference.get("shift"), "offload": offload}
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
        takes.append({"layer_id": layer.id, "file": layer.file, "seed": f.get("seed")})
    store.save()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"takes": takes, "caption": result.get("caption"), "seconds": result.get("seconds"),
            "peak_vram_gb": result.get("peak_vram_gb"), "lora": (lora or {}).get("run_id")}


# ---------------------------------------------------------------- around a vocal

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
