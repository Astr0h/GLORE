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

    cmd = ace.py(str(SCRIPT), str(tmp / "job.json"))
    log_path = gen_dir / f"{gid}.log"
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
                report(0.05 + 0.9 * float(frac), msg)
            elif line.startswith("GLORE_RESULT "):
                result = json.loads(line[len("GLORE_RESULT "):])
            elif line.startswith("GLORE_ERROR "):
                error = json.loads(line[len("GLORE_ERROR "):]).get("error")
        proc.wait()
    if proc.returncode or not result:
        err = error or "Generation failed"
        if "out of memory" in (log_path.read_text(encoding="utf-8", errors="replace")).lower():
            err = "Ran out of GPU memory. Try a shorter length or one take at a time."
        raise GenerateError(f"{err} (log: {log_path})")

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
