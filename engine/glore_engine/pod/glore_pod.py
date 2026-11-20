"""GLORE trainer - runs on the rented cloud GPU, and on your PC for the 8 GB fit test.

Standard library only: on the pod it starts before any packages exist.

Pod layout (the network volume is mounted at /workspace; GLORE reaches the same
files from your PC through RunPod's S3 API, key = path under /workspace):
    glore/queue/<job>.json        jobs GLORE has asked for
    glore/runs/<job>/status.json  live progress GLORE reads back
    glore/runs/<job>/log.txt
    datasets/<pack>/              FLAC training audio + dataset.json
    tensors/<pack>/<hash>/        preprocessed tensors (reused between runs)
    loras/<pack>/<run>/final/     the trained LoRA
    ace/                          ACE-Step 1.5 + its venv + checkpoints (installed once)

The pod shuts itself down (terminates) after GLORE_IDLE_MIN minutes with no job,
and always at GLORE_HARD_CAP_S seconds after it started."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ACE_REPO = "https://github.com/ace-step/ACE-Step-1.5.git"
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"  # pinned: the CLI flags below match this commit
RUNPOD_API = "https://api.runpod.io/v2"

# "standard" ACE-Step 1.5 = the 2B DiT; xl_* = the 4B DiT (cloud only: too big for 8 GB cards)
VARIANT_DIRS = {"turbo": "acestep-v15-turbo", "base": "acestep-v15-base", "sft": "acestep-v15-sft",
                "xl_turbo": "acestep-v15-xl-turbo", "xl_base": "acestep-v15-xl-base", "xl_sft": "acestep-v15-xl-sft"}
# Inference settings each variant expects; stored with the LoRA so Generate uses the right ones
VARIANT_INFERENCE = {"turbo": {"shift": 3.0, "steps": 8}, "base": {"shift": 1.0, "steps": 50},
                     "sft": {"shift": 1.0, "steps": 50}, "xl_turbo": {"shift": 3.0, "steps": 8},
                     "xl_base": {"shift": 1.0, "steps": 50}, "xl_sft": {"shift": 1.0, "steps": 50}}
CLOUD_LM = "acestep-5Hz-lm-4B"
FFMPEG = ["ffmpeg"]  # the biggest song planner; the cloud GPU has room for it

EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+)(?:, Step (\d+))?.*?Loss: ([0-9.]+)")
OOM_RE = re.compile(r"out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", re.I)


# ---------------------------------------------------------------- small helpers

def now() -> float:
    return time.time()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Status:
    """status.json that GLORE polls. Every update refreshes the heartbeat."""

    def __init__(self, path: Path, job_id: str, mode: str):
        self.path = path
        self.data = {"job": job_id, "mode": mode, "phase": "starting", "message": "Starting",
                     "progress": 0.0, "started": now(), "heartbeat": now()}
        self.lock = threading.Lock()
        self.flush()

    def update(self, **kw) -> None:
        with self.lock:
            self.data.update(kw)
            self.data["heartbeat"] = now()
            self.flush()

    def flush(self) -> None:
        try:
            write_json(self.path, self.data)
        except OSError:
            pass


def default_epochs(n_tracks: int) -> int:
    """Side-Step guide: 1-10 songs 200-500, 10-50 songs 100-200, 50+ songs 50-100."""
    if n_tracks <= 10:
        return 300
    if n_tracks <= 50:
        return 150
    return 80


def cli_variant(variant: str) -> str:
    """What to pass as Side-Step's --model-variant. Its trainer only knows turbo/base/sft by name
    (the preprocessor knows the xl_* aliases too); both accept the model's folder name."""
    return VARIANT_DIRS[variant] if variant.startswith("xl_") else variant


def train_params(n_tracks: int, epochs: int | None = None, rank: int | None = None,
                 save_every: int | None = None) -> dict:
    rank = int(rank or 64)
    return {"epochs": int(epochs or default_epochs(n_tracks)), "rank": rank, "alpha": rank * 2,
            "lr": 1e-4, "batch_size": 1, "grad_accum": 4, "save_every": int(save_every or 10), "log_every": 10,
            "optimizer": "adamw", "offload_encoder": False}


def parse_progress(line: str) -> dict | None:
    m = EPOCH_RE.search(line)
    if not m:
        return None
    out = {"epoch": int(m.group(1)), "epochs": int(m.group(2)), "loss": float(m.group(4))}
    if m.group(3):
        out["step"] = int(m.group(3))
    return out


def run(cmd: list[str], cwd: Path | None, env: dict, log, on_line=None, should_stop=None,
        timeout: float | None = None) -> tuple[int, bool]:
    """Run a command, streaming output to the log. Returns (exit code, stopped_early)."""
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    win = os.name == "nt"
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            bufsize=1, start_new_session=not win,
                            creationflags=0x08000000 if win else 0)  # CREATE_NO_WINDOW
    t0 = now()
    stopped = False

    def watch():
        nonlocal stopped
        while proc.poll() is None:
            if (should_stop and should_stop()) or (timeout and now() - t0 > timeout):
                stopped = True
                kill_tree(proc)
                return
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if on_line:
            on_line(line.rstrip("\n"))
    proc.wait()
    return proc.returncode, stopped


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/pid", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(5)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def gpu_info() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        name, used, total = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "used_mb": int(used), "total_mb": int(total)}
    except Exception:
        return {}


# ---------------------------------------------------------------- ACE-Step install

class Ace:
    """An ACE-Step 1.5 checkout with its own venv and model checkpoints."""

    def __init__(self, ace_dir: Path, uv: list[str], env: dict):
        self.dir = ace_dir
        self.uv = uv
        self.env = env
        self.ckpt = ace_dir / "checkpoints"

    def py(self, *args: str) -> list[str]:
        # --no-sync: never let uv rewrite the env (it would drop extras like bitsandbytes).
        # --project lets the command run from another folder (see workdir).
        return [*self.uv, "run", "--no-sync", "--project", str(self.dir), "python", *args]

    def train_py(self, *args: str) -> list[str]:
        return self.py(str(self.dir / "train.py"), *args)

    def workdir(self, *paths: Path) -> Path:
        """ACE-Step's trainer only accepts data paths under the folder it was started from
        (acestep/training/path_safety.py uses the cwd), so start it from a folder that holds
        both ACE-Step and our data: /workspace on the pod, %APPDATA%/GLORE on the PC."""
        return Path(os.path.commonpath([str(Path(x).resolve()) for x in (self.dir, *paths)]))

    def ensure(self, variant: str, log, status: Status, extras: tuple[str, ...] = ()) -> None:
        marker = self.dir / ".glore_ready"
        if not (marker.exists() and marker.read_text().strip() == ACE_COMMIT):
            status.update(phase="setup", message="Installing ACE-Step 1.5 (first run only, ~5–10 min)")
            if not (self.dir / ".git").exists():
                self.dir.mkdir(parents=True, exist_ok=True)
                for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", ACE_REPO]):
                    run(cmd, self.dir, self.env, log)
            for cmd in (["git", "fetch", "-q", "--depth", "1", "origin", ACE_COMMIT],
                        ["git", "checkout", "-q", "-f", "FETCH_HEAD"]):
                code, _ = run(cmd, self.dir, self.env, log)
                if code:
                    raise RuntimeError("Couldn't download ACE-Step from GitHub")
            code, _ = run([*self.uv, "sync", "--frozen", "--no-dev"], self.dir, self.env, log)
            if code:
                raise RuntimeError("Installing ACE-Step's packages failed (see log)")
            marker.write_text(ACE_COMMIT)
        for pkg in extras:
            mod = pkg.split("[")[0].split(">")[0].split("=")[0].replace("-", "_")
            code, _ = run(self.py("-c", f"import {mod}"), self.dir, self.env, log)
            if code:
                status.update(message=f"Installing {pkg}")
                run([*self.uv, "pip", "install", "--python", str(self.venv_python()), pkg], self.dir, self.env, log)
        self.ensure_models(variant, log, status)

    def venv_python(self) -> Path:
        return self.dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def ensure_extra_model(self, name: str, log, status: Status) -> None:
        """One more checkpoint by name (e.g. the 4B song planner), downloaded once."""
        d = self.ckpt / name
        if (d / ".glore_complete").exists():
            return
        status.update(phase="setup", message=f"Downloading {name} (first time only)")
        code, _ = run([*self.uv, "run", "--no-sync", "acestep-download", "--dir", str(self.ckpt),
                       "--model", name, "--skip-main", "--force"], self.dir, self.env, log)
        if code:
            raise RuntimeError(f"Downloading {name} failed (see log)")
        (d / ".glore_complete").write_text(str(now()))

    def ensure_models(self, variant: str, log, status: Status) -> None:
        # A marker, not "folder isn't empty": a download cut short (e.g. a full disk) leaves a partial folder
        need = [VARIANT_DIRS[variant], "vae"]
        if all((self.ckpt / n / ".glore_complete").exists() for n in need):
            return
        status.update(phase="setup", message="Downloading ACE-Step model weights (first run only, ~8 GB)")
        code, _ = run([*self.uv, "run", "--no-sync", "acestep-download", "--dir", str(self.ckpt)],
                      self.dir, self.env, log)
        if code:
            raise RuntimeError("Model download failed (see log)")
        if variant != "turbo":
            # --force: ACE skips any folder that exists, even one a full disk left half-downloaded;
            # forced, the Hugging Face download resumes and checks what's already there
            code, _ = run([*self.uv, "run", "--no-sync", "acestep-download", "--dir", str(self.ckpt),
                           "--model", VARIANT_DIRS[variant], "--skip-main", "--force"], self.dir, self.env, log)
            if code:
                raise RuntimeError(f"Downloading {VARIANT_DIRS[variant]} failed (see log)")
        for n in need:
            if (self.ckpt / n).is_dir():
                (self.ckpt / n / ".glore_complete").write_text(str(now()))


    # ------------------------------------------------------------ training
    def preprocess(self, dataset_json: Path, tensor_dir: Path, variant: str, log, status: Status,
                   max_duration: float = 240.0) -> None:
        done = tensor_dir / ".glore_done"
        if done.exists():
            status.update(message="Reusing preprocessed tracks")
            return
        if tensor_dir.exists():
            shutil.rmtree(tensor_dir, ignore_errors=True)
        tensor_dir.mkdir(parents=True, exist_ok=True)
        status.update(phase="preprocessing", message="Encoding tracks for training")
        cmd = self.train_py("--plain", "--yes", "fixed",
                            "--checkpoint-dir", str(self.ckpt), "--model-variant", cli_variant(variant),
                            "--preprocess", "--dataset-json", str(dataset_json),
                      "--tensor-output", str(tensor_dir), "--max-duration", str(max_duration),
                      # required by the parser even though preprocessing ignores them
                      "--dataset-dir", str(tensor_dir), "--output-dir", str(tensor_dir / "_unused"))
        code, _ = run(cmd, self.workdir(dataset_json, tensor_dir), self.env, log)
        if code or not any(tensor_dir.glob("*.pt")):
            raise RuntimeError("Preprocessing failed (see log)")
        done.write_text(str(now()))

    def autolabel(self, dataset_json: Path, out_json: Path, cache_json: Path, log, status: Status,
                  genre_ratio: int = 30) -> None:
        """Have ACE-Step's song planner describe every track (glore_label.py, cached per track)."""
        status.update(phase="labelling", message="Describing your tracks")
        job_path = out_json.with_name("label_job.json")
        job_path.write_text(json.dumps({"ace_dir": str(self.dir), "dataset_json": str(dataset_json),
                                        "out_json": str(out_json), "cache_json": str(cache_json),
                                        "genre_ratio": genre_ratio}), encoding="utf-8")

        def on_line(line):
            if line.startswith("GLORE_PROGRESS "):
                _, p, msg = line.split(" ", 2)
                status.update(message=msg, label_progress=float(p))

        script = Path(__file__).with_name("glore_label.py")
        code, _ = run(self.py(str(script), str(job_path)), self.dir, self.env, log, on_line=on_line)
        if code or not out_json.exists():
            raise RuntimeError("Describing the tracks failed (see log)")

    def train_cmd(self, tensor_dir: Path, out_dir: Path, variant: str, p: dict) -> list[str]:
        inf = VARIANT_INFERENCE[variant]
        cmd = self.train_py("--plain", "--yes", "fixed",
                            "--checkpoint-dir", str(self.ckpt), "--model-variant", cli_variant(variant),
                            "--dataset-dir", str(tensor_dir), "--output-dir", str(out_dir),
                      "--epochs", str(p["epochs"]), "--lr", str(p.get("lr", 1e-4)),
                      "--rank", str(p.get("rank", 64)), "--alpha", str(p.get("alpha", 128)),
                      "--batch-size", str(p.get("batch_size", 1)),
                      "--gradient-accumulation", str(p.get("grad_accum", 4)),
                      "--save-every", str(p.get("save_every", 10)),
                      "--log-every", str(p.get("log_every", 10)),
                      "--optimizer-type", p.get("optimizer", "adamw"),
                      "--shift", str(inf["shift"]), "--num-inference-steps", str(inf["steps"]))
        cmd.append("--offload-encoder" if p.get("offload_encoder") else "--no-offload-encoder")
        resume = latest_checkpoint(out_dir)
        if resume:
            cmd += ["--resume-from", str(resume)]
        return cmd


def checkpoint_epoch(name: str) -> int | None:
    """Side-Step names checkpoints epoch_<n>_loss_<x> (older builds: epoch_<n>)."""
    m = re.match(r"epoch_(\d+)", name)
    return int(m.group(1)) if m else None


def latest_checkpoint(out_dir: Path) -> Path | None:
    best, best_n = None, -1
    for d in (out_dir / "checkpoints").glob("epoch_*"):
        n = checkpoint_epoch(d.name)
        if n is not None and n > best_n and (d / "training_state.pt").exists():
            best, best_n = d, n
    return best


def list_checkpoints(out_dir: Path) -> list[dict]:
    """Every saved checkpoint with an adapter, oldest first: [{epoch, dir}]."""
    found = []
    for d in (out_dir / "checkpoints").glob("epoch_*"):
        n = checkpoint_epoch(d.name)
        if n is not None and (d / "adapter_model.safetensors").exists():
            found.append({"epoch": n, "dir": d.name})
    return sorted(found, key=lambda c: c["epoch"])


# ---------------------------------------------------------------- jobs

def run_train_job(job: dict, root: Path, ace: Ace, log, status: Status, should_stop) -> dict:
    variant = job.get("variant", "turbo")
    ace.ensure(variant, log, status)
    ds_json = root / job["dataset_dir"] / "dataset.json"
    tensor_dir = root / job["tensor_dir"]
    out_dir = root / job["output_dir"]
    labels_rel = None
    if job.get("autolabel"):
        audio = (json.loads(ds_json.read_text(encoding="utf-8")).get("metadata") or {}).get("audio", "prepared")
        labels_rel = f"{job['dataset_dir']}/labels_{audio}.json"   # full and vocal-stripped audio describe differently
        labelled = ds_json.with_name("dataset_labelled.json")
        ace.autolabel(ds_json, labelled, root / labels_rel, log, status, int(job.get("genre_ratio", 30)))
        ds_json = labelled
    ace.preprocess(ds_json, tensor_dir, variant, log, status, job.get("max_duration", 240.0))

    epochs = int(job["params"]["epochs"])
    status.update(phase="training", message="Loading the model", epochs=epochs, epoch=0)
    t_train = now()

    def on_line(line):
        p = parse_progress(line)
        if p:
            frac = p["epoch"] / max(1, p["epochs"])
            el = now() - t_train
            eta = el / frac * (1 - frac) if frac > 0.02 else None
            status.update(epoch=p["epoch"], epochs=p["epochs"], loss=p["loss"], progress=round(frac, 4),
                          eta_s=round(eta) if eta else None,
                          message=f"Epoch {p['epoch']}/{p['epochs']} · loss {p['loss']:.4f}")
        elif OOM_RE.search(line):
            status.update(oom=True)

    code, stopped = run(ace.train_cmd(tensor_dir, out_dir, variant, job["params"]),
                        ace.workdir(tensor_dir, out_dir), ace.env, log, on_line=on_line, should_stop=should_stop)
    final = out_dir / "final"
    if stopped:
        raise StoppedError("Stopped before the end. Checkpoints are kept, so Train again resumes.")
    if code or not (final / "adapter_model.safetensors").exists():
        raise RuntimeError("Out of GPU memory" if status.data.get("oom") else "Training failed (see log)")
    return {"lora_dir": job["output_dir"] + "/final", "epochs": epochs, "loss": status.data.get("loss"),
            "checkpoints": [{**c, "dir": f"{job['output_dir']}/checkpoints/{c['dir']}"}
                            for c in list_checkpoints(out_dir)],
            "labels": labels_rel,
            "variant": variant, "base_model": VARIANT_DIRS[variant], "inference": VARIANT_INFERENCE[variant],
            "train_seconds": round(now() - t_train)}


class StoppedError(RuntimeError):
    pass


def run_test_job(job: dict, root: Path) -> dict:
    ping = root / "glore" / "test" / "ping.txt"
    return {"gpu": gpu_info(), "read_ok": ping.exists() and ping.read_text().strip() == job.get("token"),
            "python": sys.version.split()[0]}


def run_fit_test(ace: Ace, dataset_json: Path, tensor_dir: Path, out_dir: Path, log, status: Status,
                 train_seconds: float = 300.0) -> dict:
    """Local 8 GB check: tight settings, a few tracks, train for 5 minutes, watch VRAM."""
    variant = "turbo"
    ace.ensure(variant, log, status, extras=("bitsandbytes>=0.45",))
    ace.preprocess(dataset_json, tensor_dir, variant, log, status, max_duration=120.0)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    params = {"epochs": 1000, "rank": 16, "alpha": 32, "optimizer": "adamw8bit", "offload_encoder": True,
              "grad_accum": 4, "log_every": 1, "save_every": 100000}
    peak = {"mb": 0, "total": 0}
    stop = threading.Event()

    def sample_vram():
        while not stop.is_set():
            g = gpu_info()
            if g:
                peak["mb"] = max(peak["mb"], g["used_mb"])
                peak["total"] = g["total_mb"]
            stop.wait(2)

    threading.Thread(target=sample_vram, daemon=True).start()
    steps, oom, t_first = [], False, None
    t0 = now()

    def on_line(line):
        nonlocal oom, t_first
        p = parse_progress(line)
        if p:
            t_first = t_first or now()
            steps.append(now())
            status.update(phase="training", progress=min(0.99, (now() - t0) / train_seconds),
                          message=f"Training · {len(steps)} steps · {peak['mb'] / 1024:.1f} GB used")
        elif OOM_RE.search(line):
            oom = True

    status.update(phase="training", message="Loading the model on your GPU")
    run(ace.train_cmd(tensor_dir, out_dir, variant, params), ace.workdir(tensor_dir, out_dir), ace.env, log,
        on_line=on_line, timeout=train_seconds)
    stop.set()
    sec_per_step = (steps[-1] - steps[0]) / (len(steps) - 1) if len(steps) > 1 else None
    verdict = "no" if oom else ("yes" if steps else "unclear")
    return {"fits": verdict, "steps": len(steps), "sec_per_step": round(sec_per_step, 2) if sec_per_step else None,
            "peak_vram_gb": round(peak["mb"] / 1024, 2), "total_vram_gb": round(peak["total"] / 1024, 2),
            "settings": params}


# ---------------------------------------------------------------- pod main loop

def terminate_self(log) -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("GLORE_RUNPOD_KEY") or os.environ.get("RUNPOD_API_KEY")
    log.write(f"\n[glore] shutting down pod {pod_id}\n")
    log.flush()
    if not (pod_id and key):
        return
    for method, url, body in (("POST", f"{RUNPOD_API}/pods/{pod_id}/action", {"action": "terminate"}),
                              ("DELETE", f"{RUNPOD_API}/pods/{pod_id}", None)):
        try:
            req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30)
            return
        except Exception as e:  # try the next way
            log.write(f"[glore] {method} failed: {e}\n")


# ---------------------------------------------------------------- generation on the pod

def run_generate_job(job: dict, root: Path, ace: Ace, log, status: Status) -> dict:
    """Render takes on the cloud GPU (no PC needed). job["takes"]: [{name, caption, lora_dir?, scale?,
    seed, duration, bpm, key, rewrite_caption?, velocity_norm?}]. MP3s land in outputs/<job id>/."""
    variant = job.get("variant", "xl_sft")
    ace.ensure(variant, log, status)
    lm = job.get("lm", CLOUD_LM)
    if lm:
        ace.ensure_extra_model(lm, log, status)
    inference = VARIANT_INFERENCE[variant]
    out_dir = root / "outputs" / job["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("glore_generate.py")
    takes = job.get("takes") or []
    results = []
    for i, t in enumerate(takes):
        name = re.sub(r"[^A-Za-z0-9 ._()+-]", "", t.get("name") or f"take {i + 1}").strip() or f"take {i + 1}"
        status.update(phase="generating", progress=0.1 + 0.85 * i / max(1, len(takes)),
                      message=f"Rendering {name} ({i + 1}/{len(takes)})")
        work = root / "glore" / "runs" / job["id"] / f"take{i + 1}"
        work.mkdir(parents=True, exist_ok=True)
        lora = None
        if t.get("lora_dir"):
            lora = {"path": str(root / t["lora_dir"]), "trigger": t.get("trigger"), "scale": float(t.get("scale", 1.0))}
        gjob = {"ace_dir": str(ace.dir), "out_dir": str(work), "variant": variant, "caption": t.get("caption", ""),
                "lora": lora, "bpm": t.get("bpm"), "key": t.get("key"), "duration": float(t.get("duration", 30)),
                "seed": int(t.get("seed", -1)), "count": 1, "steps": inference["steps"], "shift": inference["shift"],
                "offload": False, "lm": lm, "rewrite_caption": bool(t.get("rewrite_caption", True)),
                "velocity_norm": float(t.get("velocity_norm", 0.0))}
        (work / "job.json").write_text(json.dumps(gjob, indent=2), encoding="utf-8")
        found = {}

        def on_line(line, found=found):
            if line.startswith("GLORE_RESULT "):
                found.update(json.loads(line[len("GLORE_RESULT "):]))
            elif line.startswith("GLORE_ERROR "):
                found["error"] = json.loads(line[len("GLORE_ERROR "):]).get("error")

        code, _ = run(ace.py(str(script), str(work / "job.json")), ace.dir, ace.env, log, on_line=on_line)
        files = found.get("files") or []
        if code or not files:
            results.append({"name": name, "error": found.get("error") or "generation failed (see log)"})
            continue
        mp3 = out_dir / f"{name}.mp3"
        code, _ = run([*FFMPEG, "-y", "-v", "error", "-i", files[0]["path"], "-c:a", "libmp3lame", "-b:a", "256k",
                       str(mp3)], ace.dir, ace.env, log)
        if code or not mp3.exists():
            raise RuntimeError("Converting the take to MP3 failed (see log)")
        results.append({"name": name, "mp3": f"outputs/{job['id']}/{mp3.name}", "seed": files[0].get("seed"),
                        "planned_caption": found.get("planned_caption"), "seconds": found.get("seconds")})
        shutil.rmtree(work, ignore_errors=True)
    if takes and not any("mp3" in r for r in results):
        raise RuntimeError(results[0].get("error") if results else "nothing rendered")
    return {"takes": results, "variant": variant, "lm": lm}


# ---------------------------------------------------------------- songs: your vocal + a beat built around it

SONG_EXTRAS = ("demucs==4.0.1", "librosa>=0.10.2")


def _run_tool(ace: Ace, script: str, args: list[str], log) -> dict:
    """Run one of GLORE's scripts in ACE's venv; returns its GLORE_RESULT (raises on GLORE_ERROR)."""
    found: dict = {}

    def on_line(line):
        if line.startswith("GLORE_RESULT "):
            found.update(json.loads(line[len("GLORE_RESULT "):]))
        elif line.startswith("GLORE_ERROR "):
            found["_error"] = json.loads(line[len("GLORE_ERROR "):]).get("error")

    code, _ = run(ace.py(str(Path(__file__).with_name(script)), *args), ace.dir, ace.env, log, on_line=on_line)
    if code or "_error" in found or not found:
        raise RuntimeError(found.get("_error") or f"{script} failed (see log)")
    return found


def run_song_job(job: dict, root: Path, ace: Ace, log, status: Status) -> dict:
    """job: {id, upload (volume key), kind: vocal|song, variant, bpm?, key?,
             takes: [{name, caption, lora_dir?, trigger?, scale?, seed, tracks?, guidance?}]}
    Each take: ACE 'complete' writes the backing while hearing your vocal (base/SFT models),
    then the model's own sung copy is separated out and your real vocal goes on top."""
    variant = job.get("variant", "xl_sft")
    ace.ensure(variant, log, status, extras=SONG_EXTRAS)
    work = root / "glore" / "runs" / job["id"] / "song"
    work.mkdir(parents=True, exist_ok=True)
    status.update(phase="preparing", progress=0.05, message="Finding the vocal, BPM and key")
    pj = work / "prep.json"
    pj.write_text(json.dumps({"input": str(root / job["upload"]), "kind": job.get("kind", "vocal"),
                              "work": str(work / "prep")}), encoding="utf-8")
    a = _run_tool(ace, "glore_song.py", ["prep", str(pj)], log)
    bpm, key = job.get("bpm") or a["bpm"], job.get("key") or a["key"]
    inference = VARIANT_INFERENCE[variant]
    out_dir = root / "outputs" / job["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    takes, results = job.get("takes") or [], []
    for i, t in enumerate(takes):
        name = re.sub(r"[^A-Za-z0-9 ._()+-]", "", t.get("name") or f"take {i + 1}").strip() or f"take {i + 1}"
        status.update(phase="generating", progress=0.15 + 0.8 * i / max(1, len(takes)),
                      message=f"Building {name} around your vocal ({i + 1}/{len(takes)})")
        tw = work / f"take{i + 1}"
        tw.mkdir(parents=True, exist_ok=True)
        lora = ({"path": str(root / t["lora_dir"]), "trigger": t.get("trigger"), "scale": float(t.get("scale", 1.0))}
                if t.get("lora_dir") else None)
        gjob = {"ace_dir": str(ace.dir), "out_dir": str(tw), "variant": variant, "caption": t.get("caption", ""),
                "lora": lora, "bpm": bpm, "key": key, "duration": a["duration"], "seed": int(t.get("seed", -1)),
                "count": 1, "steps": inference["steps"], "shift": inference["shift"], "offload": False, "lm": None,
                "mode": "complete", "src_audio": a["vocal"], "tracks": t.get("tracks") or ["drums", "bass", "keyboard", "synth"],
                "guidance": float(t.get("guidance", 7.0)), "rewrite_caption": False}
        (tw / "job.json").write_text(json.dumps(gjob), encoding="utf-8")
        try:
            g = _run_tool(ace, "glore_generate.py", [str(tw / "job.json")], log)
            mj = tw / "mix.json"
            mj.write_text(json.dumps({"take": g["files"][0]["path"], "vocal": a["vocal"], "work": str(tw / "mix"),
                                      "song_mp3": str(out_dir / f"{name}.mp3"),
                                      "beat_mp3": str(out_dir / f"{name} (beat).mp3"),
                                      "vocal_gain_db": t.get("vocal_gain_db", -2), "beat_gain_db": t.get("beat_gain_db", 0)}),
                          encoding="utf-8")
            _run_tool(ace, "glore_song.py", ["mix", str(mj)], log)
            results.append({"name": name, "mp3": f"outputs/{job['id']}/{name}.mp3",
                            "beat_mp3": f"outputs/{job['id']}/{name} (beat).mp3", "seed": g["files"][0].get("seed"),
                            "caption": t.get("caption")})
        except RuntimeError as e:
            results.append({"name": name, "error": str(e)})
        shutil.rmtree(tw, ignore_errors=True)
    if takes and not any("mp3" in r for r in results):
        raise RuntimeError(results[0].get("error") or "nothing rendered")
    return {"takes": results, "variant": variant, "bpm": bpm, "key": key, "duration": a["duration"],
            "kind": job.get("kind", "vocal")}


def next_job(queue: Path) -> Path | None:
    jobs = sorted(queue.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return jobs[0] if jobs else None


def pod_main(root: Path = Path("/workspace")) -> None:
    idle_s = float(os.environ.get("GLORE_IDLE_MIN", "10")) * 60
    hard_cap = float(os.environ.get("GLORE_HARD_CAP_S", str(4 * 3600)))
    started = now()
    queue = root / "glore" / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    pod_log = open(root / "glore" / f"pod-{os.environ.get('RUNPOD_POD_ID', 'local')}.log", "a", encoding="utf-8")
    env = dict(os.environ)
    uv = [os.environ.get("GLORE_UV", "uv")]
    ace = Ace(root / "ace", uv, env)
    state = {"idle_since": now(), "busy": False, "status": None, "stop": False}

    def watchdog():
        while True:
            t = now()
            reason = None
            if t - started > hard_cap:
                reason = "hard cap reached"
            elif not state["busy"] and t - state["idle_since"] > idle_s:
                reason = f"idle for {idle_s / 60:g} min"
            if reason:
                state["stop"] = True
                st = state["status"]
                if st and st.data.get("phase") not in ("done", "error", "stopped"):
                    st.update(phase="stopped", message=f"Cloud GPU stopped: {reason}")
                    time.sleep(10)  # give the trainer a moment to exit
                terminate_self(pod_log)
                time.sleep(120)
                os._exit(0)
            if state["status"] and state["busy"]:
                state["status"].update()  # heartbeat
            try:  # pod-level heartbeat GLORE's safety net watches, busy or idle
                write_json(root / "glore" / "pods" / f"{os.environ.get('RUNPOD_POD_ID', 'local')}.json",
                           {"heartbeat": t, "busy": state["busy"], "started": started,
                            "job": state["status"].data.get("job") if state["status"] else None})
            except OSError:
                pass
            time.sleep(15)

    threading.Thread(target=watchdog, daemon=True).start()
    while True:
        path = next_job(queue)
        if not path:
            time.sleep(10)
            continue
        process_job(path, root, ace, state)


def process_job(path: Path, root: Path, ace: Ace, state: dict) -> dict | None:
    """Run one queued job. state carries busy / idle_since / status / stop for the watchdog."""
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    run_dir = root / "glore" / "runs" / job["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    os.replace(path, run_dir / "job.json")
    status = Status(run_dir / "status.json", job["id"], job["mode"])
    status.update(gpu=gpu_info().get("name"), pod=os.environ.get("RUNPOD_POD_ID"))
    state.update(busy=True, status=status)
    with open(run_dir / "log.txt", "a", encoding="utf-8") as log:
        try:
            if job["mode"] == "test":
                result = run_test_job(job, root)
            elif job["mode"] == "generate":
                result = run_generate_job(job, root, ace, log, status)
            elif job["mode"] == "song":
                result = run_song_job(job, root, ace, log, status)
            else:
                result = run_train_job(job, root, ace, log, status, lambda: state.get("stop"))
            status.update(phase="done", progress=1.0, message="Done", result=result)
        except StoppedError as e:
            status.update(phase="stopped", message=str(e))
        except Exception as e:
            log.write(f"\n[glore] error: {e}\n")
            status.update(phase="error", message=str(e), error=str(e))
    state.update(busy=False, idle_since=now())
    return status.data

if __name__ == "__main__":
    pod_main(Path(os.environ.get("GLORE_ROOT", "/workspace")))
