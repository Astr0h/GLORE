"""LoRA training: cloud runs on RunPod, plus the local "will it fit on 8 GB?" test.

Cloud run, start to finish:
  1. Prepared tracks (_train.wav) -> 24-bit FLAC (lossless, about half the size).
  2. Upload to the network volume over S3. Files already there with the same size
     are skipped, so an interrupted upload picks up where it stopped.
  3. Queue a job file, start a GPU pod (or reuse one that's idling).
  4. Follow status.json on the volume: epochs, loss, ETA, cost so far.
  5. Download the LoRA to %APPDATA%/GLORE/loras/<pack>/<run>/.

Same tracks + same settings = same run id, so a run cut short by the hard cap
resumes from its last checkpoint when you press Train again."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import soundfile as sf

from . import cloud, datasets
from .cloud import CloudError, CloudManager
from .pod import glore_pod
from .project import PACKS

POD_DIR = Path(__file__).parent / "pod"
POLL_S = 10.0       # how often a running job checks the cloud (tests make this tiny)
Report = Callable[[float, str], None]


def glore_home() -> Path:
    base = os.environ.get("GLORE_HOME") or os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "GLORE"


def trigger(pack: str) -> str:
    return f"glore_{pack}"


# ---------------------------------------------------------------- local LoRA library

def lora_root() -> Path:
    d = glore_home() / "loras"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_loras(pack: str | None = None) -> list[dict]:
    out = []
    for m in sorted(lora_root().glob("*/*/glore_lora.json")):
        try:
            info = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if pack and info.get("pack") != pack:
            continue
        info["path"] = str(m.parent)
        out.append(info)
    # newest first; a run's own checkpoints (run_id@e<n>) sort after its final LoRA
    return sorted(out, key=lambda i: (i.get("trained_at", 0), not i.get("checkpoint_of")), reverse=True)


# ---------------------------------------------------------------- dataset bundle

@dataclass
class Bundle:
    pack: str
    files: list[tuple[Path, str]]  # (local FLAC, key on the volume)
    dataset_json: bytes
    hash: str

    @property
    def count(self) -> int:
        return len(self.files)


def to_flac(wav: Path, flac: Path) -> Path:
    if flac.exists() and flac.stat().st_mtime >= wav.stat().st_mtime:
        return flac
    flac.parent.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    tmp = flac.with_suffix(".tmp.flac")
    sf.write(str(tmp), data, sr, format="FLAC", subtype="PCM_24")
    os.replace(tmp, flac)
    return flac


def build_bundle(pack: str, report: Report | None = None, limit: int | None = None,
                 audio: str = "prepared", genre_ratio: int = 0) -> Bundle:
    """audio = "prepared" (vocals stripped, the pack's training audio) or "full" (the original
    tracks, vocal chops and all: closer to the real sound, no separation artefacts)."""
    report = report or (lambda p, m: None)
    d = datasets.pack_dir(pack)
    suffix = "" if audio == "full" else "_train"
    items = [it for it in datasets.list_items(pack)
             if (audio == "full" or it.get("prepared")) and (d / f"{it['id']}{suffix}.wav").exists()]
    if limit:
        items = items[:limit]
    files, samples = [], []
    label = PACKS[pack]["label"]
    for i, it in enumerate(items):
        report(i / max(1, len(items)), f"Compressing {i + 1}/{len(items)}: {it.get('name', it['id'])}")
        fname = f"{it['id']}_full.flac" if audio == "full" else f"{it['id']}.flac"  # never share a name
        flac = to_flac(d / f"{it['id']}{suffix}.wav", d / "_cloud" / fname)
        files.append((flac, f"datasets/{pack}/{fname}"))
        bpm = it.get("bpm")
        samples.append({
            "id": it["id"], "audio_path": f"./{fname}", "filename": fname,
            "caption": it.get("caption") or datasets.caption(pack, it.get("moods", []), bpm, it.get("key")),
            "genre": label.lower(), "lyrics": "[Instrumental]", "is_instrumental": True,
            "bpm": int(round(float(bpm))) if bpm else "N/A", "keyscale": it.get("key") or "N/A",
            "timesignature": "4", "language": "unknown", "custom_tag": trigger(pack), "labeled": True,
        })
    doc = {"metadata": {"name": f"glore_{pack}", "custom_tag": trigger(pack), "tag_position": "prepend",
                        "all_instrumental": True, "genre_ratio": genre_ratio, "num_samples": len(samples),
                        "audio": audio},
           "samples": samples}
    raw = json.dumps(doc, indent=2, sort_keys=True).encode()
    h = hashlib.sha1(raw)
    for f, _ in files:
        h.update(str(f.stat().st_size).encode())
    return Bundle(pack, files, raw, h.hexdigest()[:12])


def upload_bundle(vol: cloud.Volume, b: Bundle, report: Report) -> dict:
    total = sum(f.stat().st_size for f, _ in b.files) or 1
    done = sent = skipped = 0
    for i, (f, key) in enumerate(b.files):
        size = f.stat().st_size
        if vol.size(key) == size:
            skipped += 1
        else:
            moved = [0]

            def cb(n, i=i):
                moved[0] += n
                report((done + moved[0]) / total,
                       f"Uploading {i + 1}/{b.count} · {(done + moved[0]) / 2 ** 20:.0f} of {total / 2 ** 20:.0f} MB")

            vol.put_file(key, f, cb)
            sent += 1
        done += size
        report(done / total, f"Uploading {i + 1}/{b.count} · {done / 2 ** 20:.0f} of {total / 2 ** 20:.0f} MB")
    vol.put_bytes(f"datasets/{b.pack}/dataset.json", b.dataset_json)
    return {"uploaded": sent, "already_there": skipped, "mb": round(total / 2 ** 20)}


def sync_scripts(vol: cloud.Volume) -> None:
    for name in ("glore_pod.py", "glore_label.py", "glore_generate.py", "pod_start.sh"):
        data = (POD_DIR / name).read_bytes().replace(b"\r\n", b"\n")
        vol.put_bytes(f"glore/{name}", data)


def train_params(n_tracks: int, epochs: int | None = None, rank: int | None = None,
                 save_every: int | None = None) -> dict:
    return glore_pod.train_params(n_tracks, epochs, rank, save_every)


# ---------------------------------------------------------------- runs (survive an app restart)

def _runs_path() -> Path:
    return cloud.home() / "runs.json"


def load_runs() -> dict:
    try:
        return json.loads(_runs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_run(job_id: str, info: dict | None) -> None:
    runs = load_runs()
    if info is None:
        runs.pop(job_id, None)
    else:
        runs[job_id] = {**runs.get(job_id, {}), **info}
    _runs_path().write_text(json.dumps(runs, indent=2), encoding="utf-8")


def pending_runs() -> dict:
    return {k: v for k, v in load_runs().items() if v.get("state") == "running"}


# ---------------------------------------------------------------- cloud training

def train_cloud(pack: str, mgr: CloudManager, report: Report, epochs: int | None = None,
                rank: int | None = None, audio: str = "prepared", autolabel: bool = False,
                save_every: int | None = None) -> dict:
    s = mgr.settings()
    if cloud.missing(s):
        raise CloudError("Finish the cloud settings first")
    prepared = [it for it in datasets.list_items(pack) if audio == "full" or it.get("prepared")]
    if len(prepared) < datasets.MIN_TRACKS:
        raise CloudError(f"Prepare at least {datasets.MIN_TRACKS} tracks first ({len(prepared)} prepared)")
    rp, vol = mgr.rp(), mgr.volume()
    hard_cap = float(s["hard_cap_hours"]) * 3600

    reuse = mgr.reusable_pod("train")
    if not reuse:
        report(0.01, f"Finding a free {s['gpu']}")
        choices = cloud.gpu_choices(rp.gpus(), s["datacenter"], s["gpu"])
        mgr.check_budget(max(p for _, p in choices[:4]) * hard_cap / 3600)

    b = build_bundle(pack, lambda p, m: report(0.02 + 0.08 * p, m), audio=audio, genre_ratio=30 if autolabel else 0)
    params = train_params(b.count, epochs, rank, save_every)
    run_id = (f"{b.hash}-{s['variant']}-r{params['rank']}-e{params['epochs']}"
              + ("-full" if audio == "full" else "") + ("-cap" if autolabel else ""))
    if any(l.get("run_id") == run_id for l in list_loras(pack)):
        raise CloudError("This pack is already trained with these exact tracks and settings")

    up = upload_bundle(vol, b, lambda p, m: report(0.10 + 0.28 * p, m))
    sync_scripts(vol)

    if not reuse:  # nothing is running, so anything still queued is left over from a failed start
        for key in vol.list_keys("glore/queue/"):
            vol.delete(key)
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "mode": "train", "pack": pack, "run_id": run_id, "variant": s["variant"],
           "dataset_dir": f"datasets/{pack}", "tensor_dir": f"tensors/{pack}/{b.hash}-{s['variant']}",
           "output_dir": f"loras/{pack}/{run_id}", "params": params, "max_duration": 240.0,
           "autolabel": autolabel, "genre_ratio": 30}
    vol.put_bytes(f"glore/queue/{job_id}.json", json.dumps(job).encode())

    if reuse:
        row = reuse
        report(0.40, "Sending the job to the cloud GPU that's already running")
    else:
        try:
            row = mgr.start_pod_any("train", choices[:4], hard_cap, float(s["idle_minutes"]),
                                    meta={"pack": pack, "run_id": run_id}, report=lambda m: report(0.40, m))
        except CloudError:
            vol.delete(f"glore/queue/{job_id}.json")
            raise
    meta = {"pack": pack, "run_id": run_id, "pod_id": row["pod_id"], "state": "running", "tracks": b.count,
            "params": params, "variant": s["variant"], "upload": up, "started": time.time()}
    save_run(job_id, meta)
    return follow(job_id, mgr, report)


def follow(job_id: str, mgr: CloudManager, report: Report, poll_s: float | None = None) -> dict:
    """Watch a queued job until it finishes, then bring the LoRA home."""
    poll_s = poll_s or POLL_S
    run = load_runs()[job_id]
    pid = run["pod_id"]
    vol = mgr.volume()
    last_hb, gone_at = None, None
    read_errors = 0
    while True:
        try:
            st = vol.get_json(f"glore/runs/{job_id}/status.json")
            read_errors = 0
        except CloudError:
            # a blip reading the status file must not abandon a run that is still training
            read_errors += 1
            if read_errors > 30:
                raise
            time.sleep(poll_s)
            continue
        row = next((r for r in mgr.ledger.rows() if r["pod_id"] == pid), {})
        cost = cloud.Ledger.cost(row) if row else 0.0
        if st:
            if st.get("heartbeat") != last_hb:
                last_hb = st.get("heartbeat")
                mgr.heartbeats[pid] = time.time()
            phase = st.get("phase")
            if phase == "done":
                break
            if phase in ("error", "stopped"):
                save_run(job_id, {"state": phase, "message": st.get("message")})
                tail = vol.tail(f"glore/runs/{job_id}/log.txt", 1500)
                _save_log(job_id, tail)
                raise CloudError(st.get("message") or "Training failed")
            report(_progress(st), _message(st, cost))
        else:
            report(0.42, f"Starting the cloud GPU · ${cost:.2f} so far")
        if row.get("ended"):
            gone_at = gone_at or time.time()
            if time.time() - gone_at > 3 * poll_s:
                save_run(job_id, {"state": "error", "message": row.get("end_reason")})
                raise CloudError(f"The cloud GPU stopped ({row.get('end_reason', 'unknown reason')}) before "
                                 "training finished. Press Train again to resume from the last checkpoint.")
        time.sleep(poll_s)

    report(0.97, "Downloading the LoRA")
    res = st.get("result") or {}
    dest = lora_root() / run["pack"] / run["run_id"]
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        vol.download(f"{res.get('lora_dir', run['run_id'])}/{name}", dest / name)
    captions = None
    if res.get("labels"):
        try:
            vol.download(res["labels"], dest / "captions.json")
            captions = str(dest / "captions.json")
        except CloudError:
            pass
    info = {"pack": run["pack"], "label": PACKS[run["pack"]]["label"], "run_id": run["run_id"],
            "trained_at": time.time(), "tracks": run["tracks"], "epochs": res.get("epochs"),
            "rank": run["params"]["rank"], "loss": res.get("loss"), "train_seconds": res.get("train_seconds"),
            "model": "ACE-Step 1.5", "size": "standard", "variant": run["variant"],
            "base_model": res.get("base_model") or glore_pod.VARIANT_DIRS[run["variant"]],
            "inference": res.get("inference") or glore_pod.VARIANT_INFERENCE[run["variant"]],
            "trigger": trigger(run["pack"]), "source": "glore-cloud", "captions": captions}
    (dest / "glore_lora.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    # Every saved checkpoint too, as its own LoRA (run_id@e<epoch>), so they can be compared by ear
    finals = res.get("epochs")
    for c in res.get("checkpoints") or []:
        if c["epoch"] == finals:
            continue
        cdest = lora_root() / run["pack"] / f"{run['run_id']}@e{c['epoch']}"
        report(0.98, f"Downloading the epoch {c['epoch']} checkpoint")
        try:
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                vol.download(f"{c['dir']}/{name}", cdest / name)
        except CloudError:
            continue
        (cdest / "glore_lora.json").write_text(json.dumps(
            {**info, "run_id": f"{run['run_id']}@e{c['epoch']}", "epochs": c["epoch"], "loss": None,
             "checkpoint_of": run["run_id"]}, indent=2), encoding="utf-8")
    save_run(job_id, {"state": "done", "lora": str(dest)})
    report(1.0, "LoRA ready")
    return info


def _progress(st: dict) -> float:
    phase = st.get("phase")
    if phase == "setup":
        return 0.45
    if phase == "labelling":
        return 0.46
    if phase == "preprocessing":
        return 0.48
    if phase == "training":
        return 0.50 + 0.46 * float(st.get("progress") or 0)
    return 0.43


def _message(st: dict, cost: float) -> str:
    msg = st.get("message") or st.get("phase", "")
    eta = st.get("eta_s")
    if eta:
        msg += f" · ~{_dur(eta)} left"
    return f"{msg} · ${cost:.2f} so far"


def _dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{max(1, s // 60)} min"


def _save_log(job_id: str, text: str) -> None:
    if text:
        (cloud.home() / f"log-{job_id}.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- connection test (~2 min, ~1 cent)

def connection_test(mgr: CloudManager, report: Report, poll_s: float | None = None) -> dict:
    poll_s = poll_s or POLL_S / 2
    s = mgr.settings()
    if cloud.missing(s):
        raise CloudError("Finish the cloud settings first")
    t0 = time.time()
    report(0.05, "Checking the API key")
    rp = mgr.rp()
    gpus = rp.gpus()
    report(0.15, "Checking cloud storage")
    vol = mgr.volume()
    token = uuid.uuid4().hex
    vol.put_bytes("glore/test/ping.txt", token.encode())
    if (vol.get_bytes("glore/test/ping.txt") or b"").decode() != token:
        raise CloudError("Wrote a test file to the network volume but couldn't read it back")
    sync_scripts(vol)
    gpu_id, price = cloud.pick_cheapest(gpus, s["datacenter"])
    mgr.check_budget(price * cloud.TEST_HARD_CAP_S / 3600)
    job_id = uuid.uuid4().hex[:10]
    vol.put_bytes(f"glore/queue/{job_id}.json", json.dumps({"id": job_id, "mode": "test", "token": token}).encode())
    report(0.25, f"Starting a test pod ({gpu_id}, ${price:.2f}/h)")
    try:
        row = mgr.start_pod("test", gpu_id, price, cloud.TEST_HARD_CAP_S, 0)
    except CloudError:
        vol.delete(f"glore/queue/{job_id}.json")
        raise
    pid = row["pod_id"]
    try:
        while True:
            el = time.time() - row["started"]
            st = vol.get_json(f"glore/runs/{job_id}/status.json")
            if st and st.get("phase") == "done":
                break
            if st and st.get("phase") == "error":
                raise CloudError(st.get("message") or "The test pod hit an error")
            cur = next((r for r in mgr.ledger.rows() if r["pod_id"] == pid), {})
            if cur.get("ended") or el > cloud.TEST_HARD_CAP_S + 30:
                raise CloudError("The test pod stopped before it reported back. "
                                 f"Pod log: glore/pod-{pid}.log on the volume.")
            report(min(0.95, 0.25 + 0.7 * el / 150), f"Waiting for the test pod · {int(el)} s")
            time.sleep(poll_s)
    finally:
        try:
            mgr.stop(pid, "test finished")
        except CloudError:
            pass
    res = st.get("result") or {}
    if not res.get("read_ok"):
        raise CloudError("The pod started but couldn't see the network volume")
    cur = next((r for r in mgr.ledger.rows() if r["pod_id"] == pid), row)
    return {"ok": True, "gpu": (res.get("gpu") or {}).get("name") or gpu_id, "seconds": round(time.time() - t0),
            "cost": round(cloud.Ledger.cost(cur), 4), "datacenter": s["datacenter"]}


def setup_volume(mgr: CloudManager, report: Report) -> dict:
    s = mgr.settings()
    rp = mgr.rp()
    dc = s.get("datacenter") or ""
    if not dc:
        report(0.3, f"Finding a datacenter with a free {s['gpu']} and S3 storage")
        dc = cloud.choose_datacenter(rp, s["gpu"])
    report(0.7, f"Creating a {s['volume_gb']} GB network volume in {dc}")
    v = rp.create_volume("glore", int(s["volume_gb"]), dc)
    if not v.get("id"):
        raise CloudError("RunPod didn't return a volume id")
    cloud.save_settings({"datacenter": dc, "volume_id": v["id"]})
    return {"volume_id": v["id"], "datacenter": dc}


# ---------------------------------------------------------------- local 8 GB fit test

class _ReportStatus(glore_pod.Status):
    def __init__(self, path: Path, report: Report, base: float, span: float):
        self.report, self.base, self.span = report, base, span
        super().__init__(path, "fit-test", "fit")

    def flush(self) -> None:
        super().flush()
        if getattr(self, "report", None):
            p = float(self.data.get("progress") or 0) if self.data.get("phase") == "training" else 0
            self.report(self.base + self.span * p, self.data.get("message", ""))


def fit_test(pack: str, report: Report, train_seconds: float = 300.0) -> dict:
    """Train on 3 of your tracks on this PC for 5 minutes with low-VRAM settings."""
    prepared = [it for it in datasets.list_items(pack) if it.get("prepared")]
    if len(prepared) < 3:
        raise CloudError("Prepare at least 3 tracks in this pack first")
    root = glore_home() / "fit-test"
    data = root / "data"
    if data.exists():
        shutil.rmtree(data, ignore_errors=True)
    data.mkdir(parents=True)
    b = build_bundle(pack, lambda p, m: report(0.01 * p, m), limit=3)
    for f, key in b.files:
        shutil.copyfile(f, data / Path(key).name)
    (data / "dataset.json").write_bytes(b.dataset_json)

    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # uv must use ACE-Step's own venv
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONUNBUFFERED"] = "1"
    ace = glore_pod.Ace(glore_home() / "ace", [sys.executable, "-m", "uv"], env)
    status = _ReportStatus(root / "status.json", report, 0.05, 0.93)
    with open(root / "log.txt", "w", encoding="utf-8") as log:
        res = glore_pod.run_fit_test(ace, data / "dataset.json", root / "tensors" / b.hash, root / "out",
                                     log, status, train_seconds)
    res["log"] = str(root / "log.txt")
    n = len(prepared)
    if res.get("sec_per_step"):
        p = train_params(n)
        steps = n * p["epochs"] / p["grad_accum"]
        res["full_run_hours_here"] = round(steps * res["sec_per_step"] / 3600, 1)
    return res
