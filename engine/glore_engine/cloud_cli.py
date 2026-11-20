"""GLORE cloud, without your PC.

Runs anywhere with Python + boto3 (a GitHub Actions runner, a laptop): it talks to RunPod
directly and works on what's already on your network volume (datasets, captions, LoRAs).

    python -m glore_engine.cloud_cli status
    python -m glore_engine.cloud_cli train --pack house --variant xl_sft --epochs 200 --save-every 50
    python -m glore_engine.cloud_cli generate --takes takes.json [--wait --out out/]
    python -m glore_engine.cloud_cli fetch --job <id> --out out/
    python -m glore_engine.cloud_cli loras
    python -m glore_engine.cloud_cli stop

Settings come from environment variables (GitHub repo secrets):
    GLORE_RUNPOD_API_KEY, GLORE_S3_ACCESS_KEY, GLORE_S3_SECRET, GLORE_VOLUME_ID, GLORE_DATACENTER
optional: GLORE_GPU (preference list), GLORE_MONTHLY_LIMIT (USD, default 30)

The spend ledger lives on the volume (glore/ledger-cloud.json) so the monthly limit holds
across runs, and pods still shut themselves down (idle timeout + hard cap) on their own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import cloud
from .cloud import CloudError, CloudManager, Ledger
from .pod import glore_pod

LEDGER_KEY = "glore/ledger-cloud.json"
DEFAULT_GPUS = "A100, RTX 6000 Ada, L40S, RTX 5090"  # no H100: $4.39/h would eat the budget


def env_settings() -> dict:
    s = dict(cloud.DEFAULTS)
    s.update({
        "api_key": os.environ.get("GLORE_RUNPOD_API_KEY", ""),
        "s3_access_key": os.environ.get("GLORE_S3_ACCESS_KEY", ""),
        "s3_secret": os.environ.get("GLORE_S3_SECRET", ""),
        "volume_id": os.environ.get("GLORE_VOLUME_ID", ""),
        "datacenter": os.environ.get("GLORE_DATACENTER", ""),
        "gpu": os.environ.get("GLORE_GPU") or DEFAULT_GPUS,
        "monthly_limit": float(os.environ.get("GLORE_MONTHLY_LIMIT") or 30),
    })
    miss = cloud.missing(s)
    if miss:
        raise CloudError("Missing settings: " + ", ".join(miss) + " (add them as GitHub repo secrets)")
    return s


class VolumeLedger(Ledger):
    """The same ledger, kept on the network volume instead of a PC."""

    def __init__(self, vol: cloud.Volume):
        super().__init__(Path(tempfile.mkdtemp()) / "ledger.json")
        self.vol = vol

    def _load(self) -> list[dict]:
        raw = self.vol.get_bytes(LEDGER_KEY)
        try:
            return json.loads(raw) if raw else []
        except ValueError:
            return []

    def _save(self, rows: list[dict]) -> None:
        self.vol.put_bytes(LEDGER_KEY, json.dumps(rows, indent=2).encode())


def manager(s: dict | None = None) -> CloudManager:
    s = s or env_settings()
    vol = cloud.Volume(s)
    return CloudManager(settings_loader=lambda: s, ledger=VolumeLedger(vol))


def settle_ledger(mgr: CloudManager) -> None:
    """Close rows for pods that are gone (their last heartbeat = when they stopped costing)."""
    rp, vol = mgr.rp(), mgr.volume()
    for r in mgr.ledger.active():
        p = rp.get_pod(r["pod_id"])
        if p and p.get("status") in ("RUNNING", "STARTING", "PROVISIONING", "CREATED"):
            continue
        hb = vol.get_bytes(f"glore/pods/{r['pod_id']}.json")
        try:
            end = float(json.loads(hb)["heartbeat"]) if hb else None
        except (ValueError, KeyError):
            end = None
        cap_end = r["started"] + r.get("hard_cap_s", 4 * 3600)
        mgr.ledger.update(r["pod_id"], ended=min(end or cap_end, cap_end, time.time()))


def _start_or_reuse(mgr: CloudManager, kind: str, hard_cap_s: float, idle_min: float, meta: dict) -> dict:
    reuse = mgr.reusable_pod(kind)
    if reuse:
        return reuse
    s = mgr.settings()
    choices = cloud.gpu_choices(mgr.rp().gpus(), s["datacenter"], s["gpu"])
    mgr.check_budget(max(p for _, p in choices[:4]) * hard_cap_s / 3600)
    return mgr.start_pod_any(kind, choices[:4], hard_cap_s, idle_min, meta=meta,
                             report=lambda m: print(f"  {m}", flush=True))


def sync_scripts(vol: cloud.Volume) -> None:
    pod_dir = Path(glore_pod.__file__).parent
    for name in ("glore_pod.py", "glore_label.py", "glore_generate.py", "pod_start.sh"):
        vol.put_bytes(f"glore/{name}", (pod_dir / name).read_bytes().replace(b"\r\n", b"\n"))


# ---------------------------------------------------------------- commands

def cmd_train(a) -> dict:
    if a.variant not in glore_pod.VARIANT_DIRS:
        raise CloudError(f"variant must be one of {', '.join(glore_pod.VARIANT_DIRS)}")
    mgr = manager()
    settle_ledger(mgr)
    vol = mgr.volume()
    raw = vol.get_bytes(f"datasets/{a.pack}/dataset.json")
    if not raw:
        raise CloudError(f"No dataset for '{a.pack}' on the volume yet: upload it once from GLORE on your PC")
    doc = json.loads(raw)
    n = len(doc.get("samples", []))
    audio = (doc.get("metadata") or {}).get("audio", "prepared")
    dhash = hashlib.sha256(raw).hexdigest()[:12]
    params = glore_pod.train_params(n, a.epochs, a.rank, a.save_every)
    run_id = (f"{dhash}-{a.variant}-r{params['rank']}-e{params['epochs']}"
              + ("-full" if audio == "full" else "") + ("-cap" if a.autolabel else ""))
    if vol.list_keys(f"loras/{a.pack}/{run_id}/final/"):
        raise CloudError(f"Already trained: {run_id}")
    sync_scripts(vol)
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "mode": "train", "pack": a.pack, "run_id": run_id, "variant": a.variant,
           "dataset_dir": f"datasets/{a.pack}", "tensor_dir": f"tensors/{a.pack}/{dhash}-{a.variant}",
           "output_dir": f"loras/{a.pack}/{run_id}", "params": params, "max_duration": 240.0,
           "autolabel": a.autolabel, "genre_ratio": 30}
    vol.put_bytes(f"glore/queue/{job_id}.json", json.dumps(job).encode())
    try:
        row = _start_or_reuse(mgr, "train", a.hard_cap_hours * 3600, 10, {"pack": a.pack, "run_id": run_id})
    except CloudError:
        vol.delete(f"glore/queue/{job_id}.json")
        raise
    out = {"job": job_id, "run_id": run_id, "pod": row["pod_id"], "gpu": row.get("gpu"),
           "per_hour": row.get("cost_per_hr"), "tracks": n, "params": params}
    print(json.dumps(out, indent=2))
    return out


def resolve_lora(vol: cloud.Volume, pack: str, ref: str) -> str:
    """'<run_id>' -> its final LoRA dir; '<run_id>@e100' -> that checkpoint's dir (volume keys)."""
    run_id, _, ep = ref.partition("@e")
    base = f"loras/{pack}/{run_id}"
    if not ep:
        if not vol.list_keys(f"{base}/final/"):
            raise CloudError(f"No finished LoRA {run_id}")
        return f"{base}/final"
    for key in vol.list_keys(f"{base}/checkpoints/"):
        d = key.split("/")[-2]
        if glore_pod.checkpoint_epoch(d) == int(ep) and key.endswith("adapter_model.safetensors"):
            return key.rsplit("/", 1)[0]
    raise CloudError(f"No epoch {ep} checkpoint for {run_id}")


def cmd_generate(a) -> dict:
    takes = json.loads(Path(a.takes).read_text(encoding="utf-8")) if a.takes else []
    if not takes:
        raise CloudError("No takes given")
    mgr = manager()
    settle_ledger(mgr)
    vol = mgr.volume()
    for t in takes:
        if t.get("lora"):
            t["lora_dir"] = resolve_lora(vol, t.get("pack", "house"), t["lora"])
            t.setdefault("trigger", f"glore_{t.get('pack', 'house')}")
    sync_scripts(vol)
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "mode": "generate", "variant": a.variant, "lm": a.lm or None, "takes": takes}
    vol.put_bytes(f"glore/queue/{job_id}.json", json.dumps(job).encode())
    row = _start_or_reuse(mgr, "train", a.hard_cap_hours * 3600, 10, {"job": job_id})
    print(json.dumps({"job": job_id, "pod": row["pod_id"], "gpu": row.get("gpu"), "takes": len(takes)}, indent=2))
    if a.wait:
        return cmd_fetch(argparse.Namespace(job=job_id, out=a.out, wait=True, timeout_min=a.timeout_min))
    return {"job": job_id}


def job_status(vol: cloud.Volume, job_id: str) -> dict:
    raw = vol.get_bytes(f"glore/runs/{job_id}/status.json")
    if raw:
        return json.loads(raw)
    return {"phase": "queued"} if vol.get_bytes(f"glore/queue/{job_id}.json") else {"phase": "unknown"}


def cmd_fetch(a) -> dict:
    mgr = manager()
    vol = mgr.volume()
    t0 = time.time()
    while True:
        st = job_status(vol, a.job)
        print(f"  {st.get('phase')}: {st.get('message', '')}", flush=True)
        if st.get("phase") in ("done", "error", "stopped") or not a.wait:
            break
        if time.time() - t0 > a.timeout_min * 60:
            raise CloudError("Timed out waiting; the job keeps running - fetch again later")
        time.sleep(30)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for t in (st.get("result") or {}).get("takes", []):
        if t.get("mp3"):
            dest = out / Path(t["mp3"]).name
            vol.download(t["mp3"], dest)
            got.append(str(dest))
    (out / "result.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(json.dumps({"phase": st.get("phase"), "files": got, "error": st.get("error")}, indent=2))
    if st.get("phase") == "error":
        raise CloudError(st.get("error") or "the job failed")
    return {"files": got, "status": st}


def cmd_status(a) -> dict:
    mgr = manager()
    settle_ledger(mgr)
    vol = mgr.volume()
    runs = sorted({k.split("/")[2] for k in vol.list_keys("glore/runs/") if k.endswith("status.json")})
    rows = []
    for jid in runs:
        st = job_status(vol, jid)
        rows.append({"job": jid, "mode": st.get("mode"), "phase": st.get("phase"), "message": st.get("message"),
                     "epoch": st.get("epoch"), "loss": st.get("loss"), "updated": st.get("updated")})
    rows.sort(key=lambda r: r.get("updated") or 0)
    out = {"jobs": rows[-int(a.last):], "active_pods": mgr.ledger.active(), "month_spend": mgr.month_spend(),
           "monthly_limit": mgr.settings()["monthly_limit"]}
    print(json.dumps(out, indent=2, default=str))
    return out


def cmd_loras(a) -> dict:
    vol = manager().volume()
    found: dict[str, list] = {}
    for k in vol.list_keys(f"loras/{a.pack}/"):
        if not k.endswith("adapter_model.safetensors"):
            continue
        parts = k.split("/")
        run_id = parts[2]
        if parts[3] == "final":
            found.setdefault(run_id, []).append("final")
        elif parts[3] == "checkpoints":
            ep = glore_pod.checkpoint_epoch(parts[4])
            if ep is not None:
                found.setdefault(run_id, []).append(f"e{ep}")
    print(json.dumps(found, indent=2))
    return found


def cmd_check(a) -> dict:
    """Which volumes this S3 key can see, and whether the configured one is among them."""
    s = env_settings()
    vol = cloud.Volume(s)
    out = {"datacenter": s["datacenter"], "volume_id_set": s["volume_id"][:3] + "…" + s["volume_id"][-2:],
           "access_key_set": s["s3_access_key"][:5] + "…"}
    try:
        names = [b["Name"] for b in vol.s3.list_buckets().get("Buckets", [])]
        out["volumes_this_key_sees"] = names
        out["configured_volume_found"] = s["volume_id"] in names
    except Exception as e:
        out["list_error"] = cloud._s3_msg(e)
    try:
        vol.s3.list_objects_v2(Bucket=vol.bucket, MaxKeys=3)
        out["volume_readable"] = True
    except Exception as e:
        out["volume_readable"] = False
        out["read_error"] = cloud._s3_msg(e)
    try:  # can the web app play takes straight from the volume through a time-limited link?
        import urllib.request
        key = "glore/glore_pod.py"  # read-only test on a file that's always there
        url = vol.s3.generate_presigned_url("get_object", Params={"Bucket": vol.bucket, "Key": key}, ExpiresIn=600)
        with urllib.request.urlopen(url, timeout=20) as r:
            body = r.read()
            out["presigned_links"] = "ok" if body.startswith(b'"""GLORE') else f"unexpected body {body[:40]!r}"
            out["presign_cors"] = r.headers.get("Access-Control-Allow-Origin")
    except Exception as e:
        out["presigned_links"] = f"failed: {e}"
    try:
        out["runpod_api"] = f"ok, {len(cloud.RunPod(s['api_key']).gpus())} GPU types listed"
    except CloudError as e:
        out["runpod_api"] = str(e)
    print(json.dumps(out, indent=2))
    return out


def cmd_du(a) -> dict:
    """What's using the network volume, by folder (read-only)."""
    vol = cloud.Volume(env_settings())
    depth = int(a.depth)
    sizes: dict[str, int] = {}
    total = 0
    prefixes = [a.prefix] if a.prefix else ["tensors/", "datasets/", "loras/", "outputs/", "glore/",
                                              "ace/checkpoints/", ".cache/hf/"]
    for k, n in (kn for p in prefixes for kn in vol.list_sizes(p)):
        total += n
        key = "/".join(k.split("/")[:depth])
        sizes[key] = sizes.get(key, 0) + n
    rows = sorted(sizes.items(), key=lambda kv: -kv[1])
    out = {"total_gb": round(total / 2 ** 30, 2),
           "folders": [{"path": k, "gb": round(v / 2 ** 30, 2)} for k, v in rows if v > 2 ** 20][:60]}
    print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- GLORE Web state

def encrypt_state(data: dict, passphrase: str) -> dict:
    """AES-256-GCM with a PBKDF2-SHA256 key: the web app decrypts it with Web Crypto.
    The repo is public, so the state branch only ever holds this ciphertext."""
    import base64
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = os.urandom(16), os.urandom(12)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000, dklen=32)
    ct = AESGCM(key).encrypt(iv, json.dumps(data).encode(), None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"v": 1, "kdf": "pbkdf2-sha256", "iter": 200_000, "salt": b64(salt), "iv": b64(iv), "ct": b64(ct)}


def build_state(mgr: CloudManager, link_days: int = 7, max_jobs: int = 25) -> dict:
    vol = mgr.volume()
    try:
        settle_ledger(mgr)
    except CloudError:
        pass  # a full volume can't take ledger writes; report what we can
    jobs = []
    for jid in sorted({k.split("/")[2] for k in vol.list_keys("glore/runs/") if k.endswith("status.json")}):
        st = job_status(vol, jid)
        row = {k: st.get(k) for k in ("job", "mode", "phase", "message", "progress", "epoch", "epochs", "loss",
                                       "eta_s", "started", "heartbeat", "gpu", "error")}
        res = st.get("result") or {}
        if st.get("mode") == "generate":
            takes = []
            for t in res.get("takes", []):
                t = dict(t)
                if t.get("mp3"):
                    t["url"] = vol.s3.generate_presigned_url(
                        "get_object", Params={"Bucket": vol.bucket, "Key": t["mp3"]}, ExpiresIn=link_days * 86400)
                takes.append(t)
            row["takes"] = takes
        elif st.get("mode") == "train":
            row["run_id"] = res.get("lora_dir", "").split("/")[2] if res.get("lora_dir") else None
        jobs.append(row)
    jobs.sort(key=lambda r: r.get("started") or 0, reverse=True)
    loras = {}
    for k in vol.list_keys("loras/"):
        parts = k.split("/")
        if len(parts) < 5 or not k.endswith("adapter_model.safetensors"):
            continue
        name = "final" if parts[3] == "final" else (
            f"e{glore_pod.checkpoint_epoch(parts[4])}" if parts[3] == "checkpoints" else None)
        if name:
            loras.setdefault(parts[2], {"pack": parts[1], "points": []})["points"].append(name)
    s = mgr.settings()
    return {"updated": time.time(), "jobs": jobs[:max_jobs], "loras": loras,
            "active_pods": mgr.ledger.active(), "month_spend": mgr.month_spend(),
            "monthly_limit": s.get("monthly_limit"), "gpu_pref": s.get("gpu"), "datacenter": s.get("datacenter")}


def cmd_publish_state(a) -> dict:
    key = os.environ.get("GLORE_WEB_KEY", "")
    if not key:
        raise CloudError("Add a GLORE_WEB_KEY repo secret (any long passphrase) to use GLORE Web")
    state = build_state(manager())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(encrypt_state(state, key)), encoding="utf-8")
    print(json.dumps({"jobs": len(state["jobs"]), "loras": len(state["loras"]), "written": a.out}))
    return state


def cmd_grow(a) -> dict:
    """Make the network volume bigger (RunPod can't shrink it again)."""
    s = env_settings()
    rp = cloud.RunPod(s["api_key"])
    before = rp.get_volume(s["volume_id"])
    if int(before.get("size") or 0) >= a.gb:
        out = {"size_gb": before.get("size"), "changed": False}
    else:
        after = rp.resize_volume(s["volume_id"], a.gb)
        out = {"was_gb": before.get("size"), "size_gb": after.get("size", a.gb), "changed": True}
    print(json.dumps(out))
    return out


def cmd_stop(a) -> dict:
    mgr = manager()
    n = 0
    for r in mgr.ledger.active():
        mgr.rp().terminate(r["pod_id"])
        n += 1
    settle_ledger(mgr)
    print(json.dumps({"stopped": n}))
    return {"stopped": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="glore-cloud")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--pack", default="house")
    t.add_argument("--variant", default="xl_sft")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--rank", type=int, default=64)
    t.add_argument("--save-every", type=int, default=50)
    t.add_argument("--autolabel", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--hard-cap-hours", type=float, default=8.0)
    g = sub.add_parser("generate")
    g.add_argument("--takes", required=True, help="JSON file: [{name, caption, lora?, scale?, seed, duration, bpm, key}]")
    g.add_argument("--variant", default="xl_sft")
    g.add_argument("--lm", default=glore_pod.CLOUD_LM)
    g.add_argument("--hard-cap-hours", type=float, default=1.5)
    g.add_argument("--wait", action="store_true")
    g.add_argument("--out", default="out")
    g.add_argument("--timeout-min", type=float, default=80)
    f = sub.add_parser("fetch")
    f.add_argument("--job", required=True)
    f.add_argument("--out", default="out")
    f.add_argument("--wait", action="store_true")
    f.add_argument("--timeout-min", type=float, default=80)
    s = sub.add_parser("status")
    s.add_argument("--last", default=5)
    lo = sub.add_parser("loras")
    lo.add_argument("--pack", default="house")
    sub.add_parser("stop")
    sub.add_parser("check")
    gr = sub.add_parser("grow")
    gr.add_argument("--gb", type=int, required=True)
    ps = sub.add_parser("publish-state")
    ps.add_argument("--out", default="state/state.json")
    du = sub.add_parser("du")
    du.add_argument("--prefix", default="")
    du.add_argument("--depth", default=3)
    a = ap.parse_args(argv)
    try:
        {"train": cmd_train, "generate": cmd_generate, "fetch": cmd_fetch, "status": cmd_status,
         "loras": cmd_loras, "stop": cmd_stop, "check": cmd_check, "du": cmd_du, "publish-state": cmd_publish_state, "grow": cmd_grow}[a.cmd](a)
        return 0
    except CloudError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
