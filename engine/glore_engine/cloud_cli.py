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


WORK_GPUS = "RTX 4090, L40S, RTX 6000 Ada, RTX 5090, A40, RTX A6000, L40, A100"  # songs / tools: plenty for XL-SFT, far cheaper


def work_gpus() -> str:
    return os.environ.get("GLORE_WORK_GPU") or WORK_GPUS


def _idle_pod(mgr: CloudManager, vol: cloud.Volume) -> dict | None:
    """A pod that's up, has time left and isn't busy (a training pod mid-run is not reusable)."""
    s = mgr.settings()
    for r in mgr.ledger.active():
        if r.get("datacenter") != s["datacenter"] or r["hard_cap_s"] - (time.time() - r["started"]) < 1200:
            continue
        hb = vol.get_json(f"glore/pods/{r['pod_id']}.json") or {}
        if hb.get("busy") or time.time() - float(hb.get("heartbeat", 0)) > 180:
            continue
        p = mgr.rp().get_pod(r["pod_id"])
        if p and p.get("status") == "RUNNING":
            return r
    return None


def _start_or_reuse(mgr: CloudManager, kind: str, hard_cap_s: float, idle_min: float, meta: dict,
                    gpus: str | None = None) -> dict:
    try:
        return _start_or_reuse_inner(mgr, kind, hard_cap_s, idle_min, meta, gpus)
    except Exception:
        if meta.get("job"):  # no GPU: don't leave the job queued for some later pod to run by surprise
            mgr.volume().delete(f"glore/queue/{meta['job']}.json")
        raise


def _start_or_reuse_inner(mgr: CloudManager, kind: str, hard_cap_s: float, idle_min: float, meta: dict,
                          gpus: str | None = None) -> dict:
    reuse = _idle_pod(mgr, mgr.volume())
    if reuse:
        print(f"  reusing idle pod {reuse['pod_id']}")
        return reuse
    s = mgr.settings()
    choices = cloud.gpu_choices(mgr.rp().gpus(), s["datacenter"], gpus or s["gpu"])
    mgr.check_budget(max(p for _, p in choices[:4]) * hard_cap_s / 3600)
    return mgr.start_pod_any(kind, choices[:4], hard_cap_s, idle_min, meta=meta,
                             report=lambda m: print(f"  {m}", flush=True))


ENGINE_MODULES = ("__init__", "audio_io", "analysis", "separation", "cleanup", "vocal_lock", "chops", "fx", "mixing")


def sync_scripts(vol: cloud.Volume) -> None:
    pod_dir = Path(glore_pod.__file__).parent
    for name in ("glore_pod.py", "glore_label.py", "glore_generate.py", "glore_song.py", "glore_tools.py",
                 "pod_start.sh"):
        vol.put_bytes(f"glore/{name}", (pod_dir / name).read_bytes().replace(b"\r\n", b"\n"))
    for mod in ENGINE_MODULES:  # GLORE's own audio code, for glore_tools.py
        vol.put_bytes(f"glore/engine/glore_engine/{mod}.py", (pod_dir.parent / f"{mod}.py").read_bytes())


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
    row = _start_or_reuse(mgr, "train", a.hard_cap_hours * 3600, 10, {"job": job_id}, gpus=work_gpus())
    print(json.dumps({"job": job_id, "pod": row["pod_id"], "gpu": row.get("gpu"), "takes": len(takes)}, indent=2))
    if a.wait:
        return cmd_fetch(argparse.Namespace(job=job_id, out=a.out, wait=True, timeout_min=a.timeout_min))
    return {"job": job_id}


# ---------------------------------------------------------------- uploads from GLORE Web

UPLOAD_BRANCH = "glore-uploads"


def decrypt_upload(blob: bytes, passphrase: str) -> bytes:
    """GLORE Web encrypts files before they touch the (public) repo: salt(16) | iv(12) | AES-GCM ciphertext."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv, ct = blob[:16], blob[16:28], blob[28:]
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000, dklen=32)
    return AESGCM(key).decrypt(iv, ct, None)


def _github(method: str, path: str, accept: str = "application/vnd.github+json", body: dict | None = None):
    import urllib.request
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if not (repo and token):
        raise CloudError("GITHUB_REPOSITORY / GITHUB_TOKEN missing (run this from the GitHub workflow)")
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}{path}", method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {token}", "Accept": accept,
                                          "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "GLORE"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def take_upload(path: str) -> bytes:
    """Fetch an encrypted upload from the glore-uploads branch, then delete it there."""
    raw = _github("GET", f"/contents/{path}?ref={UPLOAD_BRANCH}", accept="application/vnd.github.raw+json")
    try:
        meta = json.loads(_github("GET", f"/contents/{path}?ref={UPLOAD_BRANCH}"))
        _github("DELETE", f"/contents/{path}", body={"message": "GLORE: upload taken", "sha": meta["sha"],
                                                     "branch": UPLOAD_BRANCH})
    except Exception as e:  # leaving it there is harmless: it's encrypted
        print(f"  (couldn't tidy the upload away: {e})")
    return raw


# ---------------------------------------------------------------- the library (everything you bring in)

LIB = "library"
KINDS = ("vocal", "song", "beat", "stem", "chops", "other")


def _ffmpeg() -> str:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    from .linkimport import ffmpeg_path
    return ffmpeg_path()


def to_flac(src: Path, dest: Path) -> float:
    """Any audio -> 44.1 kHz stereo FLAC; returns the duration in seconds."""
    import subprocess
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100",
                    "-sample_fmt", "s16", str(dest)], check=True)
    out = subprocess.run([_ffmpeg(), "-i", str(dest)], capture_output=True, text=True).stderr
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def safe_file(name: str) -> str:
    return "".join(c for c in Path(name or "audio").name if c.isalnum() or c in " ._-()&'").strip()[:80] or "audio"


def library_add(vol: cloud.Volume, path: Path, name: str, kind: str, origin: str, **extra) -> dict:
    """Store a file (any format) in the library as FLAC, with a little item.json beside it."""
    item_id = uuid.uuid4().hex[:10]
    stem = Path(safe_file(name)).stem or "audio"
    with tempfile.TemporaryDirectory() as tmp:
        flac = Path(tmp) / f"{stem}.flac"
        dur = to_flac(path, flac)
        key = f"{LIB}/{item_id}/{flac.name}"
        vol.put_file(key, flac)
    item = {"id": item_id, "name": stem, "kind": kind if kind in KINDS else "other", "file": key,
            "duration": round(dur, 2), "origin": origin, "created": time.time(), **extra}
    vol.put_bytes(f"{LIB}/{item_id}/item.json", json.dumps(item).encode())
    return item


def library_items(vol: cloud.Volume) -> list[dict]:
    items = []
    for k in vol.list_keys(f"{LIB}/"):
        if k.endswith("/item.json"):
            raw = vol.get_bytes(k)
            if raw:
                items.append(json.loads(raw))
    return sorted(items, key=lambda i: i.get("created", 0), reverse=True)


def library_get(vol: cloud.Volume, item_id: str) -> dict:
    raw = vol.get_bytes(f"{LIB}/{item_id}/item.json")
    if not raw:
        raise CloudError(f"No library item {item_id}")
    return json.loads(raw)


def cmd_import(a) -> dict:
    """A file uploaded from GLORE Web -> the library."""
    key = os.environ.get("GLORE_WEB_KEY", "")
    if not key:
        raise CloudError("GLORE_WEB_KEY is needed to decrypt uploads")
    vol = manager().volume()
    with tempfile.TemporaryDirectory() as tmp:
        name = (a.name or "upload").replace("_", " ")
        src = Path(tmp) / safe_file(name)
        src.write_bytes(decrypt_upload(take_upload(a.upload), key))
        item = library_add(vol, src, name, a.kind, "upload")
    print(json.dumps(item, indent=2))
    return item


HOME_ONLY = ("youtube", "spotify", "apple")  # YouTube bot-blocks cloud machines; Spotify/Apple match via YouTube
PENDING = f"{LIB}/pending"


def cmd_link(a) -> dict:
    """SoundCloud: downloaded right here. YouTube / Spotify / Apple Music: YouTube blocks cloud servers, so the
    link waits in the library for GLORE on your PC, which downloads it at home and uploads it."""
    from .linkimport import LinkError, detect_service, fetch_link
    vol = manager().volume()
    if a.cancel:  # a waiting or failed link, one id or several
        for rid in [i for i in a.cancel.split(",") if i]:
            vol.delete(f"{PENDING}/{rid}.json")
            print(f"Removed {rid}")
        return {"cancelled": a.cancel}
    if not a.url:
        raise CloudError("Give --url (or --cancel)")
    try:
        service = detect_service(a.url)
    except Exception as e:
        raise CloudError(str(e)) from e
    if service in HOME_ONLY and not a.here:
        req = {"id": uuid.uuid4().hex[:10], "url": a.url, "kind": a.kind, "service": service, "created": time.time()}
        vol.put_bytes(f"{PENDING}/{req['id']}.json", json.dumps(req).encode())
        print(f"Waiting for your PC: {service.title()} blocks cloud servers, so GLORE on your PC will download it "
              f"next time it's running, then it shows up in your library.")
        return {"pending": req}
    last = [0.0]

    def report(p, m):
        if p - last[0] >= 0.2:
            last[0] = p
            print(f"  {m} {p:.0%}")
    try:
        wav, info, tmp = fetch_link(a.url, report)
    except LinkError as e:  # keep it visible in GLORE Web (with a remove button) instead of failing silently
        req = {"id": uuid.uuid4().hex[:10], "url": a.url, "kind": a.kind, "service": service, "created": time.time(),
               "error": str(e), "failed": True}
        vol.put_bytes(f"{PENDING}/{req['id']}.json", json.dumps(req).encode())
        raise CloudError(str(e)) from e
    try:
        name = f"{info.artist} - {info.title}" if info.artist and info.title else (info.title or "Imported")
        item = library_add(vol, wav, name, a.kind, "link", url=a.url, service=info.service,
                           title=info.title, artist=info.artist, matched=info.matched)
    finally:
        tmp.cleanup()
    print(json.dumps(item, indent=2))
    return item


def pending_links(vol: cloud.Volume) -> list[dict]:
    out = []
    for k in vol.list_keys(f"{PENDING}/"):
        raw = vol.get_bytes(k) if k.endswith(".json") else None
        if raw:
            out.append(json.loads(raw))
    return sorted(out, key=lambda r: r.get("created", 0))


def process_pending_links(vol: cloud.Volume, report=print) -> list[dict]:
    """Run on the PC (home internet): download every waiting link, put it in the library."""
    from .linkimport import LinkError, fetch_link
    done = []
    for req in pending_links(vol):
        if req.get("failed"):  # failed in the cloud for good (e.g. DRM): only shown, never retried
            continue
        key = f"{PENDING}/{req['id']}.json"
        try:
            wav, info, tmp = fetch_link(req["url"])
        except LinkError as e:
            if req.get("error") != str(e):  # shows in GLORE Web; tried again each minute in case it was a blip
                req["error"] = str(e)
                vol.put_bytes(key, json.dumps(req).encode())
            report(f"link failed: {req['url']}: {e}")
            continue
        try:
            name = f"{info.artist} - {info.title}" if info.artist and info.title else (info.title or "Imported")
            item = library_add(vol, wav, name, req.get("kind", "song"), "link", url=req["url"], service=info.service,
                               title=info.title, artist=info.artist, matched=info.matched)
        finally:
            tmp.cleanup()
        vol.delete(key)
        report(f"downloaded {item['name']} into your cloud library")
        done.append(item)
    return done


def cmd_probe(a) -> dict:
    """Fetch a stored file the way a phone's audio player does (presigned link, Range request)."""
    import urllib.error
    import urllib.request
    vol = manager().volume()
    key = a.key or next((k for k in reversed(vol.list_keys("outputs/")) if k.endswith(".mp3")), None)
    keys = [key] + [k for k in vol.list_keys("glore/")[:200] if k.endswith("glore_pod.py")][:1]
    print("keys:", keys)
    for key in keys:
        _probe_key(vol, key, a.days)
    return {}


def _probe_key(vol, key, days0):
    import urllib.error
    import urllib.request
    print("key:", key)
    out = {}
    for days in (days0, 600 / 86400):
        url = vol.s3.generate_presigned_url("get_object", Params={"Bucket": vol.bucket, "Key": key},
                                            ExpiresIn=int(days * 86400))
        ua = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"}
        for hdrs in ({**ua}, {**ua, "Range": "bytes=0-1"}, {**ua, "Range": "bytes=0-", "Origin": "https://astr0h.github.io"}):
            req = urllib.request.Request(url, headers=hdrs)
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    res = f"{r.status} type={r.headers.get('Content-Type')} len={r.headers.get('Content-Length')} " \
                          f"range={r.headers.get('Content-Range')} accept-ranges={r.headers.get('Accept-Ranges')} " \
                          f"cors={r.headers.get('Access-Control-Allow-Origin')}"
            except urllib.error.HTTPError as e:
                res = f"HTTP {e.code}: {e.read()[:300]!r}"
            except Exception as e:
                res = f"failed: {e}"
            print(f"expires {days:.3g}d {hdrs}: {res}")
            out[f"{days}-{hdrs}"] = res
    return out


def cmd_queue(a) -> dict:
    """Jobs waiting for a pod. --drop <id> (or 'all') removes them."""
    vol = manager().volume()
    keys = [k for k in vol.list_keys("glore/queue/") if k.endswith(".json") and "/." not in k]
    jobs = [json.loads(vol.get_bytes(k) or b"{}") for k in keys]
    for k, j in zip(keys, jobs):
        drop = a.drop in ("all", j.get("id"))
        if drop:
            vol.delete(k)
        print(f"{j.get('id')}  {j.get('mode'):<8} {'dropped' if drop else 'waiting'}")
    return {"queue": jobs}


def cmd_library(a) -> dict:
    vol = manager().volume()
    if a.delete:  # one id or several, comma-separated
        gone = []
        for item_id in [i for i in a.delete.split(",") if i]:
            try:
                item = library_get(vol, item_id)
            except CloudError:
                print(f"Already gone: {item_id}")
                continue
            for k in vol.list_keys(f"{LIB}/{item['id']}/"):
                vol.delete(k)
            print(f"Deleted {item['name']}")
            gone.append(item["id"])
        return {"deleted": gone}
    items = library_items(vol)
    for i in items:
        print(f"{i['id']}  {i['kind']:<6} {i.get('duration', 0):6.0f}s  {i['name']}")
    return {"items": items}


def cmd_song(a) -> dict:
    """Your vocal (or a whole song, vocal taken out) -> beats built around it, mixed, as MP3s."""
    key = os.environ.get("GLORE_WEB_KEY", "")
    if a.upload and not key:
        raise CloudError("GLORE_WEB_KEY is needed to decrypt uploads")
    takes = json.loads(Path(a.takes).read_text(encoding="utf-8")) if a.takes else []
    if not takes:
        raise CloudError("No takes given")
    if a.kind not in ("vocal", "song"):
        raise CloudError("kind must be vocal or song")
    mgr = manager()
    settle_ledger(mgr)
    vol = mgr.volume()
    job_id = uuid.uuid4().hex[:10]
    item_id = a.item
    if a.upload:
        with tempfile.TemporaryDirectory() as tmp:
            name = (a.name or "upload").replace("_", " ")
            f = Path(tmp) / safe_file(name)
            f.write_bytes(decrypt_upload(take_upload(a.upload), key))
            item_id = library_add(vol, f, name, a.kind, "upload")["id"]
        print(f"  saved to your library ({item_id})")
    elif a.source:
        item_id = None
    elif not item_id:
        raise CloudError("Give --upload (from GLORE Web), --item (from your library) or --source (a volume path)")
    if item_id:
        item = library_get(vol, item_id)
        a.bpm = a.bpm or item.get("bpm")
        a.key = a.key or item.get("key")
    for t in takes:
        if t.get("lora"):
            t["lora_dir"] = resolve_lora(vol, t.get("pack", "house"), t["lora"])
            t.setdefault("trigger", f"glore_{t.get('pack', 'house')}")
    sync_scripts(vol)
    job = {"id": job_id, "mode": "song", "kind": a.kind, "variant": a.variant, "gh_run": os.environ.get("GITHUB_RUN_ID"),
           **({"item": item_id} if item_id else {"upload": a.source}),
           "fx": a.fx, "reverb": a.reverb, "delay": a.delay, "chops": a.chops or None,
           "chops_density": a.chops_density, "vocal_db": a.vocal_db, "beat_db": a.beat_db, "chops_db": a.chops_db,
           "bpm": a.bpm, "key": (a.key or "").replace("_", " ") or None, "takes": takes}
    vol.put_bytes(f"glore/queue/{job_id}.json", json.dumps(job).encode())
    row = _start_or_reuse(mgr, "train", a.hard_cap_hours * 3600, 10, {"job": job_id}, gpus=work_gpus())
    print(json.dumps({"job": job_id, "pod": row["pod_id"], "gpu": row.get("gpu"), "takes": len(takes)}, indent=2))
    if a.wait:
        return cmd_fetch(argparse.Namespace(job=job_id, out=a.out, wait=True, timeout_min=a.timeout_min))
    return {"job": job_id}


TOOL_OPS = ("stems", "clean", "analyse", "lock", "chops")


def cmd_tool(a) -> dict:
    """Run one GLORE tool on a library item, on a cloud GPU. Results land back in the library."""
    if a.op not in TOOL_OPS:
        raise CloudError(f"op must be one of {', '.join(TOOL_OPS)}")
    params = {}
    for kv in a.set or []:  # key=value, numbers become numbers, a,b,c becomes a list
        k, _, v = kv.partition("=")
        v = v.replace("_", " ") if k == "key" else v
        try:
            params[k] = float(v)
        except ValueError:
            params[k] = v.split(",") if "," in v else v
    if a.op == "clean" and isinstance(params.get("stages"), str):
        params["stages"] = [params["stages"]]
    mgr = manager()
    settle_ledger(mgr)
    vol = mgr.volume()
    item = library_get(vol, a.item)
    sync_scripts(vol)
    job_id = uuid.uuid4().hex[:10]
    vol.put_bytes(f"glore/queue/{job_id}.json", json.dumps(
        {"id": job_id, "mode": "tool", "op": a.op, "item": a.item, "params": params}).encode())
    row = _start_or_reuse(mgr, "train", 3600, 10, {"job": job_id}, gpus=work_gpus())
    print(json.dumps({"job": job_id, "op": a.op, "item": item["name"], "pod": row["pod_id"]}, indent=2))
    if a.wait:
        st = wait_job(vol, job_id, a.timeout_min)
        if st.get("phase") != "done":
            raise CloudError(st.get("error") or st.get("message") or "the tool didn't finish")
        print(json.dumps(st.get("result"), indent=2))
    return {"job": job_id}


def job_status(vol: cloud.Volume, job_id: str) -> dict:
    raw = vol.get_bytes(f"glore/runs/{job_id}/status.json")
    if raw:
        return json.loads(raw)
    return {"phase": "queued"} if vol.get_bytes(f"glore/queue/{job_id}.json") else {"phase": "unknown"}


def wait_job(vol: cloud.Volume, job_id: str, timeout_min: float, wait: bool = True, every: float | None = None) -> dict:
    every = every or float(os.environ.get("GLORE_POLL_S", "30"))
    t0, last = time.time(), None
    while True:
        st = job_status(vol, job_id)
        line = f"  {st.get('phase')}: {st.get('message', '')}"
        if line != last:
            print(line, flush=True)
            last = line
        if st.get("phase") in ("done", "error", "stopped") or not wait:
            return st
        if time.time() - t0 > timeout_min * 60:
            raise CloudError("Timed out waiting; the job keeps running - fetch again later")
        time.sleep(every)


def cmd_fetch(a) -> dict:
    mgr = manager()
    vol = mgr.volume()
    st = wait_job(vol, a.job, a.timeout_min, a.wait)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for t in (st.get("result") or {}).get("takes", []):
        for k in ("mp3", "beat_mp3", "zip"):
            if t.get(k):
                dest = out / Path(t[k]).name
                vol.download(t[k], dest)
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


MEDIA_BRANCH = "glore-media"
MEDIA_MAX_BYTES = 95 * 2 ** 20   # GitHub refuses files over 100 MB
MEDIA_TAKE_JOBS = 12             # newest song/generate jobs whose audio is playable in the app
MEDIA_LIBRARY = 60               # newest library items with a playable preview


def media_name(key: str, kind: str) -> str:
    return hashlib.sha256(f"{key}|{kind}".encode()).hexdigest()[:24] + ".bin"


def encrypt_blob(data: bytes, passphrase: str) -> bytes:
    """Same format as GLORE Web's uploads: salt(16) | iv(12) | AES-GCM ciphertext."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = os.urandom(16), os.urandom(12)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000, dklen=32)
    return salt + iv + AESGCM(key).encrypt(iv, data, None)


def build_state(mgr: CloudManager, max_jobs: int = 25) -> dict:
    """Everything GLORE Web shows. Audio is referenced by `media` names: encrypted copies on the
    glore-media branch (RunPod's storage only takes signed requests, so a phone can't stream from it)."""
    vol = mgr.volume()
    try:
        settle_ledger(mgr)
    except CloudError:
        pass  # a full volume can't take ledger writes; report what we can
    wanted: list[dict] = []

    def want(key: str, kind: str, filename: str) -> str:
        name = media_name(key, kind)
        wanted.append({"name": name, "key": key, "kind": kind, "filename": filename})
        return name

    jobs = []
    for jid in sorted({k.split("/")[2] for k in vol.list_keys("glore/runs/") if k.endswith("status.json")}):
        st = job_status(vol, jid)
        row = {k: st.get(k) for k in ("job", "mode", "phase", "message", "progress", "epoch", "epochs", "loss",
                                       "eta_s", "started", "heartbeat", "gpu", "error")}
        res = st.get("result") or {}
        if st.get("mode") in ("generate", "song"):
            row["takes"] = [dict(t) for t in res.get("takes", [])]
            row["gh_run"] = res.get("gh_run")
            if st.get("mode") == "song":
                row.update({k: res.get(k) for k in ("bpm", "key", "duration", "kind")})
        elif st.get("mode") == "tool":
            row.update({"op": res.get("op"), "item": res.get("item"),
                        "made": [i.get("id") for i in res.get("items", [])]})
        elif st.get("mode") == "train":
            row["run_id"] = res.get("lora_dir", "").split("/")[2] if res.get("lora_dir") else None
        jobs.append(row)
    jobs.sort(key=lambda r: r.get("started") or 0, reverse=True)
    jobs = jobs[:max_jobs]
    for row in [j for j in jobs if j.get("takes")][:MEDIA_TAKE_JOBS]:
        for t in row["takes"]:
            for field, kind, suffix in (("mp3", "full", ""), ("beat_mp3", "full", ""), ("zip", "full", "")):
                if t.get(field):
                    t[{"mp3": "media", "beat_mp3": "beat_media", "zip": "zip_media"}[field]] = \
                        want(t[field], kind, Path(t[field]).name)
    loras = {}
    for k in vol.list_keys("loras/"):
        parts = k.split("/")
        if len(parts) < 5 or not k.endswith("adapter_model.safetensors"):
            continue
        name = "final" if parts[3] == "final" else (
            f"e{glore_pod.checkpoint_epoch(parts[4])}" if parts[3] == "checkpoints" else None)
        if name:
            loras.setdefault(parts[2], {"pack": parts[1], "points": []})["points"].append(name)
    library = []
    for n, it in enumerate(library_items(vol)[:200]):
        it = dict(it)
        if n < MEDIA_LIBRARY:
            it["media"] = want(it["file"], "preview", f"{it['name']}.mp3")
        library.append(it)
    s = mgr.settings()
    pending = [{k: r.get(k) for k in ("id", "url", "service", "kind", "created", "error", "failed")}
               for r in pending_links(vol)]
    return {"updated": time.time(), "jobs": jobs, "loras": loras, "library": library, "pending": pending,
            "active_pods": mgr.ledger.active(), "month_spend": mgr.month_spend(), "media_branch": MEDIA_BRANCH,
            "monthly_limit": s.get("monthly_limit"), "gpu_pref": s.get("gpu"), "datacenter": s.get("datacenter"),
            "_media": wanted}


def _git(repo: Path, *args: str, input: bytes | None = None, check: bool = True) -> str:
    import subprocess
    r = subprocess.run(["git", *args], cwd=repo, input=input, capture_output=True)
    if check and r.returncode:
        raise CloudError(f"git {args[0]} failed: {r.stderr.decode(errors='replace')[-400:]}")
    return r.stdout.decode(errors="replace")


def publish_media(vol: cloud.Volume, wanted: list[dict], passphrase: str, remote: str, workdir: Path,
                  report=print) -> dict:
    """Make the glore-media branch hold exactly the wanted files, encrypted. Files already there are
    kept as they are (no re-download, no re-upload); the branch is one commit, so old audio doesn't pile up."""
    import subprocess
    repo = workdir / "media-repo"
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    have: dict[str, str] = {}
    if not subprocess.run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", remote,
                           f"{MEDIA_BRANCH}:refs/remotes/origin/{MEDIA_BRANCH}"], cwd=repo, capture_output=True).returncode:
        for line in _git(repo, "ls-tree", f"origin/{MEDIA_BRANCH}").splitlines():
            meta, name = line.split("\t", 1)
            have[name] = meta.split()[2]
    entries, added, skipped = {}, 0, 0
    seen = set()
    for w in wanted:
        if w["name"] in seen:
            continue
        seen.add(w["name"])
        if w["name"] in have:
            entries[w["name"]] = have[w["name"]]
            continue
        size = vol.size(w["key"])
        if not size:
            continue
        if w["kind"] == "full" and size > MEDIA_MAX_BYTES:
            skipped += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / ("src" + Path(w["key"]).suffix)
            vol.download(w["key"], src)
            if w["kind"] == "preview":
                out = Path(tmp) / "preview.mp3"
                subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-vn", "-c:a", "libmp3lame",
                                "-b:a", "192k", str(out)], check=True)
                src = out
            blob = Path(tmp) / "blob"
            blob.write_bytes(encrypt_blob(src.read_bytes(), passphrase))
            entries[w["name"]] = _git(repo, "hash-object", "-w", str(blob)).strip()
            added += 1
    if set(entries.items()) == set(have.items()):
        report(f"media: nothing new ({len(entries)} files)")
        return {"files": len(entries), "added": 0}
    index = "".join(f"100644 blob {sha}\t{name}\n" for name, sha in sorted(entries.items()))
    tree = _git(repo, "mktree", "--missing", input=index.encode()).strip()
    commit = _git(repo, "-c", "user.name=glore-bot", "-c", "user.email=glore-bot@users.noreply.github.com",
                  "commit-tree", tree, "-m", "GLORE media (encrypted)").strip()
    _git(repo, "push", "-q", "-f", remote, f"{commit}:refs/heads/{MEDIA_BRANCH}")
    report(f"media: {len(entries)} files, {added} new, {len(have) - len(set(have) & set(entries))} removed"
           + (f", {skipped} too big for GitHub" if skipped else ""))
    return {"files": len(entries), "added": added, "skipped": skipped}


def cmd_publish_state(a) -> dict:
    key = os.environ.get("GLORE_WEB_KEY", "")
    if not key:
        raise CloudError("Add a GLORE_WEB_KEY repo secret (any long passphrase) to use GLORE Web")
    mgr = manager()
    state = build_state(mgr)
    wanted = state.pop("_media")
    if a.media_remote:
        try:
            done = publish_media(mgr.volume(), wanted, key, a.media_remote, Path(a.out).parent / "work")
            state["media_files"] = done["files"]
        except Exception as e:  # the status still goes out; audio catches up next run
            print(f"  media publish failed: {e}")
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


def cmd_log(a) -> dict:
    """The end of a job's log (what went wrong)."""
    vol = cloud.Volume(env_settings())
    text = vol.tail(f"glore/runs/{a.job}/log.txt", int(a.bytes))
    print(text or "(no log)")
    return {"log": text}


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
    im = sub.add_parser("import")
    im.add_argument("--upload", required=True)
    im.add_argument("--name", default="")
    im.add_argument("--kind", default="vocal")
    li = sub.add_parser("link")
    li.add_argument("--url")
    li.add_argument("--cancel", help="remove waiting/failed link(s) by id, comma-separated")
    li.add_argument("--kind", default="song")
    li.add_argument("--here", action="store_true", help="try downloading on this machine even for YouTube")
    to = sub.add_parser("tool")
    to.add_argument("--op", required=True)
    to.add_argument("--item", required=True)
    to.add_argument("--set", action="append", help="key=value (repeat)")
    to.add_argument("--wait", action="store_true")
    to.add_argument("--timeout-min", type=float, default=40)
    pr = sub.add_parser("probe")
    pr.add_argument("--key")
    pr.add_argument("--days", type=float, default=7)
    qu = sub.add_parser("queue")
    qu.add_argument("--drop")
    lb = sub.add_parser("library")
    lb.add_argument("--delete")
    so = sub.add_parser("song")
    so.add_argument("--upload", help="path in the glore-uploads branch (from GLORE Web)")
    so.add_argument("--source", help="or: a file already on the volume")
    so.add_argument("--item", help="or: a library item id")
    so.add_argument("--fx", default="clean", choices=["clean", "airy", "club", "none"])
    so.add_argument("--reverb", type=float, default=0.5)
    so.add_argument("--delay", type=float, default=0.5)
    so.add_argument("--chops", default="", help="house | rnb | gospel (blank = no chops)")
    so.add_argument("--chops-density", type=float, default=1.0)
    so.add_argument("--vocal-db", type=float, default=0.0)
    so.add_argument("--beat-db", type=float, default=0.0)
    so.add_argument("--chops-db", type=float, default=0.0)
    so.add_argument("--name", default="")
    so.add_argument("--kind", default="vocal", help="vocal (a cappella) or song (vocal gets separated out)")
    so.add_argument("--takes", required=True)
    so.add_argument("--variant", default="xl_sft")
    so.add_argument("--bpm", type=float)
    so.add_argument("--key")
    so.add_argument("--hard-cap-hours", type=float, default=1.5)
    so.add_argument("--wait", action="store_true")
    so.add_argument("--out", default="out")
    so.add_argument("--timeout-min", type=float, default=100)
    lg = sub.add_parser("log")
    lg.add_argument("--job", required=True)
    lg.add_argument("--bytes", default=12000)
    gr = sub.add_parser("grow")
    gr.add_argument("--gb", type=int, required=True)
    ps = sub.add_parser("publish-state")
    ps.add_argument("--out", default="state/state.json")
    ps.add_argument("--media-remote", help="git remote to publish encrypted audio to (branch glore-media)")
    du = sub.add_parser("du")
    du.add_argument("--prefix", default="")
    du.add_argument("--depth", default=3)
    a = ap.parse_args(argv)
    try:
        {"train": cmd_train, "generate": cmd_generate, "fetch": cmd_fetch, "status": cmd_status,
         "loras": cmd_loras, "stop": cmd_stop, "check": cmd_check, "du": cmd_du, "publish-state": cmd_publish_state, "grow": cmd_grow, "log": cmd_log, "song": cmd_song,
                "import": cmd_import, "link": cmd_link, "library": cmd_library,
                "tool": cmd_tool, "queue": cmd_queue, "probe": cmd_probe}[a.cmd](a)
        return 0
    except CloudError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
