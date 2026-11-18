"""Cloud GPU (RunPod) for heavy jobs - LoRA training first.

How it fits together:
  * Your RunPod API key starts and stops pods (REST API v2).
  * Training audio and results live on a RunPod *network volume*. GLORE reads and
    writes it from your PC through RunPod's S3-compatible API (needs an S3 API key),
    so no pod has to be running for uploads, and nothing needs an open port.
  * Each pod runs pod/glore_pod.py, which terminates the pod after N idle minutes
    and always at the hard cap. GLORE watches from your PC as a second safety net,
    and stops everything if the monthly limit is reached.

Settings (including keys) are stored in %APPDATA%/GLORE/cloud/settings.json."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

API = "https://api.runpod.io/v2"

# Datacenters where the S3-compatible API works (docs.runpod.io/storage/s3-api).
# EU first: you're in Ireland, so uploads go a shorter way.
S3_DATACENTERS = ["EUR-IS-1", "EUR-NO-1", "EU-RO-1", "EU-CZ-1",
                  "US-KS-2", "US-CA-2", "US-GA-2", "US-IL-1", "US-MD-1", "US-MO-1", "US-MO-2",
                  "US-NC-1", "US-NC-2", "US-NE-1", "US-WA-1"]

DEFAULT_IMAGE = "nvidia/cuda:12.8.1-runtime-ubuntu22.04"

DEFAULTS = {
    "api_key": "",
    "s3_access_key": "",       # your RunPod user id, e.g. user_2abc... (shown next to the S3 key)
    "s3_secret": "",
    "datacenter": "",
    "volume_id": "",
    "volume_gb": 50,
    "idle_minutes": 10,
    "hard_cap_hours": 4.0,
    "monthly_limit": 20.0,     # USD; 0 = no limit
    "gpu": "RTX 5090, A100",   # preference list: first in-stock match wins (5090 32GB ~$1.39/h, A100 80GB ~$2.19/h)
    "variant": "turbo",        # ACE-Step 1.5 standard (2B) DiT variant the LoRA is trained on
    "image": DEFAULT_IMAGE,
}
SECRETS = ("api_key", "s3_secret")
TEST_HARD_CAP_S = 240          # the connection test pod can never run longer than this


class CloudError(RuntimeError):
    pass


def home() -> Path:
    base = os.environ.get("GLORE_HOME") or os.environ.get("APPDATA") or str(Path.home())
    d = Path(base) / "GLORE" / "cloud"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- settings

def load_settings() -> dict:
    p = home() / "settings.json"
    data = dict(DEFAULTS)
    if p.exists():
        try:
            data.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return data


def save_settings(patch: dict) -> dict:
    data = load_settings()
    for k, v in patch.items():
        if k not in DEFAULTS or v is None:
            continue
        if k in SECRETS and isinstance(v, str) and v.startswith("••••"):
            continue  # the masked value came back unchanged
        if isinstance(DEFAULTS[k], (int, float)) and not isinstance(DEFAULTS[k], bool):
            v = type(DEFAULTS[k])(v)
            if v < 0:
                raise CloudError(f"{k} can't be negative")
        if isinstance(v, str):
            v = v.strip()
        data[k] = v
    if data["datacenter"] and data["datacenter"] not in S3_DATACENTERS:
        raise CloudError(f"{data['datacenter']} doesn't support RunPod's S3 API. Use one of: {', '.join(S3_DATACENTERS)}")
    if data["variant"] not in ("turbo", "base", "sft"):
        raise CloudError("variant must be turbo, base or sft")
    (home() / "settings.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def mask(v: str) -> str:
    return f"••••{v[-4:]}" if v else ""


def public_settings(s: dict | None = None) -> dict:
    s = dict(s or load_settings())
    s["missing"] = missing(s)
    for k in SECRETS:
        s[k] = mask(s[k])
    s["s3_datacenters"] = S3_DATACENTERS
    return s


def missing(s: dict) -> list[str]:
    need = []
    if not s.get("api_key"):
        need.append("api_key")
    if not s.get("s3_access_key"):
        need.append("s3_access_key")
    if not s.get("s3_secret"):
        need.append("s3_secret")
    if not s.get("volume_id") or not s.get("datacenter"):
        need.append("volume")
    return need


# ---------------------------------------------------------------- RunPod REST v2

class RunPod:
    def __init__(self, api_key: str):
        if not api_key:
            raise CloudError("Add your RunPod API key first")
        self.key = api_key

    def _req(self, method: str, path: str, body: dict | None = None, query: dict | None = None):
        url = API + path + ("?" + urllib.parse.urlencode(query, doseq=True) if query else "")
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json", "Accept": "application/json",
                                              "User-Agent": "GLORE"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise CloudError(self._problem(e)) from None
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise CloudError(f"Can't reach RunPod: {e.reason}") from None

    @staticmethod
    def _problem(e: urllib.error.HTTPError) -> str:
        try:
            p = json.loads(e.read() or b"{}")
        except (ValueError, OSError):
            p = {}
        if e.code == 401:
            return "RunPod rejected the API key"
        if e.code == 404:
            return "Not found on RunPod"
        detail = p.get("detail") or p.get("title") or p.get("error") or f"RunPod error {e.code}"
        errs = p.get("errors") or []
        if errs:
            detail += ": " + "; ".join(str(x.get("detail") or x.get("message") or x) if isinstance(x, dict) else str(x)
                                        for x in errs[:3])
        return detail

    # catalog
    def gpus(self, include_availability: bool = True) -> list[dict]:
        # availability needs a product context; per-datacenter stock comes back as g["dataCenters"]
        q = {"include": "AVAILABILITY", "product": "POD"} if include_availability else None
        r = self._req("GET", "/catalog/gpus", query=q)
        return r.get("gpus", r if isinstance(r, list) else [])

    def datacenters(self) -> list[dict]:
        r = self._req("GET", "/catalog/datacenters", query={"include": "GPU_AVAILABILITY"})
        return r.get("dataCenters", r.get("datacenters", []))

    # storage
    def create_volume(self, name: str, size_gb: int, dc: str) -> dict:
        return self._req("POST", "/network-volumes",
                         {"name": name, "size": int(size_gb), "dataCenter": dc, "type": "STANDARD"})

    def get_volume(self, vid: str) -> dict:
        return self._req("GET", f"/network-volumes/{vid}")

    # pods
    def create_pod(self, body: dict) -> dict:
        return self._req("POST", "/pods", body)

    def get_pod(self, pid: str) -> dict | None:
        try:
            return self._req("GET", f"/pods/{pid}")
        except CloudError as e:
            if str(e) == "Not found on RunPod":
                return None
            raise

    def list_pods(self) -> list[dict]:
        out, cursor = [], None
        for _ in range(10):
            r = self._req("GET", "/pods", query={"cursor": cursor} if cursor else None)
            out += r.get("pods", [])
            pag = r.get("pagination") or {}
            if not pag.get("hasNextPage"):
                break
            cursor = pag.get("nextCursor")
        return out

    def terminate(self, pid: str) -> None:
        try:
            self._req("POST", f"/pods/{pid}/action", {"action": "terminate"})
        except CloudError as e:
            if str(e) == "Not found on RunPod":
                return
            self._req("DELETE", f"/pods/{pid}")


def gpu_price(g: dict) -> float | None:
    p = g.get("price") or {}
    v = p.get("secure") if isinstance(p, dict) else None
    if isinstance(v, dict):  # tolerate {"secure": {"onDemand": x}}
        v = v.get("onDemand") or v.get("price")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def available_in(g: dict, dc: str) -> bool:
    """Real v2 shape (product=POD): g["dataCenters"] = [{"id": "EUR-IS-1", "availability": "LOW"}, ...]
    for GPUs in stock somewhere; g["availability"] is one overall level ("NONE" = nowhere)."""
    dcs = g.get("dataCenters")
    if isinstance(dcs, list):
        return any(isinstance(a, dict) and a.get("id") == dc
                   and str(a.get("availability", "")).upper() not in ("NONE", "") for a in dcs)
    av = g.get("availability")
    if av is None:
        return True  # not reported: let RunPod decide
    if isinstance(av, str):
        return False  # only an overall level, no datacenter list -> not in stock anywhere we can tell
    for a in av if isinstance(av, list) else []:
        if isinstance(a, dict) and (a.get("dataCenter") or a.get("id")) == dc \
                and str(a.get("availability", "")).upper() not in ("NONE", ""):
            return True
    return False


def gpu_choices(gpus: list[dict], dc: str, wants: str = "A100") -> list[tuple[str, float]]:
    """In-stock GPUs in the volume's datacenter, for a comma-separated preference list
    ("RTX 5090, A100"): every match for the first name (cheapest first), then the next name..."""
    out: list[tuple[str, float]] = []
    names = [w.strip() for w in str(wants).split(",") if w.strip()] or ["A100"]
    for want in names:
        cands = [g for g in gpus if want.lower() in (str(g.get("id", "")) + " " + str(g.get("name", ""))).lower()
                 and g.get("secure", True) is not False and gpu_price(g) and available_in(g, dc)]
        for g in sorted(cands, key=gpu_price):
            if g["id"] not in [o[0] for o in out]:
                out.append((g["id"], gpu_price(g)))
    if not out:
        raise CloudError(f"No {' or '.join(names)} free in {dc} right now. Try again in a few minutes.")
    return out


def pick_gpu(gpus: list[dict], dc: str, want: str = "A100") -> tuple[str, float]:
    return gpu_choices(gpus, dc, want)[0]


def is_capacity_error(e: Exception) -> bool:
    m = str(e).lower()
    return any(k in m for k in ("no longer any instances", "no instances", "not enough", "insufficient",
                                "out of stock", "unavailable", "not available"))


def pick_cheapest(gpus: list[dict], dc: str) -> tuple[str, float]:
    stocked = [g for g in gpus if gpu_price(g) and g.get("secure", True) is not False and available_in(g, dc)]
    if not stocked:
        raise CloudError(f"No GPUs free in {dc} right now. Try again in a few minutes.")
    g = min(stocked, key=gpu_price)
    return g["id"], gpu_price(g)


def choose_datacenter(rp: RunPod, want: str = "H100") -> str:
    """S3-capable datacenter that can hold a standard network volume and has `want` in stock, EU first."""
    dcs = rp.datacenters()
    rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    best, best_score = None, 0
    for dc in dcs:
        if dc.get("id") not in S3_DATACENTERS or "STANDARD" not in (dc.get("networkVolumeTypes") or []):
            continue
        score = max([rank.get(str(a.get("availability", "")).upper(), 0) for a in dc.get("gpuAvailability") or []
                     if want.lower() in (str(a.get("id", "")) + " " + str(a.get("name", ""))).lower()] or [0])
        if not score:
            continue
        score = score * 100 - S3_DATACENTERS.index(dc["id"])
        if score > best_score:
            best, best_score = dc["id"], score
    if not best:
        raise CloudError(f"No datacenter with RunPod's S3 API and standard storage has a {want} free right now. "
                         "Try again later, or set GPU to another card (e.g. A100).")
    return best


def pod_body(s: dict, gpu_id: str, name: str, env: dict) -> dict:
    return {
        "name": name,
        "image": s.get("image") or DEFAULT_IMAGE,
        "gpu": {"id": gpu_id, "count": 1},
        "cloud": "SECURE",
        "dataCenterIds": [s["datacenter"]],
        "mounts": {"network": [{"volumeId": s["volume_id"], "path": "/workspace"}]},
        "disk": 30,
        "env": {k: str(v) for k, v in env.items()},
        "args": "bash /workspace/glore/pod_start.sh",
    }


# ---------------------------------------------------------------- network volume over S3

class Volume:
    """Files on the network volume. key 'a/b.txt' == /workspace/a/b.txt on a pod."""

    def __init__(self, s: dict, client=None):
        if client is None:
            import boto3
            from botocore.config import Config
            dc = s["datacenter"]
            client = boto3.client(
                "s3", endpoint_url=f"https://s3api-{dc.lower()}.runpod.io", region_name=dc,
                aws_access_key_id=s["s3_access_key"], aws_secret_access_key=s["s3_secret"],
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                              retries={"max_attempts": 8, "mode": "standard"},
                              connect_timeout=20, read_timeout=120))
        self.s3 = client
        self.bucket = s["volume_id"]

    def size(self, key: str) -> int | None:
        try:
            return int(self.s3.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
        except Exception as e:
            if _is_missing(e):
                return None
            raise CloudError(_s3_msg(e)) from None

    def put_bytes(self, key: str, data: bytes) -> None:
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        except Exception as e:
            raise CloudError(_s3_msg(e)) from None

    def put_file(self, key: str, path: Path, progress: Callable[[int], None] | None = None) -> None:
        from boto3.s3.transfer import TransferConfig
        # RunPod: single PUT up to 500 MB, parts up to 500 MB
        cfg = TransferConfig(multipart_threshold=400 * 2 ** 20, multipart_chunksize=200 * 2 ** 20, max_concurrency=4)
        try:
            self.s3.upload_file(str(path), self.bucket, key, Config=cfg, Callback=progress)
        except Exception as e:
            raise CloudError(_s3_msg(e)) from None

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:
            if _is_missing(e):
                return None
            raise CloudError(_s3_msg(e)) from None

    def get_json(self, key: str) -> dict | None:
        b = self.get_bytes(key)
        if not b:
            return None
        try:
            return json.loads(b)
        except ValueError:
            return None  # caught mid-write; read again next poll

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.s3.download_file(self.bucket, key, str(dest))
        except Exception as e:
            raise CloudError(_s3_msg(e)) from None

    def list_keys(self, prefix: str) -> list[str]:
        try:
            r = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [o["Key"] for o in r.get("Contents", [])]
        except Exception as e:
            raise CloudError(_s3_msg(e)) from None

    def delete(self, key: str) -> None:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            if not _is_missing(e):
                raise CloudError(_s3_msg(e)) from None

    def tail(self, key: str, n_bytes: int = 4000) -> str:
        try:
            r = self.s3.get_object(Bucket=self.bucket, Key=key, Range=f"bytes=-{n_bytes}")
            return r["Body"].read().decode("utf-8", "replace")
        except Exception:
            return ""


def _is_missing(e: Exception) -> bool:
    code = str(getattr(e, "response", {}).get("Error", {}).get("Code", "")) if hasattr(e, "response") else ""
    return code in ("404", "NoSuchKey", "NotFound") or isinstance(e, (FileNotFoundError, KeyError))


def _s3_msg(e: Exception) -> str:
    resp = getattr(e, "response", None) or {}
    code = resp.get("Error", {}).get("Code", "")
    if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "403", "AccessDenied"):
        return ("RunPod's S3 API refused the key. Check the S3 access key (your user_… id) and secret, "
                "and that your PC's clock is right.")
    if code in ("NoSuchBucket",):
        return "That network volume wasn't found in this datacenter"
    return f"Cloud storage error: {e}"


# ---------------------------------------------------------------- spending ledger

class Ledger:
    """Every pod GLORE starts, with its hourly price, so spend can be shown and capped."""

    def __init__(self, path: Path | None = None):
        self.path = path or home() / "ledger.json"
        self.lock = threading.Lock()

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _save(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, row: dict) -> None:
        with self.lock:
            rows = self._load()
            rows.append(row)
            self._save(rows)

    def update(self, pod_id: str, **kw) -> dict | None:
        with self.lock:
            rows = self._load()
            for r in rows:
                if r["pod_id"] == pod_id:
                    r.update(kw)
                    self._save(rows)
                    return r
        return None

    def rows(self) -> list[dict]:
        with self.lock:
            return self._load()

    def active(self) -> list[dict]:
        return [r for r in self.rows() if not r.get("ended")]

    @staticmethod
    def cost(r: dict, t: float | None = None) -> float:
        end = r.get("ended") or t or time.time()
        return max(0.0, end - r["started"]) / 3600 * float(r.get("cost_per_hr") or 0)

    def month_total(self, t: float | None = None) -> float:
        t = t or time.time()
        m = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m")
        return sum(self.cost(r, t) for r in self.rows()
                   if datetime.fromtimestamp(r["started"], timezone.utc).strftime("%Y-%m") == m)


# ---------------------------------------------------------------- manager

class CloudManager:
    """Starts pods, watches them, and stops them when they should stop."""

    POLL_S = 15
    STALE_S = 20 * 60          # no heartbeat for this long while busy -> stuck, stop it
    BOOT_GRACE_S = 25 * 60     # first-run setup can take a while before the first heartbeat

    def __init__(self, runpod_factory=None, volume_factory=None, ledger: Ledger | None = None,
                 settings_loader=load_settings):
        self.settings = settings_loader
        self.runpod_factory = runpod_factory or (lambda s: RunPod(s["api_key"]))
        self.volume_factory = volume_factory or (lambda s: Volume(s))
        self.ledger = ledger or Ledger()
        self._watching: set[str] = set()
        self._lock = threading.Lock()
        self.heartbeats: dict[str, float] = {}   # pod id -> when we last saw its heartbeat change
        self._hb_values: dict[str, float] = {}

    def rp(self) -> RunPod:
        return self.runpod_factory(self.settings())

    def volume(self) -> Volume:
        s = self.settings()
        need = [m for m in missing(s) if m != "api_key"]
        if need:
            raise CloudError("Finish the cloud settings first (S3 key and network volume)")
        return self.volume_factory(s)

    # spend
    def month_spend(self) -> float:
        return round(self.ledger.month_total(), 4)

    def check_budget(self, max_cost: float) -> None:
        limit = float(self.settings().get("monthly_limit") or 0)
        spent = self.month_spend()
        if limit and spent + max_cost > limit:
            raise CloudError(f"This could cost up to ${max_cost:.2f}, and you've used ${spent:.2f} of your "
                             f"${limit:.2f} monthly limit. Raise the limit or lower the hard cap.")

    # pods
    def start_pod(self, kind: str, gpu_id: str, price: float, hard_cap_s: float, idle_min: float,
                  meta: dict | None = None) -> dict:
        s = self.settings()
        env = {"GLORE_RUNPOD_KEY": s["api_key"], "GLORE_IDLE_MIN": f"{float(idle_min):g}", "GLORE_HARD_CAP_S": int(hard_cap_s)}
        pod = self.rp().create_pod(pod_body(s, gpu_id, f"glore-{kind}", env))
        pid = pod.get("id")
        if not pid:
            raise CloudError("RunPod didn't return a pod id")
        row = {"pod_id": pid, "kind": kind, "gpu": gpu_id, "cost_per_hr": float(pod.get("cost") or price),
               "started": time.time(), "ended": None, "hard_cap_s": hard_cap_s, "datacenter": s["datacenter"],
               **(meta or {})}
        self.ledger.add(row)
        self.watch(pid)
        return row

    def start_pod_any(self, kind: str, choices: list[tuple[str, float]], hard_cap_s: float, idle_min: float,
                      meta: dict | None = None, report=None) -> dict:
        """Try each GPU in order; one selling out between the stock check and the start is common."""
        last = None
        for gpu_id, price in choices:
            if report:
                report(f"Starting a {gpu_id} (${price:.2f}/h)")
            try:
                return self.start_pod(kind, gpu_id, price, hard_cap_s, idle_min, meta)
            except CloudError as e:
                if not is_capacity_error(e):
                    raise
                last = e
        raise CloudError(f"Every matching GPU sold out while starting ({last}). Try again in a few minutes.")

    def reusable_pod(self, kind: str) -> dict | None:
        """A training pod that's already up (e.g. idling after a run) can take the next job."""
        s = self.settings()
        for r in self.ledger.active():
            if r["kind"] != kind or r.get("datacenter") != s["datacenter"]:
                continue
            left = r["hard_cap_s"] - (time.time() - r["started"])
            if left < 3600:
                continue  # not enough time left before its hard cap
            p = self.rp().get_pod(r["pod_id"])
            if p and p.get("status") in ("RUNNING", "STARTING", "PROVISIONING"):
                return r
        return None

    def watch(self, pid: str) -> None:
        with self._lock:
            if pid in self._watching:
                return
            self._watching.add(pid)
        threading.Thread(target=self._watch, args=(pid,), daemon=True).start()

    def resume_watching(self) -> None:
        for r in self.ledger.active():
            self.watch(r["pod_id"])

    def _end(self, pid: str, reason: str) -> None:
        self.ledger.update(pid, ended=time.time(), end_reason=reason)

    def stop(self, pid: str, reason: str) -> None:
        try:
            self.rp().terminate(pid)
        finally:
            self._end(pid, reason)

    def stop_all(self, reason: str = "stopped by you") -> int:
        n = 0
        for r in self.ledger.active():
            try:
                self.stop(r["pod_id"], reason)
                n += 1
            except CloudError:
                pass
        return n

    def _watch(self, pid: str) -> None:
        try:
            while True:
                row = next((r for r in self.ledger.rows() if r["pod_id"] == pid), None)
                if not row or row.get("ended"):
                    return
                if self.check_pod(row):
                    return
                time.sleep(self.POLL_S)
        finally:
            with self._lock:
                self._watching.discard(pid)

    def check_pod(self, row: dict, t: float | None = None) -> bool:
        """One safety check. Returns True once the pod is gone (or has been stopped)."""
        t = t or time.time()
        pid = row["pod_id"]
        try:
            pod = self.rp().get_pod(pid)
        except CloudError:
            return False  # network blip: try again next poll
        if pod is None or pod.get("status") == "TERMINATED":
            self._end(pid, "shut itself down")
            return True
        if pod.get("status") in ("EXITED", "ERROR"):
            self.stop(pid, "pod exited")
            return True
        age = t - row["started"]
        if age > row["hard_cap_s"] + 120:
            self.stop(pid, "hard cap")
            return True
        limit = float(self.settings().get("monthly_limit") or 0)
        if limit and self.ledger.month_total(t) >= limit:
            self.stop(pid, "monthly limit")
            return True
        self.refresh_heartbeat(pid)
        hb = self.heartbeats.get(pid)
        if age > self.BOOT_GRACE_S and (hb is None or t - hb > self.STALE_S):
            self.stop(pid, "stopped responding")
            return True
        return False

    def refresh_heartbeat(self, pid: str) -> None:
        """The pod writes glore/pods/<id>.json every 15 s. Note when it last *changed*, by
        our own clock, so a wrong clock on either side can't confuse the check."""
        try:
            hb = (self.volume().get_json(f"glore/pods/{pid}.json") or {}).get("heartbeat")
        except CloudError:
            return
        if hb is not None and hb != self._hb_values.get(pid):
            self._hb_values[pid] = hb
            self.heartbeats[pid] = time.time()

    def status(self) -> dict:
        t = time.time()
        active = []
        for r in self.ledger.active():
            active.append({**r, "cost_so_far": round(self.ledger.cost(r, t), 3),
                           "runtime_s": round(t - r["started"]),
                           "cap_left_s": round(r["hard_cap_s"] - (t - r["started"]))})
        s = self.settings()
        return {"active": active, "month_spend": self.month_spend(),
                "monthly_limit": float(s.get("monthly_limit") or 0),
                "configured": not missing(s), "missing": missing(s)}
