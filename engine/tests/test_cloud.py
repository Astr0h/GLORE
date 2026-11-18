"""Cloud LoRA training. RunPod and its S3 storage are replaced by fakes: the fake
volume is a folder, and the fake pod runs the real glore_pod job loop on it with a
tiny stand-in trainer, so the whole path (upload -> queue -> train -> status ->
download) is exercised without a network or a GPU."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from glore_engine import cloud, datasets, training
from glore_engine.cloud import CloudError, CloudManager, Ledger
from glore_engine.pod import glore_pod
from glore_engine.project import ProjectStore
from glore_engine.server import create_app


# ---------------------------------------------------------------- fakes

class FakeVolume:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.puts = []

    def _p(self, key):
        return self.root / key

    def size(self, key):
        p = self._p(key)
        return p.stat().st_size if p.exists() else None

    def put_bytes(self, key, data):
        self._p(key).parent.mkdir(parents=True, exist_ok=True)
        self._p(key).write_bytes(data)
        self.puts.append(key)

    def put_file(self, key, path, progress=None):
        data = Path(path).read_bytes()
        self.put_bytes(key, data)
        if progress:
            progress(len(data))

    def get_bytes(self, key):
        p = self._p(key)
        return p.read_bytes() if p.exists() else None

    def get_json(self, key):
        b = self.get_bytes(key)
        try:
            return json.loads(b) if b else None
        except ValueError:
            return None

    def download(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._p(key).read_bytes())

    def list_keys(self, prefix):
        base = self._p(prefix)
        return [str(f.relative_to(self.root)).replace("\\", "/") for f in base.rglob("*") if f.is_file()] \
            if base.exists() else []

    def delete(self, key):
        self._p(key).unlink(missing_ok=True)

    def tail(self, key, n=4000):
        b = self.get_bytes(key) or b""
        return b[-n:].decode()


TRAINER = r"""
import sys, time, pathlib
out = pathlib.Path(sys.argv[1]); epochs = int(sys.argv[2]); delay = float(sys.argv[3])
for e in range(1, epochs + 1):
    print(f"[OK] Epoch {e}/{epochs} in 0.1s, Loss: {1.0 / e:.4f}", flush=True)
    time.sleep(delay)
(out / "final").mkdir(parents=True, exist_ok=True)
(out / "final" / "adapter_config.json").write_text('{"r": 64}')
(out / "final" / "adapter_model.safetensors").write_bytes(b"weights")
"""


class FakeAce(glore_pod.Ace):
    def __init__(self, tmp, delay=0.0):
        super().__init__(tmp / "ace", ["uv"], dict(os.environ))
        self.delay = delay
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cmds = []

    def ensure(self, variant, log, status, extras=()):
        status.update(phase="setup", message="pretend install")

    def preprocess(self, dataset_json, tensor_dir, variant, log, status, max_duration=240.0):
        assert json.loads(dataset_json.read_text())["samples"]
        tensor_dir.mkdir(parents=True, exist_ok=True)
        (tensor_dir / "a.pt").write_bytes(b"t")

    def train_cmd(self, tensor_dir, out_dir, variant, p):
        self.cmds.append(super().train_cmd(tensor_dir, out_dir, variant, p))
        return [sys.executable, "-c", TRAINER, str(out_dir), str(p["epochs"]), str(self.delay)]


class FakeRunPod:
    # shapes copied from a real GET /v2/catalog/gpus?include=AVAILABILITY&product=POD (18 Nov 2026)
    GPUS = [
        {"availability": "LOW", "id": "NVIDIA H100 80GB HBM3", "name": "H100 SXM", "secure": True,
         "price": {"community": 2.69, "secure": 4.39, "serverless": 5.89},
         "dataCenters": [{"availability": "LOW", "id": "US-GA-2", "name": "US-GA-2"},
                         {"availability": "LOW", "id": "EU-RO-1", "name": "EU-RO-1"}]},
        {"availability": "NONE", "id": "NVIDIA H100 PCIe", "name": "H100 PCIe", "secure": True,
         "price": {"community": 1.99, "secure": 2.89, "serverless": 5.89}},
        {"availability": "LOW", "id": "NVIDIA A100-SXM4-80GB", "name": "A100 SXM", "secure": True,
         "price": {"community": 1.39, "secure": 2.19, "serverless": 2.99},
         "dataCenters": [{"availability": "LOW", "id": "EUR-IS-1", "name": "EUR-IS-1"},
                         {"availability": "LOW", "id": "EU-RO-1", "name": "EU-RO-1"}]},
        {"availability": "NONE", "id": "NVIDIA A100-SXM4-40GB", "name": "A100 SXM 40GB", "secure": False,
         "price": {"community": 1, "secure": 0}},
        {"availability": "MEDIUM", "id": "NVIDIA RTX A4000", "name": "A4000", "secure": True,
         "price": {"community": 0.09, "secure": 0.17},
         "dataCenters": [{"availability": "MEDIUM", "id": "EU-RO-1", "name": "EU-RO-1"}]},
    ]

    def __init__(self, vol: FakeVolume, ace: FakeAce):
        self.vol, self.ace = vol, ace
        self.pods, self.bodies = {}, []
        self.sold_out = set()   # gpu ids that fail at create time

    def gpus(self, include_availability=True):
        return self.GPUS

    def datacenters(self):
        return [{"id": "CA-MTL-1", "networkVolumeTypes": ["STANDARD"],     # no S3 API
                 "gpuAvailability": [{"id": "NVIDIA A100-SXM4-80GB", "name": "A100 SXM", "availability": "HIGH"}]},
                {"id": "US-KS-2", "networkVolumeTypes": [],                # can't hold a volume
                 "gpuAvailability": [{"id": "NVIDIA A100-SXM4-80GB", "name": "A100 SXM", "availability": "HIGH"}]},
                {"id": "EU-RO-1", "networkVolumeTypes": ["STANDARD"],
                 "gpuAvailability": [{"id": "NVIDIA A100-SXM4-80GB", "name": "A100 SXM", "availability": "LOW"}]},
                {"id": "EUR-NO-1", "networkVolumeTypes": ["STANDARD"],
                 "gpuAvailability": [{"id": "NVIDIA RTX 4090", "name": "RTX 4090", "availability": "LOW"}]}]

    def create_volume(self, name, size, dc):
        return {"id": "vol123", "name": name, "size": size, "dataCenter": dc}

    def create_pod(self, body):
        if body["gpu"]["id"] in self.sold_out:
            raise CloudError("There are no longer any instances available with the requested specifications. "
                             "Please refresh and try again.")
        pid = f"pod{len(self.pods) + 1}"
        self.bodies.append(body)
        self.pods[pid] = "RUNNING"
        threading.Thread(target=self._run, args=(pid,), daemon=True).start()
        return {"id": pid, "cost": 2.69, "status": "STARTING"}

    def _run(self, pid):
        state = {"busy": False, "idle_since": time.time(), "status": None, "stop": False}
        queue = self.vol.root / "glore" / "queue"
        while self.pods.get(pid) == "RUNNING":
            job = glore_pod.next_job(queue) if queue.exists() else None
            if job:
                glore_pod.process_job(job, self.vol.root, self.ace, state)
            time.sleep(0.02)

    def get_pod(self, pid):
        st = self.pods.get(pid)
        return None if st is None else {"id": pid, "status": st, "cost": 2.69}

    def terminate(self, pid):
        self.pods[pid] = "TERMINATED"


SETTINGS = {"api_key": "rpa_secretkey1234", "s3_access_key": "user_abc", "s3_secret": "s3secret9876",
            "datacenter": "EU-RO-1", "volume_id": "vol123"}


@pytest.fixture
def cl(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "POLL_S", 0.05)
    cloud.save_settings({**SETTINGS, "gpu": "H100"})
    vol = FakeVolume(tmp_path / "volume")
    ace = FakeAce(tmp_path)
    rp = FakeRunPod(vol, ace)
    mgr = CloudManager(runpod_factory=lambda s: rp, volume_factory=lambda s: vol)
    mgr.POLL_S = 0.05
    return {"mgr": mgr, "rp": rp, "vol": vol, "ace": ace, "tmp": tmp_path}


def make_pack(pack, n, tmp, prepared=True):
    sr = 44100
    tmp.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        wav = tmp / f"t{i}.wav"
        t = np.arange(sr // 4) / sr
        sf.write(wav, np.stack([np.sin(2 * np.pi * (220 + i) * t)] * 2, axis=1).astype(np.float32) * 0.3, sr)
        it = datasets.add_or_update(pack, wav, f"Track {i}", ["soft"] if i % 2 else ["hype"], bpm=128, key="A minor")
        if prepared:
            d = datasets.pack_dir(pack)
            (d / f"{it['id']}_train.wav").write_bytes((d / f"{it['id']}.wav").read_bytes())
            it["prepared"] = True
            (d / f"{it['id']}.json").write_text(json.dumps(it))


def collect(report_log):
    return lambda p, m: report_log.append((p, m))


# ---------------------------------------------------------------- settings

def test_settings_masked_and_kept():
    pub = cloud.public_settings(cloud.save_settings(SETTINGS))
    assert pub["api_key"] == "••••1234" and pub["s3_secret"] == "••••9876"
    assert pub["missing"] == []
    # the masked value coming back from the UI must not overwrite the real key
    cloud.save_settings({"api_key": "••••1234", "idle_minutes": "15"})
    s = cloud.load_settings()
    assert s["api_key"] == SETTINGS["api_key"] and s["idle_minutes"] == 15


def test_settings_validation():
    with pytest.raises(CloudError):
        cloud.save_settings({"datacenter": "CA-MTL-1"})  # no S3 API there
    with pytest.raises(CloudError):
        cloud.save_settings({"variant": "xl"})
    with pytest.raises(CloudError):
        cloud.save_settings({"monthly_limit": -1})
    assert set(cloud.missing({})) == {"api_key", "s3_access_key", "s3_secret", "volume"}


# ---------------------------------------------------------------- choosing GPUs / datacenters

def test_pick_gpu_uses_per_datacenter_stock():
    assert cloud.pick_gpu(FakeRunPod.GPUS, "EU-RO-1", "H100") == ("NVIDIA H100 80GB HBM3", 4.39)  # PCIe: none
    assert cloud.pick_gpu(FakeRunPod.GPUS, "EUR-IS-1", "A100") == ("NVIDIA A100-SXM4-80GB", 2.19)  # 40GB: no secure
    with pytest.raises(CloudError, match="No H100 free"):
        cloud.pick_gpu(FakeRunPod.GPUS, "EUR-IS-1", "H100")
    assert cloud.pick_cheapest(FakeRunPod.GPUS, "EU-RO-1")[0] == "NVIDIA RTX A4000"
    assert not cloud.available_in({"availability": "LOW"}, "EU-RO-1")  # no datacenter list -> unknown, skip


def test_choose_datacenter_needs_s3_and_volumes(cl):
    # CA-MTL-1 has no S3 API, US-KS-2 can't hold a volume
    assert cloud.choose_datacenter(cl["rp"], "A100") == "EU-RO-1"
    with pytest.raises(CloudError, match="No datacenter"):
        cloud.choose_datacenter(cl["rp"], "H100")


def test_pod_body():
    b = cloud.pod_body({**cloud.DEFAULTS, **SETTINGS}, "NVIDIA H100 80GB HBM3", "glore-train",
                       {"GLORE_IDLE_MIN": 10})
    assert b["gpu"] == {"id": "NVIDIA H100 80GB HBM3", "count": 1}
    assert b["mounts"]["network"] == [{"volumeId": "vol123", "path": "/workspace"}]
    assert b["dataCenterIds"] == ["EU-RO-1"] and b["cloud"] == "SECURE"
    assert b["env"] == {"GLORE_IDLE_MIN": "10"}
    assert "pod_start.sh" in b["args"]


# ---------------------------------------------------------------- spend

def test_ledger_and_budget(tmp_path):
    led = Ledger(tmp_path / "l.json")
    t = time.time()
    led.add({"pod_id": "a", "started": t - 3600, "ended": t, "cost_per_hr": 2.5})
    led.add({"pod_id": "b", "started": t - 1800, "ended": None, "cost_per_hr": 3.0})
    assert led.month_total(t) == pytest.approx(4.0, abs=0.01)
    cloud.save_settings({"monthly_limit": 10})
    mgr = CloudManager(ledger=led, runpod_factory=lambda s: None, volume_factory=lambda s: None)
    mgr.check_budget(5.0)
    with pytest.raises(CloudError, match="monthly limit"):
        mgr.check_budget(6.5)


def test_safety_net(cl):
    mgr, rp = cl["mgr"], cl["rp"]
    rp.pods["p"] = "RUNNING"
    t = time.time()
    # hard cap
    mgr.ledger.add({"pod_id": "p", "started": t - 4000, "ended": None, "cost_per_hr": 1, "hard_cap_s": 3600})
    assert mgr.check_pod(mgr.ledger.rows()[-1], t) and rp.pods["p"] == "TERMINATED"
    assert mgr.ledger.rows()[-1]["end_reason"] == "hard cap"
    # stuck: past the boot grace with no heartbeat
    rp.pods["q"] = "RUNNING"
    mgr.ledger.add({"pod_id": "q", "started": t - mgr.BOOT_GRACE_S - 60, "ended": None, "cost_per_hr": 1,
                    "hard_cap_s": 99999})
    assert mgr.check_pod(mgr.ledger.rows()[-1], t) and rp.pods["q"] == "TERMINATED"
    # healthy pod with a fresh heartbeat keeps running
    rp.pods["r"] = "RUNNING"
    cl["vol"].put_bytes("glore/pods/r.json", json.dumps({"heartbeat": 123}).encode())
    mgr.ledger.add({"pod_id": "r", "started": t - mgr.BOOT_GRACE_S - 60, "ended": None, "cost_per_hr": 1,
                    "hard_cap_s": 99999})
    assert not mgr.check_pod(mgr.ledger.rows()[-1], t) and rp.pods["r"] == "RUNNING"
    # pod already gone -> just recorded
    rp.pods["r"] = "TERMINATED"
    assert mgr.check_pod(mgr.ledger.rows()[-1], t)
    assert mgr.ledger.rows()[-1]["end_reason"] == "shut itself down"


def test_monthly_limit_stops_pod(cl):
    mgr, rp = cl["mgr"], cl["rp"]
    cloud.save_settings({"monthly_limit": 1.0})
    rp.pods["m"] = "RUNNING"
    t = time.time()
    mgr.ledger.add({"pod_id": "m", "started": t - 3600, "ended": None, "cost_per_hr": 2.0, "hard_cap_s": 99999})
    assert mgr.check_pod(mgr.ledger.rows()[-1], t)
    assert mgr.ledger.rows()[-1]["end_reason"] == "monthly limit"


# ---------------------------------------------------------------- trainer helpers (pod side)

def test_parse_progress_and_epochs():
    p = glore_pod.parse_progress("2026-11-17 [INFO] train: Epoch 12/80, Step 240, Loss: 0.4123")
    assert p == {"epoch": 12, "epochs": 80, "step": 240, "loss": 0.4123}
    assert glore_pod.parse_progress("[OK] Epoch 3/80 in 12.1s, Loss: 0.5")["epoch"] == 3
    assert glore_pod.parse_progress("loading model") is None
    assert glore_pod.default_epochs(95) == 80 and glore_pod.default_epochs(30) == 150


def test_train_cmd_resumes_from_latest_checkpoint(tmp_path):
    ace = glore_pod.Ace(tmp_path / "ace", ["uv"], {})
    out = tmp_path / "out"
    for n in (10, 30, 20):
        (out / "checkpoints" / f"epoch_{n}").mkdir(parents=True)
        (out / "checkpoints" / f"epoch_{n}" / "training_state.pt").write_bytes(b"x")
    (out / "checkpoints" / "epoch_40").mkdir()  # no training state: ignored
    cmd = ace.train_cmd(tmp_path / "t", out, "turbo", training.train_params(95))
    s = " ".join(cmd)
    assert "--resume-from" in cmd and cmd[cmd.index("--resume-from") + 1].endswith("epoch_30")
    assert "--no-sync" in cmd and "fixed" in cmd and "--yes" in cmd
    assert "--model-variant turbo" in s and "--shift 3.0" in s and "--epochs 80" in s and "--rank 64" in s


# ---------------------------------------------------------------- bundle

def test_bundle_flac_and_captions(tmp_path):
    make_pack("house", 3, tmp_path)
    make_pack("house", 1, tmp_path / "x", prepared=False)
    b = training.build_bundle("house")
    assert b.count == 3  # the unprepared track is left out
    doc = json.loads(b.dataset_json)
    s0 = doc["samples"][0]
    assert doc["metadata"]["custom_tag"] == "glore_house"
    assert s0["is_instrumental"] and s0["lyrics"] == "[Instrumental]" and s0["bpm"] == 128
    assert s0["caption"].startswith("house, ") and s0["keyscale"] == "A minor"
    f = b.files[0][0]
    info = sf.info(str(f))
    assert info.format == "FLAC" and info.subtype == "PCM_24"
    wav = datasets.pack_dir("house") / f"{f.stem}_train.wav"
    a, _ = sf.read(wav)
    c, _ = sf.read(f)
    assert np.max(np.abs(a - c)) < 1e-6  # lossless at 24-bit
    assert training.build_bundle("house").hash == b.hash  # stable -> same run id -> resumable


# ---------------------------------------------------------------- end to end

def test_train_cloud_end_to_end(cl, monkeypatch):
    monkeypatch.setattr(datasets, "MIN_TRACKS", 3)
    cl["ace"].delay = 0.15
    make_pack("house", 4, cl["tmp"])
    log = []
    info = training.train_cloud("house", cl["mgr"], collect(log), epochs=5)
    dest = Path(training.list_loras("house")[0]["path"])
    assert (dest / "adapter_model.safetensors").read_bytes() == b"weights"
    assert info["variant"] == "turbo" and info["base_model"] == "acestep-v15-turbo"
    assert info["epochs"] == 5 and info["trigger"] == "glore_house" and info["tracks"] == 4
    assert any("Epoch" in m for _, m in log) and any("$" in m for _, m in log)
    body = cl["rp"].bodies[0]
    assert body["env"]["GLORE_IDLE_MIN"] == "10" and body["env"]["GLORE_HARD_CAP_S"] == "14400"
    assert (cl["vol"].root / "glore" / "glore_pod.py").exists()
    assert b"\r\n" not in (cl["vol"].root / "glore" / "pod_start.sh").read_bytes()
    # second pack reuses the idling pod instead of starting another
    make_pack("rnb_rap", 3, cl["tmp"] / "r")
    training.train_cloud("rnb_rap", cl["mgr"], collect([]), epochs=2)
    assert len(cl["rp"].bodies) == 1
    # same tracks + settings again: refused rather than paying twice
    with pytest.raises(CloudError, match="already trained"):
        training.train_cloud("house", cl["mgr"], collect([]), epochs=5)
    # uploads skip what's already there
    n_before = len([k for k in cl["vol"].puts if k.endswith(".flac")])
    b = training.build_bundle("house")
    res = training.upload_bundle(cl["vol"], b, collect([]))
    assert res["uploaded"] == 0 and res["already_there"] == 4
    assert len([k for k in cl["vol"].puts if k.endswith(".flac")]) == n_before


def test_train_needs_enough_tracks(cl):
    make_pack("house", 2, cl["tmp"])
    with pytest.raises(CloudError, match="Prepare at least"):
        training.train_cloud("house", cl["mgr"], collect([]))


def test_pod_stopping_midway_is_reported(cl, monkeypatch):
    monkeypatch.setattr(datasets, "MIN_TRACKS", 1)
    cl["ace"].delay = 0.2
    make_pack("house", 1, cl["tmp"])
    errs = []

    def run():
        try:
            training.train_cloud("house", cl["mgr"], collect([]), epochs=50)
        except CloudError as e:
            errs.append(str(e))

    th = threading.Thread(target=run)
    th.start()
    time.sleep(1.5)
    cl["mgr"].stop_all()
    th.join(20)
    assert errs and "stopped" in errs[0].lower()
    assert all(r.get("ended") for r in cl["mgr"].ledger.rows())


def test_connection_test(cl):
    res = training.connection_test(cl["mgr"], collect([]))
    assert res["ok"] and res["datacenter"] == "EU-RO-1"
    assert cl["rp"].bodies[0]["gpu"]["id"] == "NVIDIA RTX A4000"  # cheapest in stock
    assert cl["rp"].bodies[0]["env"]["GLORE_HARD_CAP_S"] == str(cloud.TEST_HARD_CAP_S)
    assert cl["rp"].pods["pod1"] == "TERMINATED"


def test_setup_volume(cl):
    cloud.save_settings({"volume_id": "", "datacenter": "", "gpu": "A100"})
    res = training.setup_volume(cl["mgr"], collect([]))
    assert res == {"volume_id": "vol123", "datacenter": "EU-RO-1"}
    assert cloud.load_settings()["volume_id"] == "vol123"


def test_fit_test_times_out_and_reports(cl, monkeypatch, tmp_path):
    monkeypatch.setattr(glore_pod, "gpu_info", lambda: {"name": "RTX 4060 Ti", "used_mb": 7000, "total_mb": 8188})
    ace = FakeAce(tmp_path, delay=0.3)
    monkeypatch.setattr(glore_pod, "Ace", lambda *a, **k: ace)
    make_pack("house", 3, tmp_path)
    t0 = time.time()
    res = training.fit_test("house", collect([]), train_seconds=2.5)
    assert time.time() - t0 < 15  # killed at the time limit
    assert res["fits"] == "yes" and res["steps"] >= 2 and res["peak_vram_gb"] == pytest.approx(6.84, 0.01)
    assert res["settings"]["optimizer"] == "adamw8bit"
    assert "full_run_hours_here" in res


# ---------------------------------------------------------------- HTTP API

def test_cloud_endpoints(tmp_path):
    c = TestClient(create_app(ProjectStore(tmp_path / "projects")))
    st = c.get("/cloud/settings").json()
    assert "api_key" in st["missing"] and "EU-RO-1" in st["s3_datacenters"]
    assert c.post("/datasets/house/train", json={}).status_code == 400  # not set up yet
    r = c.put("/cloud/settings", json={**SETTINGS, "monthly_limit": 15})
    assert r.status_code == 200 and r.json()["api_key"] == "••••1234" and r.json()["missing"] == []
    assert c.put("/cloud/settings", json={"datacenter": "XX-1"}).status_code == 400
    s = c.get("/cloud/status").json()
    assert s["monthly_limit"] == 15 and s["active"] == [] and s["configured"]
    assert c.post("/datasets/house/train", json={"rank": 7}).status_code == 400
    packs = c.get("/packs").json()
    assert all("loras" in p for p in packs)


def test_volume_uses_runpod_s3_endpoint():
    from botocore.stub import Stubber
    v = cloud.Volume({**SETTINGS})
    assert v.s3.meta.endpoint_url == "https://s3api-eu-ro-1.runpod.io"
    assert v.s3.meta.region_name == "EU-RO-1" and v.bucket == "vol123"
    with Stubber(v.s3) as st:
        st.add_client_error("head_object", service_error_code="404", http_status_code=404)
        st.add_response("head_object", {"ContentLength": 42})
        st.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
        st.add_client_error("put_object", service_error_code="SignatureDoesNotMatch", http_status_code=403)
        assert v.size("a.flac") is None
        assert v.size("a.flac") == 42
        assert v.get_json("glore/runs/x/status.json") is None
        with pytest.raises(CloudError, match="refused the key"):
            v.put_bytes("x", b"1")


def test_sold_out_gpu_falls_back_and_leaves_no_stale_job(cl, monkeypatch):
    monkeypatch.setattr(datasets, "MIN_TRACKS", 1)
    cloud.save_settings({"gpu": "H100, A100", "datacenter": "EU-RO-1"})
    make_pack("house", 1, cl["tmp"])
    # a job left behind by an earlier failed start must not run again
    cl["vol"].put_bytes("glore/queue/stale.json", json.dumps({"id": "stale", "mode": "test"}).encode())
    cl["rp"].sold_out = {"NVIDIA H100 80GB HBM3"}
    training.train_cloud("house", cl["mgr"], collect([]), epochs=2)
    assert [b["gpu"]["id"] for b in cl["rp"].bodies] == ["NVIDIA A100-SXM4-80GB"]
    assert not (cl["vol"].root / "glore" / "runs" / "stale").exists()
    # everything sold out: clear error, and the queued job is removed
    cl["rp"].sold_out = {"NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"}
    cl["mgr"].stop_all()
    make_pack("rnb_rap", 1, cl["tmp"] / "r")
    with pytest.raises(CloudError, match="sold out"):
        training.train_cloud("rnb_rap", cl["mgr"], collect([]), epochs=2)
    assert cl["vol"].list_keys("glore/queue/") == []


def test_gpu_choices_order():
    ch = cloud.gpu_choices(FakeRunPod.GPUS, "EU-RO-1", "A4000, H100, A100")
    assert [c[0] for c in ch] == ["NVIDIA RTX A4000", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"]


def test_trainer_runs_from_a_folder_holding_ace_and_data(tmp_path):
    # ACE-Step's path_safety only accepts paths under the trainer's cwd
    ace = glore_pod.Ace(tmp_path / "ace", ["uv"], {})
    tensors, out = tmp_path / "tensors" / "house" / "h", tmp_path / "loras" / "house" / "r"
    assert ace.workdir(tensors, out) == tmp_path.resolve()
    cmd = ace.train_cmd(tensors, out, "turbo", training.train_params(10))
    assert cmd[:6] == ["uv", "run", "--no-sync", "--project", str(tmp_path / "ace"), "python"]
    assert cmd[6] == str(tmp_path / "ace" / "train.py")


def test_follow_survives_storage_read_blips(cl, monkeypatch):
    monkeypatch.setattr(datasets, "MIN_TRACKS", 1)
    cl["ace"].delay = 0.1
    make_pack("house", 1, cl["tmp"])
    real = cl["vol"].get_json
    calls = {"n": 0}

    def flaky(key):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise CloudError("Cloud storage error: Expected checksum Ndw89w== did not match")
        return real(key)

    cl["vol"].get_json = flaky
    info = training.train_cloud("house", cl["mgr"], collect([]), epochs=5)
    assert info["epochs"] == 5 and calls["n"] >= 3


def test_volume_disables_crc_checksums():
    v = cloud.Volume({**SETTINGS})
    assert v.s3.meta.config.response_checksum_validation == "when_required"
    assert v.s3.meta.config.request_checksum_calculation == "when_required"
