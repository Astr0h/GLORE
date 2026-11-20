"""The PC-free cloud CLI (GitHub Actions): train from the volume, generate on the pod, ledger on the volume."""
import json
import sys

import pytest

from glore_engine import cloud, cloud_cli
from glore_engine.cloud import CloudError, CloudManager
from glore_engine.pod import glore_pod
from test_cloud import FakeAce, FakeRunPod, FakeVolume, SETTINGS

GEN = r'''
import json, sys, wave, os
job = json.load(open(sys.argv[1]))
os.makedirs(job["out_dir"], exist_ok=True)
p = os.path.join(job["out_dir"], "take.wav")
w = wave.open(p, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\0\0" * 800); w.close()
json.dump(job, open(os.path.join(job["out_dir"], "seen.json"), "w"))
print("GLORE_RESULT " + json.dumps({"files": [{"path": p, "seed": job["seed"]}], "planned_caption": "planned: " + job["caption"]}))
'''
COPY = "import sys, shutil; a = sys.argv; shutil.copy(a[a.index('-i') + 1], a[-1])"


class GenAce(FakeAce):
    def ensure_extra_model(self, name, log, status):
        self.extra = name

    def py(self, *args):
        if args and str(args[0]).endswith("glore_generate.py"):
            return [sys.executable, "-c", GEN, *args[1:]]
        return super().py(*args)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    vol = FakeVolume(tmp_path / "volume")
    ace = GenAce(tmp_path)
    rp = FakeRunPod(vol, ace)
    s = {**cloud.DEFAULTS, **SETTINGS, "gpu": "H100", "monthly_limit": 100.0}
    mgr = CloudManager(runpod_factory=lambda _s: rp, volume_factory=lambda _s: vol, settings_loader=lambda: s,
                       ledger=cloud_cli.VolumeLedger(vol))
    mgr.POLL_S = 0.05
    monkeypatch.setattr(cloud_cli, "manager", lambda s=None: mgr)
    monkeypatch.setattr(glore_pod, "FFMPEG", [sys.executable, "-c", COPY])
    return {"vol": vol, "rp": rp, "ace": ace, "mgr": mgr, "tmp": tmp_path}


def put_dataset(vol, n=3):
    doc = {"metadata": {"custom_tag": "glore_house", "audio": "full", "genre_ratio": 30},
           "samples": [{"id": f"t{i}", "audio_path": f"./t{i}_full.flac", "caption": "house"} for i in range(n)]}
    vol.put_bytes("datasets/house/dataset.json", json.dumps(doc).encode())
    for i in range(n):
        vol.put_bytes(f"datasets/house/t{i}_full.flac", b"flac")
    return doc


def wait_done(vol, job_id, timeout=20):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = cloud_cli.job_status(vol, job_id)
        if st.get("phase") in ("done", "error", "stopped"):
            return st
        time.sleep(0.05)
    raise AssertionError("job didn't finish")


def test_train_from_the_volume_without_a_pc(cli, capsys):
    put_dataset(cli["vol"])
    rc = cloud_cli.main(["train", "--variant", "xl_sft", "--epochs", "4", "--save-every", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    jid = json.loads(out[out.index("{"):])["job"]
    st = wait_done(cli["vol"], jid)
    assert st["phase"] == "done", st
    job = json.loads(cli["vol"].get_bytes(f"glore/runs/{jid}/job.json"))
    assert job["variant"] == "xl_sft" and job["tensor_dir"].endswith("-xl_sft") and job["autolabel"]
    assert "--model-variant" in cli["ace"].cmds[-1] and "xl_sft" in cli["ace"].cmds[-1]
    assert job["run_id"].endswith("-xl_sft-r64-e4-full-cap")
    # the ledger lives on the volume, so spend survives the runner going away
    rows = json.loads(cli["vol"].get_bytes(cloud_cli.LEDGER_KEY))
    assert rows[0]["kind"] == "train" and rows[0]["hard_cap_s"] == 8 * 3600
    loras = cloud_cli.cmd_loras(type("A", (), {"pack": "house"}))
    assert sorted(loras[job["run_id"]]) == ["e2", "e4", "final"]
    # same data + settings again = refused, nothing started
    assert cloud_cli.main(["train", "--variant", "xl_sft", "--epochs", "4", "--save-every", "2"]) == 1


def test_train_needs_the_dataset_on_the_volume(cli, capsys):
    assert cloud_cli.main(["train"]) == 1
    assert "No dataset" in capsys.readouterr().err


def test_generate_on_the_pod_and_fetch(cli, tmp_path):
    vol = cli["vol"]
    vol.put_bytes("loras/house/runX/final/adapter_model.safetensors", b"w")
    vol.put_bytes("loras/house/runX/checkpoints/epoch_100_loss_0.8/adapter_model.safetensors", b"w")
    takes = [{"name": "e100 fred", "caption": "emotional uk house", "lora": "runX@e100", "seed": 7, "duration": 30,
              "bpm": 129, "key": "E minor"},
             {"name": "no lora", "caption": "emotional uk house", "seed": 7, "duration": 30}]
    tf = tmp_path / "takes.json"
    tf.write_text(json.dumps(takes))
    out = tmp_path / "out"
    assert cloud_cli.main(["generate", "--takes", str(tf), "--wait", "--out", str(out), "--timeout-min", "1"]) == 0
    res = json.loads((out / "result.json").read_text())
    names = sorted(p.name for p in out.glob("*.mp3"))
    assert names == ["e100 fred.mp3", "no lora.mp3"]
    t0 = res["result"]["takes"][0]
    assert t0["planned_caption"] == "planned: emotional uk house" and res["result"]["lm"] == glore_pod.CLOUD_LM
    assert cli["ace"].extra == glore_pod.CLOUD_LM


def test_generate_rejects_unknown_checkpoint(cli, tmp_path, capsys):
    cli["vol"].put_bytes("loras/house/runX/final/adapter_model.safetensors", b"w")
    tf = tmp_path / "t.json"
    tf.write_text(json.dumps([{"name": "x", "caption": "c", "lora": "runX@e999"}]))
    assert cloud_cli.main(["generate", "--takes", str(tf)]) == 1
    assert "No epoch 999" in capsys.readouterr().err


def test_env_settings_need_the_secrets(monkeypatch):
    for k in ("GLORE_RUNPOD_API_KEY", "GLORE_S3_ACCESS_KEY", "GLORE_S3_SECRET", "GLORE_VOLUME_ID", "GLORE_DATACENTER"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(CloudError, match="repo secrets"):
        cloud_cli.env_settings()
    for k, v in (("GLORE_RUNPOD_API_KEY", "rpa_x"), ("GLORE_S3_ACCESS_KEY", "user_x"), ("GLORE_S3_SECRET", "s"),
                 ("GLORE_VOLUME_ID", "v"), ("GLORE_DATACENTER", "EUR-IS-1")):
        monkeypatch.setenv(k, v)
    s = cloud_cli.env_settings()
    assert s["datacenter"] == "EUR-IS-1" and "A100" in s["gpu"] and s["monthly_limit"] == 30
