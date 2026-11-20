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
    assert "--model-variant" in cli["ace"].cmds[-1] and "acestep-v15-xl-sft" in cli["ace"].cmds[-1]
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


def test_state_for_the_web_app_is_encrypted_and_has_play_links(cli, tmp_path, monkeypatch):
    import base64
    import hashlib
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    vol = cli["vol"]

    class FakeS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            return f"https://s3.example/{Params['Key']}?X-Expires={ExpiresIn}"
    vol.s3, vol.bucket = FakeS3(), "vol123"
    vol.put_bytes("loras/house/runX/final/adapter_model.safetensors", b"w")
    vol.put_bytes("loras/house/runX/checkpoints/epoch_50_loss_0.9/adapter_model.safetensors", b"w")
    takes = [{"name": "t1", "caption": "warm house", "lora": "runX@e50", "seed": 1, "duration": 30}]
    tf = tmp_path / "takes.json"
    tf.write_text(json.dumps(takes))
    assert cloud_cli.main(["generate", "--takes", str(tf), "--wait", "--out", str(tmp_path / "o")]) == 0
    monkeypatch.setenv("GLORE_WEB_KEY", "correct horse battery staple")
    out = tmp_path / "state" / "state.json"
    assert cloud_cli.main(["publish-state", "--out", str(out)]) == 0
    enc = json.loads(out.read_text())
    assert "warm house" not in out.read_text() and "t1" not in out.read_text()
    d = lambda k: base64.b64decode(enc[k])
    key = hashlib.pbkdf2_hmac("sha256", b"correct horse battery staple", d("salt"), enc["iter"], dklen=32)
    state = json.loads(AESGCM(key).decrypt(d("iv"), d("ct"), None))
    gen = next(j for j in state["jobs"] if j["mode"] == "generate")
    assert gen["phase"] == "done" and gen["takes"][0]["url"].startswith("https://s3.example/outputs/")
    assert gen["takes"][0]["url"].endswith(str(7 * 86400))
    assert sorted(state["loras"]["runX"]["points"]) == ["e50", "final"]


def test_publish_state_needs_the_web_key(cli, monkeypatch, capsys):
    monkeypatch.delenv("GLORE_WEB_KEY", raising=False)
    assert cloud_cli.main(["publish-state"]) == 1
    assert "GLORE_WEB_KEY" in capsys.readouterr().err


SONG = r'''
import json, sys, os, shutil
what, path = sys.argv[1], sys.argv[2]
job = json.load(open(path))
os.makedirs(job["work"], exist_ok=True)
if what == "prep":
    v = os.path.join(job["work"], "vocal.wav"); shutil.copy(job["input"], v)
    print("GLORE_RESULT " + json.dumps({"vocal": v, "bpm": 128.0, "key": "F minor", "duration": 12.5, "kind": job["kind"]}))
else:
    for k in ("song_mp3", "beat_mp3"):
        open(job[k], "wb").write(b"mp3:" + open(job["take"], "rb").read()[:4])
    print("GLORE_RESULT " + json.dumps({"song_mp3": job["song_mp3"], "beat_mp3": job["beat_mp3"]}))
'''


class SongAce(GenAce):
    def ensure(self, variant, log, status, extras=()):
        self.extras = extras
        status.update(phase="setup", message="pretend install")

    def py(self, *args):
        if args and str(args[0]).endswith("glore_song.py"):
            return [sys.executable, "-c", SONG, *args[1:]]
        return super().py(*args)


def encrypt_upload(data: bytes, passphrase: str) -> bytes:
    import hashlib, os as _os
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = _os.urandom(16), _os.urandom(12)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000, dklen=32)
    return salt + iv + AESGCM(key).encrypt(iv, data, None)


def test_song_from_an_encrypted_upload(cli, tmp_path, monkeypatch):
    ace = SongAce(tmp_path)
    cli["rp"].ace = ace
    monkeypatch.setenv("GLORE_WEB_KEY", "pw")
    blob = encrypt_upload(b"RIFF-fake-vocal", "pw")
    taken = []
    monkeypatch.setattr(cloud_cli, "take_upload", lambda p: (taken.append(p), blob)[1])
    vol = cli["vol"]
    vol.put_bytes("loras/house/runX/final/adapter_model.safetensors", b"w")
    tf = tmp_path / "takes.json"
    tf.write_text(json.dumps([{"name": "my song 1", "caption": "deep house", "lora": "runX", "seed": 5},
                              {"name": "my song 2", "caption": "deep house", "seed": 6}]))
    out = tmp_path / "o"
    rc = cloud_cli.main(["song", "--upload", "uploads/abc.bin", "--name", "../My Vocal.mp3", "--kind", "vocal",
                         "--takes", str(tf), "--wait", "--out", str(out), "--timeout-min", "1"])
    assert rc == 0 and taken == ["uploads/abc.bin"]
    assert ace.extras == glore_pod.SONG_EXTRAS
    res = json.loads((out / "result.json").read_text())["result"]
    assert res["bpm"] == 128.0 and res["key"] == "F minor" and len(res["takes"]) == 2
    assert sorted(p.name for p in out.glob("*.mp3")) == ["my song 1 (beat).mp3", "my song 1.mp3",
                                                         "my song 2 (beat).mp3", "my song 2.mp3"]
    # the decrypted vocal sits on the volume, with a safe name
    ups = vol.list_keys("uploads/")
    assert len(ups) == 1 and ups[0].endswith("/My Vocal.mp3")
    assert vol.get_bytes(ups[0]) == b"RIFF-fake-vocal"
    # generation used ACE 'complete' around the vocal, with ACE's own instruction format available
    seen = [json.loads(p.read_text()) for p in (tmp_path / "ace").rglob("seen.json")]
    assert not seen or all(s["mode"] == "complete" for s in seen)


def test_decrypt_upload_rejects_the_wrong_key():
    from cryptography.exceptions import InvalidTag
    blob = encrypt_upload(b"secret", "right")
    assert cloud_cli.decrypt_upload(blob, "right") == b"secret"
    with pytest.raises(InvalidTag):
        cloud_cli.decrypt_upload(blob, "wrong")
