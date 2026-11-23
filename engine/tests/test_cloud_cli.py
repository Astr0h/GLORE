"""The PC-free cloud CLI (GitHub Actions): train from the volume, generate on the pod, ledger on the volume."""
import json
import time
from pathlib import Path
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
    monkeypatch.setenv("GLORE_POLL_S", "0.2")
    monkeypatch.setenv("GLORE_GPU_WAIT_S", "0")
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


def test_state_is_encrypted_and_audio_goes_to_the_media_branch(cli, tmp_path, monkeypatch):
    import base64
    import hashlib
    import subprocess
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    vol = cli["vol"]
    vol.put_bytes("loras/house/runX/final/adapter_model.safetensors", b"w")
    vol.put_bytes("loras/house/runX/checkpoints/epoch_50_loss_0.9/adapter_model.safetensors", b"w")
    takes = [{"name": "t1", "caption": "warm house", "lora": "runX@e50", "seed": 1, "duration": 30}]
    tf = tmp_path / "takes.json"
    tf.write_text(json.dumps(takes))
    assert cloud_cli.main(["generate", "--takes", str(tf), "--wait", "--out", str(tmp_path / "o")]) == 0
    src = tmp_path / "v.wav"
    src.write_bytes(tiny_wav())
    item = cloud_cli.library_add(vol, src, "My vocal", "vocal", "upload")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    monkeypatch.setenv("GLORE_WEB_KEY", "correct horse battery staple")
    out = tmp_path / "state" / "state.json"
    assert cloud_cli.main(["publish-state", "--out", str(out), "--media-remote", str(remote)]) == 0
    enc = json.loads(out.read_text())
    assert "warm house" not in out.read_text() and "\"t1\"" not in out.read_text()
    d = lambda k: base64.b64decode(enc[k])
    key = hashlib.pbkdf2_hmac("sha256", b"correct horse battery staple", d("salt"), enc["iter"], dklen=32)
    state = json.loads(AESGCM(key).decrypt(d("iv"), d("ct"), None))
    gen = next(j for j in state["jobs"] if j["mode"] == "generate")
    assert gen["phase"] == "done" and "_media" not in state and state["media_files"] == 2
    assert sorted(state["loras"]["runX"]["points"]) == ["e50", "final"]
    # the take and a preview of the library item are on the media branch, encrypted
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "--name-only", "glore-media"],
                           capture_output=True, text=True, check=True).stdout.split()
    lib = state["library"][0]
    assert sorted(files) == sorted([gen["takes"][0]["media"], lib["media"]])
    blob = subprocess.run(["git", "--git-dir", str(remote), "show", f"glore-media:{gen['takes'][0]['media']}"],
                          capture_output=True, check=True).stdout
    assert cloud_cli.decrypt_upload(blob, "correct horse battery staple") == vol.get_bytes(gen["takes"][0]["mp3"])
    prev = subprocess.run(["git", "--git-dir", str(remote), "show", f"glore-media:{lib['media']}"],
                          capture_output=True, check=True).stdout
    assert cloud_cli.decrypt_upload(prev, "correct horse battery staple")[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
    # a second publish re-uses what's there (no new commit); deleting the item drops its preview
    head = lambda: subprocess.run(["git", "--git-dir", str(remote), "rev-parse", "glore-media"], capture_output=True,
                                  text=True).stdout
    h1 = head()
    assert cloud_cli.main(["publish-state", "--out", str(out), "--media-remote", str(remote)]) == 0
    assert head() == h1
    assert cloud_cli.main(["library", "--delete", item["id"]]) == 0
    assert cloud_cli.main(["publish-state", "--out", str(out), "--media-remote", str(remote)]) == 0
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "--name-only", "glore-media"],
                           capture_output=True, text=True, check=True).stdout.split()
    assert files == [gen["takes"][0]["media"]]


def test_publish_state_needs_the_web_key(cli, monkeypatch, capsys):
    monkeypatch.delenv("GLORE_WEB_KEY", raising=False)
    assert cloud_cli.main(["publish-state"]) == 1
    assert "GLORE_WEB_KEY" in capsys.readouterr().err


class FakeS3:
    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://s3.example/{Params['Key']}?X-Expires={ExpiresIn}"


SONG = r"""
import json, sys, os, shutil
what, path = sys.argv[1], sys.argv[2]
job = json.load(open(path))
os.makedirs(job["work"], exist_ok=True)
if what == "prep":
    v = os.path.join(job["work"], "vocal.wav"); shutil.copy(job["input"], v)
    print("GLORE_RESULT " + json.dumps({"vocal": v, "bpm": 128.0, "key": "F minor", "duration": 12.5, "kind": job["kind"]}))
"""

# stands in for glore_tools.py: records what it was asked, writes small outputs, adds library items
TOOLS = r"""
import json, sys, os, time, uuid
op, path = sys.argv[1], sys.argv[2]
job = json.load(open(path)); root = job["root"]; p = job["params"]; item = job["item"]
open(os.path.join(root, f"seen-{op}.json"), "w").write(json.dumps(job))
def add(name, kind):
    i = uuid.uuid4().hex[:10]; d = os.path.join(root, "library", i); os.makedirs(d)
    open(os.path.join(d, "a.flac"), "wb").write(b"flac")
    it = {"id": i, "name": name, "kind": kind, "file": f"library/{i}/a.flac", "created": time.time(), "parent": item.get("id")}
    open(os.path.join(d, "item.json"), "w").write(json.dumps(it)); return it
if op == "pmap":
    d = os.path.join(root, "glore", "produce", p["sid"]); os.makedirs(d, exist_ok=True)
    json.dump({"bpm": 124, "key": "A minor", "offset": 0}, open(os.path.join(d, "meta.json"), "w"))
    m = {"bpm": 124.0, "key": "A minor", "bars": 16, "kind": "vocal", "delivery": {"sung": 0.9, "style": "sung"},
         "sections": [{"name": "verse 1", "kind": "verse", "start": 0, "bars": 8}, {"name": "chorus 1", "kind": "chorus", "start": 8, "bars": 8}],
         "hook": {"bar": 8, "bars": 1}, "phrases": [], "held_notes": [], "sung_notes": [9]}
    print("GLORE_RESULT " + json.dumps({"songmap": m, "sid": p["sid"], "source_bars": {"vocal": 16, "chopriff": 4,
          "bed.drums": 400, "bed.bass": 400, "bed.other": 400, "bed.vox": 400}}))
elif op == "pguide":
    g = os.path.join(root, "glore", "produce", p["sid"], "guide.wav"); open(g, "wb").write(b"guide")
    print("GLORE_RESULT " + json.dumps({"guide": g, "duration": 60.0, "warnings": []}))
elif op == "prender":
    assert os.path.exists(p["bed"])
    b = p["out_base"]
    for f in (b + ".mp3", b + " (beat).mp3", b + " (stems).zip"):
        open(f, "wb").write(b"x")
    n = len([f for f in os.listdir(root) if f.startswith("seen-prender-")])
    open(os.path.join(root, f"seen-prender-{n}.json"), "w").write(json.dumps(job))
    print("GLORE_RESULT " + json.dumps({"mp3": b + ".mp3", "beat_mp3": b + " (beat).mp3", "zip": b + " (stems).zip",
          "mastered_to": "house pack", "duration": 60.0,
          "metrics": {"length_seconds": 60.0, "sections": [], "tracks_gain": [t.get("gain_db", 0) for t in p["plan"]["tracks"]]}}))
elif op == "arrange":
    print("GLORE_RESULT " + json.dumps({"vocal": p["vocal"], "music": p.get("music"), "duration": 28.5,
          "hint": "arranged like a finished record, with big instrumental drops after the chorus",
          "plan": {"sections": [{"part": "intro", "start": 0, "end": 16}, {"part": "vocal", "start": 16, "end": 28.5}]}}))
elif op == "mixsong":
    b = p["out_base"]
    for f in (b + ".mp3", b + " (beat).mp3", b + " (stems).zip"):
        open(f, "wb").write(b"x")
    print("GLORE_RESULT " + json.dumps({"mp3": b + ".mp3", "beat_mp3": b + " (beat).mp3", "zip": b + " (stems).zip"}))
elif op == "stems":
    print("GLORE_RESULT " + json.dumps({"items": [add(item["name"] + f" ({s})", "stem") for s in ("vocals", "drums", "bass", "other")]}))
elif op == "analyse":
    item.update(bpm=124.0, key="G minor"); open(os.path.join(root, "library", item["id"], "item.json"), "w").write(json.dumps(item))
    print("GLORE_RESULT " + json.dumps({"updated": item}))
else:
    print("GLORE_RESULT " + json.dumps({"items": [add(item["name"] + " (" + op + ")", "vocal")]}))
"""


class SongAce(GenAce):
    def ensure(self, variant, log, status, extras=(), models=True):
        self.extras = extras
        status.update(phase="setup", message="pretend install")

    def py(self, *args):
        if args and str(args[0]).endswith("glore_song.py"):
            return [sys.executable, "-c", SONG, *args[1:]]
        if args and str(args[0]).endswith("glore_tools.py"):
            return [sys.executable, "-c", TOOLS, *args[1:]]
        return super().py(*args)


def encrypt_upload(data: bytes, passphrase: str) -> bytes:
    import hashlib, os as _os
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = _os.urandom(16), _os.urandom(12)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000, dklen=32)
    return salt + iv + AESGCM(key).encrypt(iv, data, None)


def tiny_wav(seconds=0.5) -> bytes:
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros((int(44100 * seconds), 2), dtype="float32"), 44100, format="WAV")
    return buf.getvalue()


def test_song_from_an_encrypted_upload(cli, tmp_path, monkeypatch):
    ace = SongAce(tmp_path)
    cli["rp"].ace = ace
    monkeypatch.setenv("GLORE_WEB_KEY", "pw")
    blob = encrypt_upload(tiny_wav(), "pw")
    taken = []
    monkeypatch.setattr(cloud_cli, "take_upload", lambda p: (taken.append(p), blob)[1])
    vol = cli["vol"]
    vol.put_bytes("loras/house/runX/final/adapter_model.safetensors", b"w")
    tf = tmp_path / "takes.json"
    tf.write_text(json.dumps([{"name": "my song 1", "caption": "deep house", "lora": "runX", "seed": 5},
                              {"name": "my song 2", "caption": "deep house", "seed": 6}]))
    out = tmp_path / "o"
    rc = cloud_cli.main(["song", "--upload", "uploads/abc.bin", "--name", "../My Vocal.wav", "--kind", "vocal",
                         "--fx", "club", "--chops", "house", "--takes", str(tf), "--wait", "--out", str(out),
                         "--timeout-min", "1"])
    assert rc == 0 and taken == ["uploads/abc.bin"]
    assert ace.extras == glore_pod.SONG_EXTRAS
    res = json.loads((out / "result.json").read_text())["result"]
    assert res["bpm"] == 128.0 and res["key"] == "F minor" and len(res["takes"]) == 2
    assert sorted(p.name for p in out.iterdir() if p.suffix in (".mp3", ".zip")) == [
        "my song 1 (beat).mp3", "my song 1 (stems).zip", "my song 1.mp3",
        "my song 2 (beat).mp3", "my song 2 (stems).zip", "my song 2.mp3"]
    # the upload went into the library as FLAC, with a safe name
    items = cloud_cli.library_items(vol)
    assert len(items) == 1 and items[0]["name"] == "My Vocal" and items[0]["file"].endswith("/My Vocal.flac")
    assert items[0]["origin"] == "upload" and abs(items[0]["duration"] - 0.5) < 0.05
    # the mix got the options from the web app, the vocal, BPM and key
    mixed = json.loads((tmp_path / "volume" / "seen-mixsong.json").read_text())["params"]
    assert mixed["fx"] == "club" and mixed["chops"] == "house" and mixed["bpm"] == 128.0 and mixed["key"] == "F minor"
    assert mixed["master"] == "pack" and mixed["ai_vox_db"] == -9.0 and mixed["pack"] == "house"
    # the arrangement was planned, its sections reach the app, and the beat was asked for them
    arranged = json.loads((tmp_path / "volume" / "seen-arrange.json").read_text())["params"]
    assert arranged["style"] == "house" and arranged["bpm"] == 128.0
    assert res["sections"][0]["part"] == "intro" and res["duration"] == 28.5
    # a second song can reuse the library item without uploading again
    rc = cloud_cli.main(["song", "--item", items[0]["id"], "--takes", str(tf), "--wait", "--out", str(tmp_path / "o2"),
                         "--timeout-min", "1"])
    assert rc == 0 and taken == ["uploads/abc.bin"]


def test_tools_add_results_to_the_library(cli, tmp_path, monkeypatch):
    ace = SongAce(tmp_path)
    cli["rp"].ace = ace
    vol = cli["vol"]
    src = tmp_path / "v.wav"
    src.write_bytes(tiny_wav())
    song = cloud_cli.library_add(vol, src, "Uzi song", "song", "upload")
    assert cloud_cli.main(["tool", "--op", "stems", "--item", song["id"], "--wait", "--timeout-min", "1"]) == 0
    names = sorted(i["name"] for i in cloud_cli.library_items(vol))
    assert names == ["Uzi song", "Uzi song (bass)", "Uzi song (drums)", "Uzi song (other)", "Uzi song (vocals)"]
    assert cloud_cli.main(["tool", "--op", "analyse", "--item", song["id"], "--wait", "--timeout-min", "1"]) == 0
    assert cloud_cli.library_get(vol, song["id"])["bpm"] == 124.0
    assert cloud_cli.main(["tool", "--op", "chops", "--item", song["id"], "--set", "style=rnb", "--set", "seed=3",
                           "--set", "key=A_minor", "--wait", "--timeout-min", "1"]) == 0
    seen = json.loads((tmp_path / "volume" / "seen-chops.json").read_text())
    assert seen["params"] == {"style": "rnb", "seed": 3.0, "key": "A minor"}
    assert cloud_cli.main(["tool", "--op", "clean", "--item", song["id"], "--set", "stages=echo,reverb",
                           "--wait", "--timeout-min", "1"]) == 0
    assert json.loads((tmp_path / "volume" / "seen-clean.json").read_text())["params"]["stages"] == ["echo", "reverb"]
    assert ace.extras == glore_pod.SONG_EXTRAS + glore_pod.CLEAN_EXTRAS
    assert cloud_cli.main(["tool", "--op", "explode", "--item", song["id"]]) == 1
    # state for the web app lists the library with play links, and the tool jobs
    vol.s3, vol.bucket = FakeS3(), "vol123"
    st = cloud_cli.build_state(cli["mgr"])
    assert len(st["library"]) == 7 and all(i["media"] for i in st["library"])
    assert {j.get("op") for j in st["jobs"]} >= {"stems", "analyse", "chops", "clean"}
    # delete
    assert cloud_cli.main(["library", "--delete", song["id"]]) == 0
    assert song["id"] not in [i["id"] for i in cloud_cli.library_items(vol)]


def test_youtube_links_wait_for_the_pc(cli, tmp_path, monkeypatch):
    vol = cli["vol"]
    assert cloud_cli.main(["link", "--url", "https://youtu.be/abc123"]) == 0
    pend = cloud_cli.pending_links(vol)
    assert len(pend) == 1 and pend[0]["service"] == "youtube"
    vol.s3, vol.bucket = FakeS3(), "vol123"
    assert cloud_cli.build_state(cli["mgr"])["pending"][0]["url"] == "https://youtu.be/abc123"
    # the PC (home internet) picks it up
    from glore_engine import linkimport

    class Info:
        service, title, artist, matched = "youtube", "Song", "Artist", "Artist - Song"

    def fake_fetch(url, progress=None):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        w = Path(tmp.name) / "a.wav"
        w.write_bytes(tiny_wav())
        return w, Info(), tmp
    monkeypatch.setattr(linkimport, "fetch_link", fake_fetch)
    done = cloud_cli.process_pending_links(vol, report=lambda m: None)
    assert [d["name"] for d in done] == ["Artist - Song"] and not cloud_cli.pending_links(vol)
    assert cloud_cli.library_items(vol)[0]["url"] == "https://youtu.be/abc123"
    # a failure is recorded for the web app, and kept for another try
    assert cloud_cli.main(["link", "--url", "https://open.spotify.com/track/xyz"]) == 0

    def broken(url, progress=None):
        raise linkimport.LinkError("That track isn't available")
    monkeypatch.setattr(linkimport, "fetch_link", broken)
    cloud_cli.process_pending_links(vol, report=lambda m: None)
    assert cloud_cli.pending_links(vol)[0]["error"] == "That track isn't available"
    # cancel it
    rid = cloud_cli.pending_links(vol)[0]["id"]
    assert cloud_cli.main(["link", "--cancel", rid]) == 0 and not cloud_cli.pending_links(vol)
    # a SoundCloud link that fails in the cloud stays visible, and the PC never retries it
    assert cloud_cli.main(["link", "--url", "https://soundcloud.com/a/b"]) == 1
    p = cloud_cli.pending_links(vol)
    assert len(p) == 1 and p[0]["failed"] and "isn't available" in p[0]["error"]
    tried = []
    monkeypatch.setattr(linkimport, "fetch_link", lambda url, progress=None: tried.append(url))
    cloud_cli.process_pending_links(vol, report=lambda m: None)
    assert tried == []


def test_decrypt_upload_rejects_the_wrong_key():
    from cryptography.exceptions import InvalidTag
    blob = encrypt_upload(b"secret", "right")
    assert cloud_cli.decrypt_upload(blob, "right") == b"secret"
    with pytest.raises(InvalidTag):
        cloud_cli.decrypt_upload(blob, "wrong")


def test_no_gpu_shows_in_the_app_and_leaves_nothing_queued(cli, tmp_path, monkeypatch):
    vol = cli["vol"]
    src = tmp_path / "v.wav"
    src.write_bytes(tiny_wav())
    item = cloud_cli.library_add(vol, src, "Fred again.. - Winny.mp3", "song", "upload")
    assert item["name"] == "Fred again.. - Winny"   # dots in a title are not a file extension

    def sold_out(*a, **k):
        raise CloudError("No RTX 4090 free in EUR-IS-1 right now. Try again in a few minutes.")
    monkeypatch.setattr(cloud, "gpu_choices", sold_out)
    monkeypatch.setattr(cloud_cli, "_idle_pod", lambda mgr, vol: None)
    tf = tmp_path / "takes.json"
    tf.write_text(json.dumps([{"name": "a", "caption": "house", "seed": 1}]))
    assert cloud_cli.main(["song", "--item", item["id"], "--takes", str(tf)]) == 1
    assert not [k for k in vol.list_keys("glore/queue/") if k.endswith(".json")]
    vol.s3, vol.bucket = FakeS3(), "vol123"
    st = cloud_cli.build_state(cli["mgr"])
    err = next(j for j in st["jobs"] if j["mode"] == "song")
    assert err["phase"] == "error" and "free in" in err["error"]


def test_produce_plans_builds_measures_and_revises(cli, tmp_path, monkeypatch):
    from glore_engine import producer
    ace = SongAce(tmp_path)
    cli["rp"].ace = ace
    vol = cli["vol"]
    src = tmp_path / "v.wav"
    src.write_bytes(tiny_wav())
    item = cloud_cli.library_add(vol, src, "My vocal", "vocal", "upload")
    vol.put_bytes("loras/house/runX/final/adapter_model.safetensors", b"w")
    plan = {"title": "Night drive", "notes": "Kept the hook, built two drops.", "bpm": 124, "bars": 32,
            "bed": {"caption": "dark garage, shuffled drums"},
            "sections": [{"name": "intro", "start": 0, "bars": 8, "energy": 0.3}, {"name": "drop", "start": 8, "bars": 24, "energy": 1}],
            "tracks": [{"id": "lead", "group": "vocal", "guide": True, "clips": [{"at": 8, "src": "vocal", "from": 0, "bars": 16}]},
                       {"id": "drums", "role": "drums", "clips": [{"at": 8, "src": "bed.drums", "from": 8, "bars": 24}]}]}
    asked = []

    def fake_ask(system, user, model, key, **k):
        asked.append(user)
        if len(asked) == 1:
            return json.loads(json.dumps(plan)), {"input_tokens": 1, "output_tokens": 2}
        louder = json.loads(json.dumps(plan))
        louder["tracks"][0]["gain_db"] = 2          # the second look: lead up 2 dB, vocal left where it is
        return louder, {}
    monkeypatch.setattr(producer, "ask", fake_ask)
    monkeypatch.setenv("GLORE_ANTHROPIC_KEY", "k")
    tf = tmp_path / "spec.json"
    tf.write_text(json.dumps({"brief": "dark garage, keep the hook", "style": "house", "model": "standard",
                              "takes": [{"name": "Night 1", "lora": "runX", "seed": 3}, {"name": "Night 2", "seed": 4}]}))
    out = tmp_path / "o"
    assert cloud_cli.main(["produce", "--item", item["id"], "--takes", str(tf), "--out", str(out), "--timeout-min", "1"]) == 0
    assert "dark garage, keep the hook" in asked[0] and "Measurements" in asked[1] and len(asked) == 2
    # first pass: take 1 only; after the revision: both takes, with the revised plan
    seen = [json.loads(p.read_text())["params"] for p in sorted((tmp_path / "volume").glob("seen-prender-*.json"))]
    assert [x["plan"]["tracks"][0]["gain_db"] for x in seen] == [0, 2, 2]
    assert seen[0]["bed"] == seen[1]["bed"] != seen[2]["bed"]     # same vocal layout + seed: the instrumental is reused
    assert sorted(p.name for p in out.iterdir() if p.suffix == ".mp3") == ["Night 1 (beat).mp3", "Night 1.mp3",
                                                                            "Night 2 (beat).mp3", "Night 2.mp3"]
    # the beat was built around the guide with the producer's caption and the LoRA
    gen = [json.loads(p.read_text()) for p in (tmp_path / "volume").rglob("seen.json")]
    vol.s3, vol.bucket = FakeS3(), "vol123"
    st = cloud_cli.build_state(cli["mgr"])
    rows = [j for j in st["jobs"] if j["mode"] in ("produce", "produce_map")]
    assert len(rows) == 1                                         # the first pass and the map job are folded away
    r = rows[0]
    assert r["title"] == "Night drive" and r["notes"].startswith("Kept") and r["producer"]["by"] == "claude"
    assert [t["name"] for t in r["takes"]] == ["Night 1", "Night 2"] and "metrics" not in r["takes"][0]
    assert r["sections"][0]["name"] == "intro" and r["brief"] == "dark garage, keep the hook"


def test_produce_without_a_key_uses_the_rules(cli, tmp_path, monkeypatch):
    cli["rp"].ace = SongAce(tmp_path)
    vol = cli["vol"]
    src = tmp_path / "v.wav"
    src.write_bytes(tiny_wav())
    item = cloud_cli.library_add(vol, src, "My vocal", "vocal", "upload")
    monkeypatch.delenv("GLORE_ANTHROPIC_KEY", raising=False)
    tf = tmp_path / "spec.json"
    tf.write_text(json.dumps({"brief": "warm", "takes": [{"name": "A", "seed": 1}]}))
    assert cloud_cli.main(["produce", "--item", item["id"], "--takes", str(tf), "--timeout-min", "1"]) == 0
    seen = [json.loads(p.read_text())["params"] for p in (tmp_path / "volume").glob("seen-prender-*.json")]
    assert len(seen) == 1 and seen[0]["plan"]["sections"][0]["name"] == "intro"
    assert any(t["guide"] for t in seen[0]["plan"]["tracks"])


def test_clean_frees_scratch_but_keeps_your_work(cli):
    vol = cli["vol"]
    keep = ["library/abc/item.json", "library/abc/a.flac", "outputs/j1/a.mp3", "loras/house/r/final/adapter_model.safetensors",
            "glore/runs/old/status.json", "glore/produce/s1/plan1.json", "glore/runs/live/song/take.wav"]
    junk = ["glore/runs/old/song/take.wav", "glore/produce/s1/bed_x.wav", "glore/produce/s1/conf_1/vocal.wav", ".cache/uv/a/b.whl"]
    for k in keep + junk:
        vol.put_bytes(k, b"x" * 100)
    vol.put_bytes("glore/runs/old/status.json", json.dumps({"phase": "done"}).encode())
    vol.put_bytes("glore/runs/live/status.json", json.dumps({"phase": "generating", "updated": time.time()}).encode())
    assert cloud_cli.main(["clean"]) == 0
    left = set(vol.list_keys(""))
    assert set(keep) <= left and ".cache/uv/a/b.whl" in left and not (set(junk[:3]) & left)
    assert cloud_cli.main(["clean", "--cache"]) == 0
    assert ".cache/uv/a/b.whl" not in set(vol.list_keys(""))
