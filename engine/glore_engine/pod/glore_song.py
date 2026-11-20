"""GLORE song tools - run inside ACE-Step's venv on the cloud GPU (needs demucs + librosa there).

    python glore_song.py prep job.json   -> vocal.wav + analysis.json (bpm, key, duration)
    python glore_song.py mix  job.json   -> strips the model's own vocal from a take, lays the real
                                            vocal on top, writes song + beat MP3s

prep job: {"input": path, "kind": "vocal"|"song", "work": dir}
mix  job: {"take": wav, "vocal": wav, "work": dir, "song_mp3": path, "beat_mp3": path,
           "vocal_gain_db": -2, "beat_gain_db": 0}
Prints GLORE_RESULT <json> / GLORE_ERROR <json>.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback

SR = 44100
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler key profiles
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def to_wav(src: str, dst: str) -> str:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "2", "-ar", str(SR), dst], check=True)
    return dst


def separate(path: str, out_dir: str) -> dict:
    """htdemucs two-stem split -> {"vocals": wav, "no_vocals": wav}."""
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", out_dir, path],
                   check=True)
    base = os.path.join(out_dir, "htdemucs", os.path.splitext(os.path.basename(path))[0])
    return {"vocals": os.path.join(base, "vocals.wav"), "no_vocals": os.path.join(base, "no_vocals.wav")}


def estimate_key(y, sr) -> str:
    import numpy as np
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best, name = -2.0, "C major"
    for i in range(12):
        for prof, mode in ((MAJOR, "major"), (MINOR, "minor")):
            r = float(np.corrcoef(chroma, np.roll(prof, i))[0, 1])
            if r > best:
                best, name = r, f"{KEYS[i]} {mode}"
    return name


def analyse(path: str, music_path: str | None = None) -> dict:
    """BPM from the music if we have it (drums keep time better than a voice), key from the whole thing."""
    import librosa
    y, sr = librosa.load(music_path or path, sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
    while bpm < 90:   # house-friendly octave
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    yk, _ = librosa.load(path, sr=22050, mono=True)
    return {"bpm": round(bpm, 1), "key": estimate_key(yk, sr), "duration": round(len(yk) / sr, 2)}


def prep(job: dict) -> dict:
    work = job["work"]
    os.makedirs(work, exist_ok=True)
    src = to_wav(job["input"], os.path.join(work, "input.wav"))
    music = None
    if job.get("kind") == "song":
        parts = separate(src, os.path.join(work, "sep"))
        vocal, music = parts["vocals"], parts["no_vocals"]
    else:
        vocal = src
    a = analyse(vocal, music)
    out = {"vocal": vocal, "music": music, **a}
    with open(os.path.join(work, "analysis.json"), "w") as f:
        json.dump(out, f)
    return out


def mix(job: dict) -> dict:
    import numpy as np
    import soundfile as sf
    work = job["work"]
    os.makedirs(work, exist_ok=True)
    take = to_wav(job["take"], os.path.join(work, "take.wav"))
    beat_path = separate(take, os.path.join(work, "sep"))["no_vocals"]  # the model sings its own copy
    beat, _ = sf.read(beat_path, dtype="float32", always_2d=True)
    voc, _ = sf.read(to_wav(job["vocal"], os.path.join(work, "vocal.wav")), dtype="float32", always_2d=True)
    n = max(len(beat), len(voc))
    pad = lambda x: np.pad(x, ((0, n - len(x)), (0, 0)))
    g = lambda db: 10 ** (float(db) / 20)
    song = pad(voc) * g(job.get("vocal_gain_db", -2)) + pad(beat) * g(job.get("beat_gain_db", 0))
    song *= 0.89 / max(1e-6, float(np.max(np.abs(song))))
    beat_n = beat * (0.89 / max(1e-6, float(np.max(np.abs(beat)))))
    for data, mp3 in ((song, job["song_mp3"]), (beat_n, job["beat_mp3"])):
        wav = os.path.join(work, "out.wav")
        sf.write(wav, data, SR)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-c:a", "libmp3lame", "-b:a", "256k", mp3], check=True)
    return {"song_mp3": job["song_mp3"], "beat_mp3": job["beat_mp3"], "seconds": round(n / SR, 1)}


if __name__ == "__main__":
    try:
        what, path = sys.argv[1], sys.argv[2]
        job = json.load(open(path, encoding="utf-8"))
        res = {"prep": prep, "mix": mix}[what](job)
        print("GLORE_RESULT " + json.dumps(res), flush=True)
    except Exception as e:
        traceback.print_exc()
        print("GLORE_ERROR " + json.dumps({"error": str(e)}), flush=True)
        sys.exit(1)
