"""Mix (SPEC step 8): kick sidechain, a pocket carved in the beat for the vocal, master limiter.

    song, parts = mix_song({"drums": d, "bass": b, "other": o}, vocal, sr, bpm, chops=c)

Every input is (2, n) float32 at sr. Missing stems are fine: pass {"beat": full_beat} if the
beat wasn't split. Returns the master and the processed parts (for the stems export).
"""

from __future__ import annotations

import numpy as np
import pedalboard as pb


def _env(x: np.ndarray, sr: int, attack: float, release: float, hop: int = 256) -> np.ndarray:
    """Follower on a mono signal, per-sample output."""
    frames = len(x) // hop + 1
    rms = np.sqrt(np.pad(x ** 2, (0, frames * hop - len(x))).reshape(frames, hop).mean(1))
    a, r = np.exp(-1 / (attack * sr / hop)), np.exp(-1 / (release * sr / hop))
    out = np.zeros_like(rms)
    for i, v in enumerate(rms):
        prev = out[i - 1] if i else 0.0
        c = a if v > prev else r
        out[i] = c * prev + (1 - c) * v
    return np.interp(np.arange(len(x)), np.arange(frames) * hop, out).astype(np.float32)


def kick_envelope(drums: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    """0..1, peaks on each kick (from the sub-150 Hz part of the drums)."""
    low = pb.Pedalboard([pb.LowpassFilter(150), pb.LowpassFilter(150)])(drums, sr).mean(0)
    env = _env(low, sr, attack=0.002, release=min(0.25, 60 / bpm / 3))
    return np.clip(env / (np.percentile(env, 98) + 1e-9), 0, 1)


def sidechain(x: np.ndarray, kick: np.ndarray, depth_db: float) -> np.ndarray:
    g = 10 ** (-depth_db * kick / 20)
    return (x * g[: x.shape[1]]).astype(np.float32)


def vocal_pocket(x: np.ndarray, vocal: np.ndarray, sr: int, depth_db: float = 3.5) -> np.ndarray:
    """Dip 1-4 kHz of the music only while the vocal is singing (a dynamic EQ)."""
    band = pb.Pedalboard([pb.HighpassFilter(1000), pb.LowpassFilter(4000)])(x, sr)
    v = _env(vocal.mean(0), sr, attack=0.01, release=0.2)
    act = np.clip(v / (np.percentile(v, 90) + 1e-9), 0, 1)
    cut = 1 - 10 ** (-depth_db / 20)
    return (x - band * (cut * act[: x.shape[1]])).astype(np.float32)


def master(x: np.ndarray, sr: int, target_rms_db: float = -11.0) -> np.ndarray:
    x = pb.Pedalboard([pb.HighpassFilter(25), pb.Compressor(threshold_db=-14, ratio=2, attack_ms=20,
                                                            release_ms=200)])(x, sr)
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    x = x * (10 ** (target_rms_db / 20) / rms)
    x = pb.Pedalboard([pb.Limiter(threshold_db=-1.0, release_ms=80)])(x, sr)
    return np.clip(x, -0.98, 0.98).astype(np.float32)


def _fit(parts: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {k: (np.pad(v, ((0, 0), (0, max(0, n - v.shape[1]))))[:, :n]).astype(np.float32)
            for k, v in parts.items() if v is not None}


def mix_song(music: dict[str, np.ndarray], vocal: np.ndarray, sr: int, bpm: float,
             chops: np.ndarray | None = None, vocal_db: float = 0.0, beat_db: float = 0.0, chops_db: float = 0.0,
             sidechain_db: float = 5.0, pocket_db: float = 3.5) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = max([v.shape[1] for v in music.values()] + [vocal.shape[1]])
    music = _fit(music, n)
    vocal = _fit({"v": vocal}, n)["v"]
    g = lambda db: 10 ** (db / 20)
    drums = music.get("drums")
    kick = kick_envelope(drums if drums is not None else music.get("beat"), sr, bpm) if sidechain_db else None
    out = {}
    for name, x in music.items():
        x = x * g(beat_db)
        if name in ("bass", "other", "beat") and kick is not None:
            x = sidechain(x, kick, sidechain_db * (0.6 if name == "beat" else 1.0))
        if name in ("other", "beat") and pocket_db:
            x = vocal_pocket(x, vocal, sr, pocket_db)
        out[name] = x
    if chops is not None:
        c = _fit({"c": chops}, n)["c"] * g(chops_db)
        out["chops"] = sidechain(c, kick, sidechain_db * 0.5) if kick is not None else c
    out["vocal"] = vocal * g(vocal_db)
    song = master(sum(out.values()), sr)
    return song, out
