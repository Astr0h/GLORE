"""Vocal FX (SPEC step 7): EQ, compression, de-ess, reverb, delay. Presets Clean / Airy / Club,
with reverb and delay amounts on top (0..1, 0.5 = the preset as designed).

    wet = vocal_fx(vocal, sr, preset="airy", bpm=128, reverb=0.5, delay=0.5)
"""

from __future__ import annotations

import numpy as np
import pedalboard as pb

PRESETS = ("clean", "airy", "club", "none")


def deess(x: np.ndarray, sr: int, freq: float = 6500.0, threshold_db: float = -30.0) -> np.ndarray:
    """Split off the 's' band, squash only that, put it back."""
    high = pb.Pedalboard([pb.HighpassFilter(freq), pb.HighpassFilter(freq)])(x, sr)
    low = x - high
    tamed = pb.Pedalboard([pb.Compressor(threshold_db=threshold_db, ratio=6, attack_ms=0.5, release_ms=40)])(high, sr)
    return (low + tamed).astype(np.float32)


def vocal_fx(x: np.ndarray, sr: int, preset: str = "clean", bpm: float = 120.0,
             reverb: float = 0.5, delay: float = 0.5) -> np.ndarray:
    if preset == "none":
        return x.astype(np.float32)
    beat = 60.0 / bpm
    chain = [pb.HighpassFilter(90), pb.LowShelfFilter(250, gain_db=-1.5),
             pb.Compressor(threshold_db=-20, ratio=3, attack_ms=8, release_ms=120)]
    if preset == "airy":
        chain += [pb.PeakFilter(3200, gain_db=1.0, q=0.9), pb.HighShelfFilter(9000, gain_db=3.5)]
    if preset == "club":
        chain += [pb.Distortion(drive_db=4), pb.Compressor(threshold_db=-16, ratio=4, attack_ms=3, release_ms=80),
                  pb.PeakFilter(2500, gain_db=2.0, q=0.8), pb.HighShelfFilter(8000, gain_db=2.0)]
    dry = pb.Pedalboard(chain)(x, sr)
    dry = deess(dry, sr)

    sends = []
    rv = {"clean": 0.10, "airy": 0.22, "club": 0.18}[preset] * reverb * 2
    dl = {"clean": 0.0, "airy": 0.10, "club": 0.16}[preset] * delay * 2
    if rv > 0.001:
        room = {"clean": 0.35, "airy": 0.75, "club": 0.6}[preset]
        sends.append(pb.Pedalboard([pb.HighpassFilter(300), pb.Reverb(room_size=room, damping=0.5, wet_level=1.0,
                                                                       dry_level=0.0, width=1.0),
                                    pb.LowpassFilter(9000)])(dry, sr) * rv)
    if dl > 0.001:
        t = beat * (0.75 if preset == "club" else 0.5)  # dotted 1/8 for club, 1/8 for airy
        sends.append(pb.Pedalboard([pb.HighpassFilter(400), pb.LowpassFilter(6000),
                                    pb.Delay(delay_seconds=t, feedback=0.35, mix=1.0)])(dry, sr) * dl)
    out = dry + sum(sends) if sends else dry
    return out.astype(np.float32)
