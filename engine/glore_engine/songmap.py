"""The producer's ears: one compact description of a song that a producer (human or AI) can plan from.

    m = build(vocal, sr, bpm, key, offset, music={"drums": d, "bass": b, "other": o})

Everything is in bars on the vocal's own grid (bar 0 = `offset` seconds). The map says what is where:
sections (verse / chorus / instrumental) with their energy, every sung phrase, the hook, the chorus
melody, the best held notes for sampling, how the vocal is delivered (sung or rapped) and, for a full
song, the original's chords and energy.
"""

from __future__ import annotations

import numpy as np

from . import chops


def _sections(act: np.ndarray, regions: list[tuple[int, int]], energy: np.ndarray) -> list[dict]:
    n = len(act)
    lab = ["instrumental"] * n
    for b in range(n):
        if act[b] > 0.25:
            lab[b] = "chorus" if any(a <= b < e for a, e in regions) else "verse"
    # absorb 1-bar blips into their neighbours
    for b in range(1, n - 1):
        if lab[b] != lab[b - 1] and lab[b] != lab[b + 1] and lab[b - 1] == lab[b + 1]:
            lab[b] = lab[b - 1]
    out, s = [], 0
    counts: dict[str, int] = {}
    for b in range(1, n + 1):
        if b == n or lab[b] != lab[s]:
            counts[lab[s]] = counts.get(lab[s], 0) + 1
            out.append({"name": f"{lab[s]} {counts[lab[s]]}", "kind": lab[s], "start": s, "bars": b - s,
                        "energy": round(float(energy[s:b].mean()), 2), "vocal": round(float(act[s:b].mean()), 2)})
            s = b
    return out


def build(vocal: np.ndarray, sr: int, bpm: float, key: str | None, offset: float = 0.0,
          music: dict[str, np.ndarray] | None = None, max_phrases: int = 70) -> dict:
    import librosa
    bar_s = 240.0 / bpm
    beat = 60.0 / bpm
    n_bars = max(1, int(np.ceil((vocal.shape[1] / sr - offset) / bar_s)))
    y = chops._mono(vocal, sr)
    act = chops.vocal_activity(vocal.mean(0), sr, bpm, offset, n_bars)
    full = None
    if music:
        full = sum(music.values())
        k = min(full.shape[1], vocal.shape[1])
        full = full[:, :k] + vocal[:, :k]
    struct = chops._mono(full, sr) if full is not None else y
    regions, _ = chops.find_chorus(struct, bpm, offset, n_bars, act)
    _, loud = chops.bar_features(struct, bpm, offset, n_bars)
    energy = np.clip(loud / (np.percentile(loud, 95) + 1e-9), 0, 1)
    to_bar = lambda t: round((t - offset) / bar_s, 3)

    iv = librosa.effects.split(y, top_db=34, frame_length=2048, hop_length=512)
    merged: list[list[float]] = []
    for s, e in iv:
        s, e = s / chops.ASR, e / chops.ASR
        if merged and s - merged[-1][1] < beat / 2:
            merged[-1][1] = e
        elif e - s > 0.12:
            merged.append([s, e])
    peak = max((float(np.sqrt(np.mean(y[int(s * chops.ASR):int(e * chops.ASR)] ** 2))) for s, e in merged), default=1.0) or 1.0
    in_chorus = lambda b: any(a <= b < e for a, e in regions)
    phrases = [{"bar": to_bar(s), "bars": round((e - s) / bar_s, 3),
                "loud": round(float(np.sqrt(np.mean(y[int(s * chops.ASR):int(e * chops.ASR)] ** 2))) / peak, 2),
                "chorus": bool(in_chorus(to_bar(s)))} for s, e in merged]
    if len(phrases) > max_phrases:  # keep the loudest, in order
        keep = sorted(sorted(range(len(phrases)), key=lambda i: -phrases[i]["loud"])[:max_phrases])
        phrases = [phrases[i] for i in keep]

    notes = chops.find_notes(y)
    sung_ratio, _ = chops.delivery(y)
    chorus_s = [(offset + a * bar_s, offset + b * bar_s) for a, b in regions]
    hook = chops.find_hook(y, bpm, chorus_s)
    for n in notes:
        n.in_chorus = any(a <= n.start < b for a, b in chorus_s)
    e_max = max((n.energy for n in notes), default=1.0) or 1.0
    best = sorted(notes, key=lambda n: -(min(n.end - n.start, 1.5) ** 0.5) * min(n.stability, 4) * (n.energy / e_max)
                  * (2.0 if n.in_chorus else 1.0))[:6]
    m = {"bpm": round(float(bpm), 2), "key": key, "bars": n_bars, "bar_seconds": round(bar_s, 3),
         "kind": "song" if music else "vocal",
         "delivery": {"sung": round(float(sung_ratio), 2), "style": "sung" if sung_ratio >= 0.3 else "rapped"},
         "sections": _sections(act, regions, energy),
         "vocal_per_bar": [round(float(a), 1) for a in act],
         "phrases": phrases,
         "hook": {"bar": to_bar(hook[0]), "bars": round((hook[1] - hook[0]) / bar_s, 3)} if hook else None,
         "held_notes": [{"bar": to_bar(n.start), "bars": round((n.end - n.start) / bar_s, 3), "midi": int(round(n.midi)),
                         "chorus": n.in_chorus} for n in sorted(best, key=lambda n: n.start)],
         "sung_notes": sorted({int(round(n.midi)) % 12 for n in notes})}
    if music:
        harm = sum(v for k, v in music.items() if k != "drums")
        ch = chops.chord_track(harm, sr, bpm, offset, n_bars, key)
        m["original"] = {"chords": [chops.NOTES[r] + q for r, q in ch],
                         "energy_per_bar": [round(float(e), 1) for e in energy]}
    return m


def riff_loop(vocal: np.ndarray, sr: int, bpm: float, key: str | None, offset: float = 0.0, style: str = "house",
              seed: int = 0, bars: int = 4) -> np.ndarray | None:
    """The vocal-chop riff (the song's own melody, sampler-style) as a loop the producer can place anywhere."""
    bar = int(round(240.0 / bpm * sr))
    off = int(round(offset * sr))
    n_bars = int(np.ceil((vocal.shape[1] - off) / bar))
    padded = np.zeros((2, off + (n_bars + bars + 6) * bar), dtype=np.float32)
    padded[:, :vocal.shape[1]] = vocal
    layer, info = chops.make_chops(padded, sr, bpm, key, style=style, seed=seed, offset=offset, duck_under_vocal=0.0)
    start = off + (n_bars + (n_bars % 2) + 2) * bar      # an even bar, clear of the vocal's tail
    loop = layer[:, start:start + bars * bar]
    return loop.astype(np.float32) if float(np.max(np.abs(loop))) > 1e-4 else None
