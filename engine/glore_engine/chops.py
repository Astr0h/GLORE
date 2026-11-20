"""Vocal chops: slice a vocal into short pieces and play them back as a rhythmic instrument.

Each genre pack has a chop style (SPEC step 6b):
  house   short 1/16-1/8 stabs, pitched up, repeated, gated
  rnb     half-bar to phrase cuts, sparse, often pitched down, with tails
  gospel  long held vowels, stretched and stacked into chords

    layer, info = make_chops(vocal, sr, bpm, key, style="house", seed=3)

`layer` is a stereo float32 array the length of the vocal, on the bar grid starting at `offset`
seconds (the first downbeat). Sliders on top of the style: density, length, pitch spread.
Chops duck under the real vocal by default, so they answer it instead of fighting it.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np

# semitones from the key's root that stay in key (minor uses the natural minor scale)
SCALE = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


@dataclass
class ChopStyle:
    slice_beats: tuple[float, float]  # min/max slice length, in beats
    density: float                    # chance a 16th step gets a hit (house) / a slot gets a chop
    step_beats: float                 # grid the chops sit on
    pitches: tuple[int, ...]          # pitch choices in semitones (kept in key)
    reverse: float                    # chance a slice plays backwards
    tail: float                       # seconds of release on each hit (0 = gated)
    repeat: float                     # chance a step repeats the previous slice (stutter)
    stack: tuple[int, ...] = ()       # extra voices on top (gospel chords)
    resample: bool = True             # pitch by speed (classic sampler) vs. formant-kept shift
    gain_db: float = -4.0


STYLES: dict[str, ChopStyle] = {
    "house": ChopStyle(slice_beats=(0.25, 0.5), density=0.42, step_beats=0.25, pitches=(0, 5, 7, 12),
                       reverse=0.08, tail=0.0, repeat=0.35, gain_db=-5.0),
    "rnb": ChopStyle(slice_beats=(2.0, 4.0), density=0.45, step_beats=2.0, pitches=(0, -5, -12),
                     reverse=0.15, tail=0.6, repeat=0.1, resample=True, gain_db=-6.0),
    "gospel": ChopStyle(slice_beats=(4.0, 8.0), density=0.6, step_beats=4.0, pitches=(0, 0, -5),
                        reverse=0.0, tail=1.2, repeat=0.0, stack=(0, 7, 12), resample=False, gain_db=-8.0),
}


@dataclass
class ChopInfo:
    style: str
    seed: int
    slices: int
    hits: int
    bars: int
    pattern: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- finding slices

def find_slices(mono: np.ndarray, sr: int, min_len: float, max_len: float, limit: int = 24) -> list[tuple[int, int]]:
    """Onset-to-onset pieces that are loud and tonal (sung vowels chop better than breaths or 's' sounds)."""
    import librosa
    onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="samples", backtrack=True)
    if len(onsets) < 2:
        onsets = np.arange(0, len(mono), int(sr * max_len))
    rms_all = float(np.sqrt(np.mean(mono ** 2))) + 1e-9
    found = []
    for i, start in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else len(mono)
        length = int(np.clip(nxt - start, min_len * sr, max_len * sr))
        end = min(len(mono), start + length)
        seg = mono[start:end]
        if len(seg) < min_len * sr * 0.8:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 0.5 * rms_all:
            continue
        flat = float(np.mean(librosa.feature.spectral_flatness(y=seg, n_fft=1024)))
        found.append((rms * (1.0 - min(flat * 4, 0.9)), int(start), int(end)))
    found.sort(reverse=True)
    best = sorted((s, e) for _, s, e in found[:limit])
    return best


# ---------------------------------------------------------------- playing slices

def _repitch(x: np.ndarray, sr: int, semis: float, resample: bool) -> np.ndarray:
    if abs(semis) < 1e-3:
        return x
    if resample:  # sampler-style: higher = shorter and brighter
        rate = 2 ** (semis / 12)
        n = max(16, int(x.shape[1] / rate))
        idx = np.linspace(0, x.shape[1] - 1, n)
        return np.stack([np.interp(idx, np.arange(x.shape[1]), ch) for ch in x]).astype(np.float32)
    import pedalboard
    return pedalboard.time_stretch(x, sr, stretch_factor=1.0, pitch_shift_in_semitones=float(semis),
                                   high_quality=True, preserve_formants=True).astype(np.float32)


def _envelope(n: int, sr: int, tail: float) -> np.ndarray:
    a = min(n // 4, int(0.004 * sr))          # 4 ms click-free attack
    r = min(n // 2, int(max(0.012, tail) * sr))
    env = np.ones(n, dtype=np.float32)
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.linspace(1, 0, r) ** (2 if tail else 1)
    return env


def in_key(pitches: tuple[int, ...], key: str | None, rng: random.Random, spread: float) -> int:
    """Pick a pitch; with spread > 1 also allow other in-key intervals."""
    p = rng.choice(pitches)
    if spread > 1.0 and rng.random() < (spread - 1.0):
        mode = "minor" if key and "minor" in key else "major"
        p += rng.choice(SCALE[mode]) * rng.choice((1, -1))
    return int(np.clip(p, -12, 12))


# ---------------------------------------------------------------- the layer

def make_chops(vocal: np.ndarray, sr: int, bpm: float, key: str | None = None, style: str = "house",
               seed: int = 0, density: float = 1.0, length: float = 1.0, pitch_spread: float = 1.0,
               offset: float = 0.0, duck_under_vocal: float = 0.75, duration: float | None = None
               ) -> tuple[np.ndarray, ChopInfo]:
    """vocal: (2, n) float32. density/length/pitch_spread are multipliers on the style (1 = the style as is)."""
    st = STYLES.get(style, STYLES["house"])
    rng = random.Random(seed)
    if vocal.ndim == 1:
        vocal = np.stack([vocal, vocal])
    beat = 60.0 / bpm
    mono = vocal.mean(0)
    lo, hi = (b * beat * length for b in st.slice_beats)
    slices = find_slices(mono, sr, lo, hi)
    total = int((duration or vocal.shape[1] / sr) * sr)
    out = np.zeros((2, total), dtype=np.float32)
    if not slices:
        return out, ChopInfo(style, seed, 0, 0, 0, [])
    pool = rng.sample(slices, k=min(len(slices), 4 if style == "house" else 3))  # a chop uses a few pieces

    steps_per_bar = int(round(4 / st.step_beats))
    pattern_bars = 2 if style == "house" else 4

    def new_pattern():
        pat, last = [], None
        for s in range(steps_per_bar * pattern_bars):
            strong = (s % max(1, steps_per_bar // 4) == 0)
            p = st.density * density * (1.25 if strong else 0.8)
            if rng.random() < min(0.95, p):
                if last is not None and rng.random() < st.repeat:
                    hit = dict(last)
                else:
                    hit = {"slice": rng.randrange(len(pool)), "pitch": in_key(st.pitches, key, rng, pitch_spread),
                           "reverse": rng.random() < st.reverse}
                hit["step"] = s
                pat.append(hit)
                last = hit
            else:
                last = None
        return pat

    pattern = new_pattern()
    bar_len = 4 * beat
    n_bars = int(np.ceil((total / sr - offset) / bar_len))
    hits = 0
    rendered: dict = {}
    for bar0 in range(0, n_bars, pattern_bars):
        if bar0 and bar0 % 8 == 0 and rng.random() < 0.5:  # every 8 bars, sometimes reshuffle a little
            pattern = [h if rng.random() > 0.3 else {**h, "pitch": in_key(st.pitches, key, rng, pitch_spread)}
                       for h in pattern]
        for h in pattern:
            t = offset + bar0 * bar_len + h["step"] * st.step_beats * beat
            start = int(t * sr)
            if start >= total or start < 0:
                continue
            k = (h["slice"], h["pitch"], h["reverse"])
            if k not in rendered:
                s0, s1 = pool[h["slice"]]
                x = vocal[:, s0:s1].astype(np.float32)
                if h["reverse"]:
                    x = x[:, ::-1]
                layers = [_repitch(x, sr, h["pitch"] + v, st.resample) for v in (st.stack or (0,))]
                n = max(l.shape[1] for l in layers)
                x = sum(np.pad(l, ((0, 0), (0, n - l.shape[1]))) for l in layers) / np.sqrt(len(layers))
                # the slot length caps a hit (gated), unless the style lets it ring
                slot = int(st.step_beats * beat * sr * (1 if st.tail == 0 else 2))
                x = x[:, :max(slot, int(0.03 * sr))] if st.tail == 0 else x
                rendered[k] = (x * _envelope(x.shape[1], sr, st.tail)).astype(np.float32)
            x = rendered[k]
            end = min(total, start + x.shape[1])
            out[:, start:end] += x[:, :end - start]
            hits += 1

    if st.tail:  # tails: a little room so long chops sit back
        import pedalboard
        out = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.6, wet_level=0.25, dry_level=0.8)])(out, sr)

    if duck_under_vocal > 0:  # make room: chops answer the vocal rather than sit on it
        out *= vocal_gap_gain(mono[:total] if len(mono) >= total else np.pad(mono, (0, total - len(mono))),
                              sr, duck_under_vocal)
    peak = float(np.max(np.abs(out))) or 1.0
    out *= (10 ** (st.gain_db / 20)) / peak
    return out.astype(np.float32), ChopInfo(style, seed, len(pool), hits, n_bars, pattern)


def vocal_gap_gain(mono: np.ndarray, sr: int, depth: float, release: float = 0.25) -> np.ndarray:
    """1 where the vocal is quiet, down to (1 - depth) where it's singing; smoothed so it breathes."""
    hop = 512
    frames = len(mono) // hop + 1
    rms = np.sqrt(np.convolve(np.pad(mono ** 2, (0, frames * hop - len(mono))).reshape(frames, hop).mean(1),
                              np.ones(4) / 4, mode="same"))
    act = np.clip(rms / (np.percentile(rms, 90) + 1e-9), 0, 1)
    coef = np.exp(-1.0 / (release * sr / hop))
    sm = np.zeros_like(act)
    for i, a in enumerate(act):
        sm[i] = a if i == 0 or a > sm[i - 1] else coef * sm[i - 1] + (1 - coef) * a
    g = 1.0 - depth * sm
    return np.interp(np.arange(len(mono)), np.arange(frames) * hop, g).astype(np.float32)
