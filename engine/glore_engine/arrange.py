"""Arrange a generated beat around a vocal, using the original song as the map.

A generated beat has no idea where the verse, chorus, breakdowns or drops are.
The song the vocal came from does: its producer already arranged around this vocal.
So we read the original's drum / bass / music stems bar by bar, turn that into a
section map, and replay the same moves on the generated beat's own stems:

- drums out in the original  -> drums muted, music low-passed (breakdown)
- drums quiet (intro)         -> kick filtered out, hats/percussion only
- bass out                    -> bass muted
- last bar of a breakdown     -> the filter opens up into the drop
- a one-bar drum gap          -> a drum cut right before the drop
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

MUTE_DB = -24.0   # this far under the stem's normal level = the stem is out
LIGHT_DB = -5.0   # between LIGHT and MUTE = quieter / thinner part
RAMP_S = 0.03     # click-free gain changes


@dataclass
class Bar:
    drums: str   # full | light | off
    bass: str    # on | off
    music: str   # full | low


def bar_levels(audio: np.ndarray, sr: int, first_downbeat: float, bar_s: float) -> np.ndarray:
    """RMS level (dB) of each bar. `audio` is (channels, samples) or (samples,)."""
    x = audio.mean(0) if audio.ndim == 2 else audio
    out = []
    b = 0
    while True:
        a = int(round((first_downbeat + b * bar_s) * sr))
        e = int(round((first_downbeat + (b + 1) * bar_s) * sr))
        if e > len(x):
            break
        seg = x[max(a, 0):e]
        out.append(10 * np.log10(float(np.mean(seg * seg)) + 1e-10))
        b += 1
    return np.array(out)


def _relative(levels: np.ndarray) -> np.ndarray:
    """Each bar's level against the stem's normal 'playing' level (75th percentile of live bars)."""
    if not len(levels):
        return levels
    live = levels[levels > levels.max() - 40]
    ref = np.percentile(live, 75) if len(live) else levels.max()
    return levels - ref


def section_map(drums: np.ndarray, bass: np.ndarray, music: np.ndarray) -> list[Bar]:
    """Per-bar arrangement from the original song's stem levels (dB per bar)."""
    n = min(len(drums), len(bass), len(music))
    d, b, m = _relative(drums[:n]), _relative(bass[:n]), _relative(music[:n])
    bars = []
    for i in range(n):
        dr = "off" if d[i] < MUTE_DB else "light" if d[i] < LIGHT_DB else "full"
        bars.append(Bar(drums=dr, bass="off" if b[i] < MUTE_DB else "on",
                        music="low" if (m[i] < -8 or dr == "off") else "full"))
    _smooth(bars)
    return bars


def _smooth(bars: list[Bar]) -> None:
    """Drop one-bar blips (808 gaps, drum fills, a quiet bar of music) inside full sections.
    Drum cuts stay: a bar or two with no drums before a drop is a deliberate move."""
    n = len(bars)
    for field, normal in (("bass", "on"), ("music", "full"), ("drums", "full")):
        i = 0
        while i < n:
            if getattr(bars[i], field) == normal or (field == "drums" and bars[i].drums == "off"):
                i += 1
                continue
            j = i
            while j < n and getattr(bars[j], field) == getattr(bars[i], field):
                j += 1
            inside = i > 0 and j < n
            if inside and j - i <= 1 and not (field == "music" and bars[i].drums == "off"):
                for k in range(i, j):
                    setattr(bars[k], field, normal)
            i = j


def sections(bars: list[Bar]) -> list[dict]:
    """Merge runs of identical bars into readable sections."""
    out: list[dict] = []
    for i, bar in enumerate(bars):
        d = asdict(bar)
        if out and all(out[-1][k] == v for k, v in d.items()):
            out[-1]["end_bar"] = i + 1
        else:
            out.append({"start_bar": i, "end_bar": i + 1, **d})
    for s in out:
        s["label"] = ("breakdown" if s["drums"] == "off" and s["end_bar"] - s["start_bar"] > 2
                      else "drum cut" if s["drums"] == "off"
                      else "intro" if s["drums"] == "light" and s["start_bar"] == 0
                      else "light" if s["drums"] == "light"
                      else "full")
    return out


def _curve(values: np.ndarray, starts: np.ndarray, n: int, sr: int) -> np.ndarray:
    """Piecewise-constant per-bar values -> per-sample curve with short ramps."""
    c = np.full(n, values[-1] if len(values) else 1.0, dtype=np.float32)
    edges = np.clip(np.append(starts, n), 0, n)
    head = np.clip(starts[0], 0, n) if len(starts) else 0
    c[:head] = values[0] if len(values) else 1.0
    for v, a, e in zip(values, edges[:-1], edges[1:]):
        c[a:e] = v
    r = max(1, int(RAMP_S * sr))
    k = np.ones(r, dtype=np.float32) / r
    return np.convolve(c, k, mode="same").astype(np.float32)


def arrange_stems(stems: dict[str, np.ndarray], sr: int, bars: list[Bar],
                  bar_starts_s: list[float]) -> np.ndarray:
    """Apply the section map to a beat's stems (drums, bass, other[, vocals]) -> stereo beat.
    bar_starts_s = where each song bar lands in the beat's timeline."""
    from pedalboard import Pedalboard, HighpassFilter, LowpassFilter

    n = max(s.shape[1] for s in stems.values())
    pad = lambda x: np.pad(x, ((0, 0), (0, n - x.shape[1]))).astype(np.float32)
    drums = pad(stems["drums"])
    bass = pad(stems["bass"])
    music = pad(stems["other"]) + (pad(stems["vocals"]) if "vocals" in stems else 0)
    starts = np.array([int(round(t * sr)) for t in bar_starts_s])
    bar_n = int(np.median(np.diff(starts))) if len(starts) > 1 else sr * 2

    # drums: off -> muted; light -> kick removed (high-passed), slightly quieter
    d_gain = np.array([0.0 if b.drums == "off" else 1.0 for b in bars], np.float32)
    d_thin = np.array([1.0 if b.drums == "light" else 0.0 for b in bars], np.float32)
    hp = Pedalboard([HighpassFilter(cutoff_frequency_hz=300)])(drums, sr)
    thin = _curve(d_thin, starts, n, sr)
    drums_out = (drums * (1 - thin) + hp * 0.8 * thin) * _curve(d_gain, starts, n, sr)

    bass_out = bass * _curve(np.array([1.0 if b.bass == "on" else 0.0 for b in bars], np.float32),
                             starts, n, sr)

    # music: low-passed in breakdowns; the filter opens over the last bar into the drop
    lp = Pedalboard([LowpassFilter(cutoff_frequency_hz=700)])(music, sr)
    wet = _curve(np.array([1.0 if b.music == "low" else 0.0 for b in bars], np.float32),
                 starts, n, sr)
    for i in range(len(bars) - 1):
        if bars[i].music == "low" and bars[i + 1].music == "full" and i < len(starts):
            a = starts[i] + bar_n // 2 if bars[max(i - 1, 0)].music == "low" else starts[i]
            e = min(starts[i] + bar_n, n)
            if 0 <= a < e:
                wet[a:e] = np.linspace(1, 0, e - a) ** 2
    music_out = music * (1 - wet) + lp * 1.15 * wet

    return drums_out + bass_out + music_out
