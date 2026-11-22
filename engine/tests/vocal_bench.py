"""Synthetic vocals of very different kinds, for testing that chops work on any song, not one.
Each: verse (4 bars) + chorus (4 bars) twice, then an 8-bar instrumental break. Returns
(vocal (2, n), sr, bpm, sung pitch classes or None for rap, n_sung_bars)."""

import numpy as np

SR = 44100


def _tone(f, dur, amp, vib=0.25, harm=7):
    t = np.arange(int(dur * SR)) / SR
    ph = 2 * np.pi * f * t + vib * np.sin(2 * np.pi * 5.5 * t)
    env = np.minimum(1, t * 30) * np.minimum(1, (t[-1] - t) * 30 + 1e-9)
    return (sum(np.sin(k * ph) / k for k in range(1, harm)) * amp * env).astype(np.float32)


def _m2f(m):
    return 440 * 2 ** ((m - 69) / 12)


def _line(notes, bpm, amp, voices=1):
    beat = 60 / bpm
    out = []
    for m, d in notes:
        if m is None:
            out.append(np.zeros(int(d * beat * SR), np.float32))
            continue
        ms = m if isinstance(m, tuple) else (m,)
        x = sum(_tone(_m2f(mm), d * beat, amp / len(ms)) for mm in ms[:voices if voices > 1 else 1] + ms[1:voices] * 0)
        out.append(x)
    return np.concatenate(out)


def _fit(x, bars, bpm):
    n = int(bars * 4 * 60 / bpm * SR)
    return np.pad(x, (0, max(0, n - len(x))))[:n]


def _song(verse, chorus, bpm, amp_v=0.18, amp_c=0.35):
    v = _fit(_line(verse, bpm, amp_v), 4, bpm)
    c = _fit(_line(chorus, bpm, amp_c), 4, bpm)
    x = np.concatenate([v, c, v, c, np.zeros(int(8 * 4 * 60 / bpm * SR), np.float32)])
    return np.stack([x, x])


def pop():
    bpm = 100  # C major
    verse = [(64, .5), (64, .5), (67, 1), (65, .5), (64, .5), (62, 1)] * 2 + [(60, 2), (None, 2)] * 2
    chorus = [(72, 1.5), (71, .5), (69, 1), (67, 1), (69, 2), (None, 2)] * 2
    return _song(verse, chorus, bpm), SR, bpm, {0, 2, 4, 5, 7, 9, 11}, 16


def house():
    bpm = 124  # F minor
    verse = [(65, 1), (68, 1), (70, 2)] * 4
    chorus = [(77, 3), (None, 1), (75, 2), (72, 2)] * 2
    return _song(verse, chorus, bpm), SR, bpm, {5, 7, 8, 10, 0, 1, 3}, 16


def gospel():
    bpm = 72  # Bb major, two voices in thirds
    beat = 60 / bpm

    def two(lo, hi, d, amp):
        return _tone(_m2f(lo), d * beat, amp / 2) + _tone(_m2f(hi), d * beat, amp / 2)
    v = np.concatenate([two(58, 62, 2, .2), two(60, 63, 2, .2), two(62, 65, 3, .2), np.zeros(int(beat * SR), np.float32)] * 2)
    c = np.concatenate([two(65, 69, 4, .4), two(63, 67, 3, .4), np.zeros(int(beat * SR), np.float32)] * 2)
    v, c = _fit(v, 4, bpm), _fit(c, 4, bpm)
    x = np.concatenate([v, c, v, c, np.zeros(int(8 * 4 * beat * SR), np.float32)])
    return np.stack([x, x]), SR, bpm, {10, 0, 2, 3, 5, 7, 9}, 16


def rnb():
    bpm = 96  # A minor with 16th-note runs
    run = [(69, .25), (67, .25), (64, .25), (62, .25)]
    verse = [(64, 1), (62, .5), (60, .5), (57, 2)] * 4
    chorus = run + [(64, 1.5), (None, .5), (60, 1)] + run + [(69, 2), (None, 1)]
    return _song(verse, chorus * 2, bpm), SR, bpm, {9, 11, 0, 2, 4, 5, 7}, 16


def rap():
    """Unpitched, rhythmic: filtered noise bursts (syllables) in a repeating flow."""
    bpm = 90
    beat = 60 / bpm
    rng = np.random.default_rng(5)

    def syl(d, amp):
        n = int(d * SR)
        t = np.arange(n) / SR
        noise = rng.standard_normal(n)
        # a crude formant: band-limit by smoothing, plus a short voiced buzz
        k = np.ones(8) / 8
        f0 = rng.uniform(120, 190) * np.exp(-t * rng.uniform(1, 4))       # speech: pitch glides down each syllable
        buzz = np.sign(np.sin(2 * np.pi * np.cumsum(f0) / SR))
        x = np.convolve(noise, k, mode="same") * 0.6 + 0.4 * buzz * rng.uniform(0.6, 1)
        env = np.exp(-t * 18) * np.minimum(1, t * 200)
        return (x * env * amp).astype(np.float32)
    flow = [.5, .25, .25, .5, .5, .25, .25, .5, 1]       # one bar of delivery
    hook = [.25, .25, .5, .25, .25, .5, .5, .5, 1]
    bar = lambda pat, amp: _fit(np.concatenate([syl(d * beat, amp) for d in pat]), 1, bpm)
    v = np.concatenate([bar(flow, .25) for _ in range(4)])
    c = np.concatenate([bar(hook, .45) for _ in range(4)])
    x = np.concatenate([v, c, v, c, np.zeros(int(8 * 4 * beat * SR), np.float32)])
    return np.stack([x, x]), SR, bpm, None, 16


BENCH = {"pop": pop, "house": house, "gospel": gospel, "rnb": rnb, "rap": rap}
