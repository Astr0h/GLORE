"""Vocal chops as an instrument.

1. Find the memorable parts of the vocal: the chorus (the section that repeats most and hits hardest),
   the hook phrase inside it (the line that keeps coming back) and the held notes (sustained vowels and
   harmonies, with their pitch).
2. Turn them into a sampler: read the beat's chords bar by bar and re-pitch the held notes so they play
   those chords - stabs, arpeggios, stacked chords - instead of chopping the song at random.
3. Arrange it: the chops play out in the gaps and instrumental parts, sit low under the vocal, tease the
   hook in the intro and stutter the hook phrase into the drops.

Each genre pack plays it its own way (SPEC step 6b):
  house   pitched-up 1/16 stabs on the chord tones, gated, a hook stutter every 4 bars
  rnb     the hook phrase pitched to the chord root (an octave down), sparse, with tails
  gospel  held vowels stacked into full chords, sustained across the bar

    layer, info = make_chops(vocal, sr, bpm, key, style="house", seed=3, harmony=beat, structure=song)

`harmony` (the new beat) gives the chords; `structure` (the original full song, if there is one) finds
the chorus more reliably than the vocal alone. Both are optional.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np

SCALE = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ASR = 22050  # analysis rate


@dataclass
class ChopStyle:
    step_beats: float          # rhythm grid
    density: float             # chance a step fires
    gate_beats: float          # how long a hit rings
    tail: float                # release seconds
    octave: int                # where the instrument plays relative to the source note
    resample: bool             # sampler-style pitch (speed) vs. formant-kept shift
    stutter_every: int         # bars between hook stutters (0 = never)
    gain_db: float
    under_vocal: float         # level while the vocal sings (1 = no ducking)


STYLES: dict[str, ChopStyle] = {
    "house": ChopStyle(step_beats=0.25, density=0.5, gate_beats=0.25, tail=0.0, octave=0, resample=True,
                       stutter_every=4, gain_db=-5.0, under_vocal=0.3),
    "rnb": ChopStyle(step_beats=1.0, density=0.45, gate_beats=1.5, tail=0.5, octave=-12, resample=True,
                     stutter_every=8, gain_db=-6.0, under_vocal=0.35),
    "gospel": ChopStyle(step_beats=4.0, density=1.0, gate_beats=4.0, tail=1.0, octave=0, resample=False,
                        stutter_every=0, gain_db=-8.0, under_vocal=0.5),
}


@dataclass
class Note:
    start: float
    end: float
    midi: float
    stability: float = 1.0
    energy: float = 1.0
    in_chorus: bool = False
    score: float = 0.0


@dataclass
class ChopInfo:
    style: str
    seed: int
    hits: int
    bars: int
    chorus: list = field(default_factory=list)
    hook: tuple | None = None
    notes: list = field(default_factory=list)
    chords: list = field(default_factory=list)
    pattern: list = field(default_factory=list)

    @property
    def slices(self) -> int:
        return len(self.notes) + (1 if self.hook else 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["slices"] = self.slices
        return d


# ---------------------------------------------------------------- 1. the memorable parts

def _mono(x: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    m = (x.mean(0) if x.ndim == 2 else x).astype(np.float32)
    return librosa.resample(m, orig_sr=sr, target_sr=ASR) if sr != ASR else m


def bar_features(y: np.ndarray, bpm: float, offset: float, n_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Per bar: a normalised harmony+timbre vector, and loudness."""
    import librosa
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=ASR, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=ASR, n_mfcc=13, hop_length=hop)[1:]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    fps, bar = ASR / hop, 4 * 60.0 / bpm
    feats, loud = [], []
    for b in range(n_bars):
        s = max(0, int((offset + b * bar) * fps))
        e = max(s + 1, min(chroma.shape[1], int((offset + (b + 1) * bar) * fps)))
        c, t = chroma[:, s:e].mean(1), mfcc[:, s:e].mean(1)
        feats.append(np.concatenate([c / (np.linalg.norm(c) + 1e-9), 0.5 * t / (np.linalg.norm(t) + 1e-9)]))
        loud.append(float(rms[s:e].mean()) if e <= rms.shape[0] else 0.0)
    return np.nan_to_num(np.array(feats)), np.array(loud)


def find_chorus(y: np.ndarray, bpm: float, offset: float, n_bars: int, vocal_active: np.ndarray
                ) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Chorus = bars whose 2-bar neighbourhood comes back elsewhere in the song, loud, with the vocal in.
    Returns [(first_bar, end_bar)] and the per-bar score."""
    if n_bars < 8:
        return [], np.zeros(n_bars)
    f, loud = bar_features(y, bpm, offset, n_bars)
    w = np.concatenate([f, np.roll(f, -1, 0)], 1)  # 2 bars at a time: lines, not single chords
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-9
    sim = w @ w.T
    rep = np.zeros(n_bars)
    for i in range(n_bars):
        far = [j for j in range(n_bars) if abs(j - i) >= 4]  # coming back later, not just the next bar
        if far:
            rep[i] = np.sort(sim[i, far])[-2:].mean()
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    score = 1.2 * z(rep) + 0.6 * z(loud) + 0.8 * vocal_active
    score = np.convolve(score, np.ones(3) / 3, mode="same")
    on = (score >= np.percentile(score, 65)) & (vocal_active > 0.3)
    if on.sum() >= 2:  # now we know what the chorus sounds like: every bar that sounds like it is chorus too
        centre = w[on].mean(0)
        centre /= np.linalg.norm(centre) + 1e-9
        like = w @ centre
        on |= (like >= np.percentile(like[on], 25)) & (vocal_active > 0.3)
    regions, b = [], 0
    while b < n_bars:
        if on[b]:
            e = b
            while e + 1 < n_bars and on[e + 1]:
                e += 1
            if e - b + 1 >= 2:
                regions.append((b, e + 1))
            b = e + 1
        else:
            b += 1
    return regions, score


def find_notes(y: np.ndarray, min_len: float = 0.12) -> list[Note]:
    """Held, pitched notes (sung vowels and harmonies) via pYIN."""
    import librosa
    hop = 512
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=1100, sr=ASR, hop_length=hop, frame_length=2048)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0][: len(f0)]
    midi = librosa.hz_to_midi(np.where(voiced, f0, np.nan))
    notes, i, n, dt = [], 0, len(midi), hop / ASR
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and not np.isnan(midi[j + 1]) and abs(midi[j + 1] - np.nanmedian(midi[i:j + 1])) < 0.6:
            j += 1
        if (j - i + 1) * dt >= min_len:
            seg = midi[i:j + 1]
            notes.append(Note(start=i * dt, end=(j + 1) * dt, midi=float(np.median(seg)),
                              stability=float(1.0 / (0.15 + np.std(seg))), energy=float(rms[i:j + 1].mean())))
        i = j + 1
    return notes


def find_hook(y: np.ndarray, bpm: float, chorus_s: list[tuple[float, float]]) -> tuple[float, float] | None:
    """The line that keeps coming back: a 1-4 beat sung phrase, preferably in the chorus, that best matches others."""
    import librosa
    beat = 60.0 / bpm
    iv = librosa.effects.split(y, top_db=32, frame_length=2048, hop_length=512)
    phrases = [(s / ASR, e / ASR) for s, e in iv if 0.6 * beat <= (e - s) / ASR <= 4.5 * beat]
    if len(phrases) < 2:  # long legato lines: take 2-beat pieces of them
        phrases = [(t, t + 2 * beat) for s, e in iv for t in np.arange(s / ASR, e / ASR - 2 * beat, 2 * beat)]
    in_chorus = [(s, e) for s, e in phrases if any(cs <= (s + e) / 2 < ce for cs, ce in chorus_s)]
    if len(in_chorus) >= 2:  # the hook lives in the chorus
        phrases = in_chorus
    if not phrases:
        return None
    feats = []
    for s, e in phrases:
        seg = y[int(s * ASR):int(e * ASR)]
        c = librosa.feature.chroma_cqt(y=seg, sr=ASR, hop_length=512).mean(1)
        m = librosa.feature.mfcc(y=seg, sr=ASR, n_mfcc=13, hop_length=512)[1:].mean(1)
        v = np.concatenate([c / (np.linalg.norm(c) + 1e-9), 0.5 * m / (np.linalg.norm(m) + 1e-9)])
        feats.append(v / (np.linalg.norm(v) + 1e-9))
    f = np.nan_to_num(np.array(feats))
    sim = f @ f.T
    np.fill_diagonal(sim, 0)
    loud = np.array([np.sqrt(np.mean(y[int(s * ASR):int(e * ASR)] ** 2)) for s, e in phrases])
    inside = np.array([any(cs <= s < ce for cs, ce in chorus_s) for s, _ in phrases], float)
    score = (sim > 0.9).sum(1) + 2 * sim.max(1) + loud / (loud.max() + 1e-9) + 1.5 * inside
    s, e = phrases[int(np.argmax(score))]
    return float(s), float(e)


# ---------------------------------------------------------------- 2. the beat's chords

TRIADS = {"": (0, 4, 7), "m": (0, 3, 7)}


def parse_key(key: str | None) -> tuple[int, str]:
    if not key:
        return 9, "minor"
    parts = key.replace("_", " ").split()
    name = parts[0][0].upper() + parts[0][1:]
    flat = name.endswith("b") and len(name) > 1
    try:
        t = NOTES.index(name.rstrip("b")) - (1 if flat else 0)
    except ValueError:
        t = 9
    return t % 12, "minor" if len(parts) > 1 and parts[1].lower().startswith("min") else "major"


def diatonic_triads(key: str | None) -> list[tuple[int, str]]:
    tonic, mode = parse_key(key)
    sc = SCALE[mode]
    out = []
    for deg, r in enumerate(sc):
        third = (sc[(deg + 2) % 7] - r) % 12
        fifth = (sc[(deg + 4) % 7] - r) % 12
        if fifth == 7:  # skip the diminished one
            out.append(((tonic + r) % 12, "m" if third == 3 else ""))
    return out


def chord_track(harmony: np.ndarray | None, sr: int, bpm: float, offset: float, n_bars: int,
                key: str | None) -> list[tuple[int, str]]:
    """(root pitch class, "" | "m") per bar, from the beat if we have it, kept to chords in the key."""
    tonic, mode = parse_key(key)
    dia = diatonic_triads(key)
    if harmony is None:  # nothing to listen to: a classic progression in the key, 2 bars per chord
        prog = [0, 5, 2, 6] if mode == "minor" else [0, 4, 5, 3]  # i-VI-III-VII / I-V-vi-IV
        roots = [(tonic + SCALE[mode][d]) % 12 for d in prog]
        return [next((c for c in dia if c[0] == roots[(b // 2) % 4]), dia[0]) for b in range(n_bars)]
    import librosa
    y = _mono(harmony, sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=ASR, hop_length=512)
    fps, bar = ASR / 512, 4 * 60.0 / bpm
    out = []
    for b in range(n_bars):
        s = max(0, int((offset + b * bar) * fps))
        e = min(chroma.shape[1], int((offset + (b + 1) * bar) * fps))
        c = chroma[:, s:e].mean(1) if e > s else np.ones(12)
        out.append(max(dia, key=lambda ch: sum(c[(ch[0] + iv) % 12] for iv in TRIADS[ch[1]])))
    return out


# ---------------------------------------------------------------- 3. playing them

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


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Loop a held vowel with crossfades until it lasts n samples: chords held across a bar."""
    if x.shape[1] >= n:
        return x[:, :n]
    fade = max(1, min(x.shape[1] // 4, 2048))
    out = x.copy()
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    while out.shape[1] < n:
        out[:, -fade:] = out[:, -fade:] * (1 - ramp) + x[:, :fade] * ramp
        out = np.concatenate([out, x[:, fade:]], 1)
    return out[:, :n]


def _env(n: int, sr: int, tail: float) -> np.ndarray:
    a = min(n // 4, int(0.004 * sr))
    r = min(n // 2, int(max(0.015, tail) * sr))
    env = np.ones(n, dtype=np.float32)
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.linspace(1, 0, r) ** 2
    return env


def chord_tone(chord: tuple[int, str], tone: int, around: float) -> float:
    """MIDI note of chord tone #tone (0 root, 1 third, 2 fifth, 3 octave) nearest to `around`."""
    root, q = chord
    pc = (root + (TRIADS[q] + (12,))[tone % 4]) % 12
    return float(round((around - pc) / 12) * 12 + pc)


def vocal_activity(mono: np.ndarray, sr: int, bpm: float, offset: float, n_bars: int) -> np.ndarray:
    bar = int(4 * 60.0 / bpm * sr)
    thr = np.percentile(np.abs(mono), 95) * 0.08 + 1e-6
    act = []
    for b in range(n_bars):
        seg = np.abs(mono[int(offset * sr) + b * bar:int(offset * sr) + (b + 1) * bar])
        act.append(float((seg > thr).mean()) if len(seg) else 0.0)
    a = np.array(act)
    return np.clip(a / (a.max() + 1e-9), 0, 1)


def make_chops(vocal: np.ndarray, sr: int, bpm: float, key: str | None = None, style: str = "house",
               seed: int = 0, density: float = 1.0, length: float = 1.0, pitch_spread: float = 1.0,
               offset: float = 0.0, duck_under_vocal: float | None = None, duration: float | None = None,
               harmony: np.ndarray | None = None, structure: np.ndarray | None = None
               ) -> tuple[np.ndarray, ChopInfo]:
    """vocal: (2, n) float32 at sr. density/length/pitch_spread multiply the style (1 = as designed).
    duck_under_vocal: None = the style's own level under the vocal, 0 = no ducking, 1 = silent under it."""
    st = STYLES.get(style, STYLES["house"])
    rng = random.Random(seed)
    if vocal.ndim == 1:
        vocal = np.stack([vocal, vocal])
    total = int((duration or vocal.shape[1] / sr) * sr)
    out = np.zeros((2, total), dtype=np.float32)
    beat = 60.0 / bpm
    bar_s = 4 * beat
    n_bars = max(1, int(np.ceil((total / sr - offset) / bar_s)))
    y = _mono(vocal, sr)
    if float(np.max(np.abs(y))) < 1e-4:
        return out, ChopInfo(style, seed, 0, n_bars)

    # 1. the memorable parts
    act = vocal_activity(vocal.mean(0), sr, bpm, offset, n_bars)
    regions, _ = find_chorus(_mono(structure, sr) if structure is not None else y, bpm, offset, n_bars, act)
    chorus_s = [(offset + a * bar_s, offset + b * bar_s) for a, b in regions]
    notes = find_notes(y)
    if notes:
        e_max = max(n.energy for n in notes) or 1.0
        pcs = [round(n.midi) % 12 for n in notes if any(a <= n.start < b for a, b in chorus_s)]
        for n in notes:
            n.in_chorus = any(a <= n.start < b for a, b in chorus_s)
            common = pcs.count(round(n.midi) % 12) / (len(pcs) or 1)
            n.score = (min(n.end - n.start, 1.5) ** 0.5) * min(n.stability, 4) * (n.energy / e_max) \
                * (2.0 if n.in_chorus else 1.0) * (1 + common)
    voice = sorted(notes, key=lambda n: -n.score)[:3]   # a sampler only needs a couple of good sources
    hook = find_hook(y, bpm, chorus_s)
    chords = chord_track(harmony, sr, bpm, offset, n_bars, key)
    info = ChopInfo(style, seed, 0, n_bars, chorus=[(round(a, 2), round(b, 2)) for a, b in chorus_s],
                    hook=(round(hook[0], 2), round(hook[1], 2)) if hook else None,
                    notes=[{"start": round(n.start, 2), "end": round(n.end, 2), "midi": round(n.midi, 1),
                            "chorus": n.in_chorus} for n in voice],
                    chords=[NOTES[r] + q for r, q in chords])
    if not voice and not hook:
        return out, info

    cache: dict = {}

    def cut(s: float, e: float) -> np.ndarray:
        return vocal[:, int(s * sr):int(e * sr)].astype(np.float32)

    def play(x: np.ndarray, t: float, gain: float):
        start = int(t * sr)
        if 0 <= start < total and x.shape[1]:
            end = min(total, start + x.shape[1])
            out[:, start:end] += x[:, :end - start] * gain
            info.hits += 1

    def sing(n: Note, target: float, dur: float, tail: float) -> np.ndarray:
        """Note n re-pitched to `target`, lasting dur seconds."""
        k = (n.start, round(target, 1), round(dur, 3))
        if k not in cache:
            shift = float(np.clip(target - n.midi, -14, 14))
            raw = cut(n.start, n.end)
            x = _repitch(raw, sr, shift, st.resample)
            x = _fit_length(x, int(dur * sr))
            cache[k] = x * _env(x.shape[1], sr, tail)
        return cache[k]

    lead = voice[0] if voice else Note(hook[0], hook[1], 60.0)
    centre = lead.midi + st.octave + 6 * (pitch_spread - 1)
    steps = int(round(4 / st.step_beats))
    groove = [rng.random() < min(0.95, st.density * density * (1.3 if s % max(1, steps // 4) == 0 else 0.85))
              for s in range(steps * 2)]                       # a 2-bar groove
    tones = [rng.choice((0, 1, 2, 3)) for _ in range(steps * 2)]
    under = st.under_vocal if duck_under_vocal is None else 1.0 - duck_under_vocal
    voicing = (0, 1, 2)
    first_vocal = next((b for b in range(n_bars) if act[b] > 0.3), n_bars)
    gate = st.gate_beats * beat * length
    for b in range(n_bars):
        t0 = offset + b * bar_s
        chord = chords[b]
        if b and b % 8 == 0:   # every 8 bars: change a little, so it moves with the song
            groove = [g if rng.random() > 0.2 else not g for g in groove]
            tones = [t if rng.random() > 0.3 else rng.choice((0, 1, 2, 3)) for t in tones]
        level = 1.0  # ducking under the vocal happens below, following the vocal moment to moment
        in_chorus = any(a <= b < e for a, e in regions)
        if style == "gospel" and voice:
            if b % 4 == 0:                              # a new voicing every 4 bars (the seed decides)
                voicing = rng.choice(((0, 1, 2), (1, 2, 3), (2, 3, 1), (0, 2, 3)))
            for i, n in enumerate(voice[:3]):          # each voice sings a chord tone: a stacked chord
                play(sing(n, chord_tone(chord, voicing[i], centre + (12 if i == 2 else 0)), bar_s * length, st.tail),
                     t0, level / np.sqrt(len(voice[:3])))
            info.pattern.append({"bar": b, "chord": NOTES[chord[0]] + chord[1]})
        else:
            for s in range(steps):
                gi = (b % 2) * steps + s
                if not groove[gi]:
                    continue
                if style == "rnb" and hook and rng.random() < 0.5:   # the hook line, moved to the chord root
                    h = Note(hook[0], hook[1], lead.midi)
                    x = sing(h, chord_tone(chord, 0, centre), gate, st.tail)
                else:
                    x = sing(lead, chord_tone(chord, tones[gi], centre), gate, st.tail)
                play(x, t0 + s * st.step_beats * beat, level)
                if b < 2:
                    info.pattern.append({"bar": b, "step": s, "tone": tones[gi]})
        if hook and b < first_vocal and b % 4 == 3:        # tease the hook in the intro
            seg = cut(*hook)
            play(seg * _env(seg.shape[1], sr, 0.3), t0 + 2 * beat, 0.8)
        if hook and st.stutter_every and b % st.stutter_every == st.stutter_every - 1 and (in_chorus or act[b] < 0.3):
            piece = cut(hook[0], hook[0] + beat / 2)       # "I-I-I-I want" into the next bar
            for k in range(4):
                p = piece[:, :int(beat / 4 * sr)] if k < 3 else piece
                play(p * _env(p.shape[1], sr, 0.0), t0 + 3 * beat + k * beat / 4, 0.9)

    if st.tail:
        import pedalboard
        out = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.6, wet_level=0.22, dry_level=0.85)])(out, sr)
    if under < 1.0:  # step back while the vocal sings, come forward in every gap
        m = vocal.mean(0)
        m = m[:total] if len(m) >= total else np.pad(m, (0, total - len(m)))
        out *= vocal_gap_gain(m, sr, 1.0 - under)
    peak = float(np.max(np.abs(out))) or 1.0
    out *= (10 ** (st.gain_db / 20)) / peak
    return out.astype(np.float32), info


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
