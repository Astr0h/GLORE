"""Step 2 - Analyse: BPM, key, beat grid, vocal phrases, and a guess at what
kind of sample a file is (vocal / full song / beat).

Every result is a best guess. The UI shows it and lets Seos override it."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import librosa
import numpy as np

ANALYSIS_SR = 22050
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel (DJ key notation): index = pitch class
CAMELOT_MAJOR = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
CAMELOT_MINOR = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]


@dataclass
class Phrase:
    start: float
    end: float
    energy: float  # 0..1 relative loudness of the phrase


@dataclass
class AnalysisResult:
    duration: float
    bpm: float
    bpm_candidates: list[float]
    bpm_confidence: float
    key: str  # e.g. "A minor"
    key_tonic: int  # 0..11
    key_mode: str  # "major" | "minor"
    camelot: str
    key_confidence: float
    beats: list[float]
    downbeats: list[float]
    phrases: list[Phrase] = field(default_factory=list)
    peak_db: float = 0.0
    rms_db: float = 0.0
    kind_guess: str = "full_song"
    kind_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- tempo

def _fold_tempo(bpm: float, lo: float = 70.0, hi: float = 180.0) -> float:
    if bpm <= 0:
        return 120.0
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def estimate_tempo(y: np.ndarray, sr: int) -> tuple[float, list[float], float, np.ndarray]:
    """Returns (bpm, candidates, confidence 0..1, beat_times)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # Confidence: how peaky the tempogram is at the chosen tempo
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tg_mean = tg.mean(axis=1)
    freqs = librosa.tempo_frequencies(tg.shape[0], sr=sr)
    valid = np.isfinite(freqs) & (freqs > 30)
    if valid.any() and tg_mean[valid].max() > 0:
        idx = np.argmin(np.abs(freqs[valid] - bpm))
        confidence = float(tg_mean[valid][idx] / tg_mean[valid].max())
    else:
        confidence = 0.0

    bpm = _fold_tempo(bpm)
    candidates = sorted({round(bpm, 2), round(bpm * 2, 2), round(bpm / 2, 2)})
    candidates = [c for c in candidates if 60 <= c <= 200]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return round(bpm, 2), candidates, round(min(confidence, 1.0), 2), beat_times


def estimate_downbeats(y: np.ndarray, sr: int, beat_times: np.ndarray) -> list[float]:
    """Rough bar starts: pick the beat phase (0-3) with the strongest low-end
    onsets, then take every 4th beat. Assumes 4/4, which house and rap are."""
    if len(beat_times) < 8:
        return [float(t) for t in beat_times[::4]]
    y_low = librosa.effects.preemphasis(y, coef=-0.95)  # tilt toward lows (kicks)
    env = librosa.onset.onset_strength(y=y_low, sr=sr)
    frames = librosa.time_to_frames(beat_times, sr=sr)
    frames = np.clip(frames, 0, len(env) - 1)
    strengths = env[frames]
    phase_scores = [strengths[p::4].mean() for p in range(4)]
    phase = int(np.argmax(phase_scores))
    return [round(float(t), 4) for t in beat_times[phase::4]]


# ---------------------------------------------------------------- key

def estimate_key(y: np.ndarray, sr: int) -> tuple[int, str, float]:
    """Krumhansl-Schmuckler key finding. Returns (tonic 0..11, mode, confidence)."""
    y_harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    profile = chroma.mean(axis=1)
    if profile.sum() == 0:
        return 0, "major", 0.0
    scores = []
    for tonic in range(12):
        scores.append((np.corrcoef(profile, np.roll(MAJOR_PROFILE, tonic))[0, 1], tonic, "major"))
        scores.append((np.corrcoef(profile, np.roll(MINOR_PROFILE, tonic))[0, 1], tonic, "minor"))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    confidence = float(np.clip((best[0] - second[0]) * 5, 0, 1))
    return best[1], best[2], round(confidence, 2)


def key_name(tonic: int, mode: str) -> str:
    return f"{NOTE_NAMES[tonic % 12]} {mode}"


def camelot(tonic: int, mode: str) -> str:
    return (CAMELOT_MAJOR if mode == "major" else CAMELOT_MINOR)[tonic % 12]


def parse_key(text: str) -> tuple[int, str]:
    """'A minor' / 'Am' / 'F# major' -> (tonic, mode)."""
    t = text.strip().replace("♯", "#").replace("♭", "b")
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    mode = "minor" if ("min" in t.lower() or (t.endswith("m") and not t.lower().endswith("maj"))) else "major"
    note = t.split()[0] if " " in t else t.rstrip("m")
    note = note[:2] if len(note) > 1 and note[1] in "#b" else note[:1]
    note = flats.get(note, note).upper().replace("B#", "C")
    if note not in NOTE_NAMES:
        raise ValueError(f"Unrecognised key: {text}")
    return NOTE_NAMES.index(note), mode


# ---------------------------------------------------------------- phrases

def detect_phrases(y: np.ndarray, sr: int, top_db: float = 35.0,
                   merge_gap: float = 0.35, min_len: float = 0.2) -> list[Phrase]:
    """Non-silent regions, with short gaps merged so a sung line stays one phrase."""
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and (start - merged[-1][1]) / sr < merge_gap:
            merged[-1][1] = end
        else:
            merged.append([int(start), int(end)])
    phrases = []
    energies = []
    for s, e in merged:
        if (e - s) / sr < min_len:
            continue
        energies.append(float(np.sqrt(np.mean(y[s:e] ** 2))))
        phrases.append((s, e))
    top = max(energies) if energies else 1.0
    return [Phrase(round(s / sr, 4), round(e / sr, 4), round(en / top, 3))
            for (s, e), en in zip(phrases, energies)]


# ---------------------------------------------------------------- kind guess

def guess_kind(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Acapellas have lots of silence; beats are mostly percussive;
    anything else is treated as a full song."""
    intervals = librosa.effects.split(y, top_db=40)
    voiced = sum(e - s for s, e in intervals) / max(len(y), 1)
    silent_fraction = 1.0 - voiced

    harm, perc = librosa.effects.hpss(y)
    perc_ratio = float(np.sum(perc ** 2) / (np.sum(harm ** 2) + np.sum(perc ** 2) + 1e-12))

    if silent_fraction > 0.2 and perc_ratio < 0.35:
        return "vocal", round(min(1.0, silent_fraction * 2), 2)
    if perc_ratio > 0.55:
        return "beat", round(min(1.0, (perc_ratio - 0.4) * 2.5), 2)
    return "full_song", 0.5


# ---------------------------------------------------------------- main

def analyse(path: str, kind: str | None = None) -> AnalysisResult:
    y, sr = librosa.load(path, sr=ANALYSIS_SR, mono=True)
    duration = len(y) / sr

    bpm, candidates, bpm_conf, beat_times = estimate_tempo(y, sr)
    downbeats = estimate_downbeats(y, sr, beat_times)
    tonic, mode, key_conf = estimate_key(y, sr)
    kind_guess, kind_conf = guess_kind(y, sr)

    effective_kind = kind or kind_guess
    phrases = detect_phrases(y, sr) if effective_kind == "vocal" else []

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0

    return AnalysisResult(
        duration=round(duration, 3),
        bpm=bpm,
        bpm_candidates=candidates,
        bpm_confidence=bpm_conf,
        key=key_name(tonic, mode),
        key_tonic=tonic,
        key_mode=mode,
        camelot=camelot(tonic, mode),
        key_confidence=key_conf,
        beats=[round(float(t), 4) for t in beat_times],
        downbeats=downbeats,
        phrases=phrases,
        peak_db=round(20 * np.log10(peak + 1e-12), 1),
        rms_db=round(20 * np.log10(rms + 1e-12), 1),
        kind_guess=kind_guess,
        kind_confidence=kind_conf,
    )
