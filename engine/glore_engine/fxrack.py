"""GLORE's effects rack: everything a producer reaches for, as plain functions on (2, n) float32 audio.

    y = apply_fx(x, sr, [{"type": "filter", "mode": "lp", "cutoff": [[0, 200], [1, 18000]]},
                         {"type": "delay", "time": 0.75, "feedback": 0.4, "mix": 0.25}], ctx)

ctx = {"bpm": 126, "pos0": 8.0, "pos1": 16.0}: the tempo and where this audio sits (in bars), so effects
run in musical time. Any numeric parameter marked (auto) can be a number or automation points
[[pos, value], ...]; pos is 0..1 across the audio when ctx["unit"] == "frac" (clips), else bars (tracks).

Effects (type):
  gain db(auto) | fade in/out (beats) | pan pos(auto) | width amount
  eq hp lp low_db high_db bells=[[hz, db, q]] | filter mode cutoff(auto) resonance drive | telephone
  wobble rate(beats) min_hz max_hz | saturate drive_db mix | bitcrush bits | compress threshold_db ratio
  reverb size damping mix(auto) | delay time(beats) feedback mix(auto) ping_pong | reverse_reverb size
  chorus | phaser rate_hz depth | tremolo rate(beats) depth shape | gate pattern="x.x.x..x" (16ths)
  autopan rate(beats) depth | pump depth_db rate(beats) | pitch semis formants | vocoder notes|per_bar bands mix
  reverse | stutter slice(beats) length(beats) | tape_stop beats | octave_up mix | octave_down mix
Synth sources (make_synth): riser, downlifter, impact, sub_drop, noise_sweep, snare_roll.
"""

from __future__ import annotations

import numpy as np
import pedalboard as pb

TAIL_FX = {"reverb", "delay", "reverse_reverb"}


# ---------------------------------------------------------------- automation

def auto(v, n: int, ctx: dict, default: float = 0.0):
    """A parameter as a float, or a per-sample float32 curve if it's automation points."""
    if v is None:
        return float(default)
    if isinstance(v, (int, float)):
        return float(v)
    pts = sorted((float(p[0]), float(p[1])) for p in v if isinstance(p, (list, tuple)) and len(p) >= 2)
    if not pts:
        return float(default)
    if len(pts) == 1:
        return pts[0][1]
    p0, p1 = (0.0, 1.0) if ctx.get("unit") == "frac" else (ctx.get("pos0", 0.0), ctx.get("pos1", 1.0))
    pos = np.linspace(p0, p1, n, dtype=np.float64)
    return np.interp(pos, [p[0] for p in pts], [p[1] for p in pts]).astype(np.float32)


def _is_curve(v) -> bool:
    return isinstance(v, np.ndarray)


def _beat(ctx) -> float:
    return 60.0 / float(ctx.get("bpm", 120.0))


def _mix(dry, wet, mix):
    n = min(dry.shape[1], wet.shape[1])
    m = mix[:n] if _is_curve(mix) else mix
    return (dry[:, :n] * (1 - m) + wet[:, :n] * m).astype(np.float32)


def _board(x, sr, *plugins):
    return pb.Pedalboard(list(plugins))(x, sr).astype(np.float32)


# ---------------------------------------------------------------- level, space

def fx_gain(x, sr, p, ctx):
    g = auto(p.get("db", 0), x.shape[1], ctx)
    return (x * (10 ** (g / 20))).astype(np.float32)


def fx_fade(x, sr, p, ctx):
    y = x.copy()
    for key, rev in (("in", False), ("out", True)):
        k = min(x.shape[1], int(float(p.get(key, 0)) * _beat(ctx) * sr))
        if k > 1:
            ramp = np.linspace(0, 1, k, dtype=np.float32) ** 1.5
            if rev:
                y[:, -k:] *= ramp[::-1]
            else:
                y[:, :k] *= ramp
    return y


def fx_pan(x, sr, p, ctx):
    pos = np.clip(auto(p.get("pos", 0), x.shape[1], ctx), -1, 1)
    ang = (pos + 1) * np.pi / 4
    return np.stack([x[0] * np.cos(ang) * 1.414, x[1] * np.sin(ang) * 1.414]).astype(np.float32)


def fx_width(x, sr, p, ctx):
    a = float(p.get("amount", 1.0))
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2 * a
    return np.stack([mid + side, mid - side]).astype(np.float32)


def fx_autopan(x, sr, p, ctx):
    t = np.arange(x.shape[1]) / sr
    lfo = np.sin(2 * np.pi * t / (float(p.get("rate", 2)) * _beat(ctx))) * float(p.get("depth", 0.7))
    ang = (lfo + 1) * np.pi / 4
    return np.stack([x[0] * np.cos(ang) * 1.414, x[1] * np.sin(ang) * 1.414]).astype(np.float32)


# ---------------------------------------------------------------- tone

def fx_eq(x, sr, p, ctx):
    chain = []
    if p.get("hp"):
        chain.append(pb.HighpassFilter(float(p["hp"])))
    if p.get("lp"):
        chain.append(pb.LowpassFilter(float(p["lp"])))
    if p.get("low_db"):
        chain.append(pb.LowShelfFilter(float(p.get("low_hz", 200)), gain_db=float(p["low_db"])))
    if p.get("high_db"):
        chain.append(pb.HighShelfFilter(float(p.get("high_hz", 6000)), gain_db=float(p["high_db"])))
    for b in p.get("bells") or []:
        chain.append(pb.PeakFilter(float(b[0]), gain_db=float(b[1]), q=float(b[2]) if len(b) > 2 else 1.0))
    return _board(x, sr, *chain) if chain else x


def fx_filter(x, sr, p, ctx):
    """Resonant ladder filter; the cutoff can sweep."""
    mode = {"lp": pb.LadderFilter.Mode.LPF24, "hp": pb.LadderFilter.Mode.HPF24,
            "bp": pb.LadderFilter.Mode.BPF24}.get(p.get("mode", "lp"), pb.LadderFilter.Mode.LPF24)
    cut = auto(p.get("cutoff", 1000), x.shape[1], ctx, 1000)
    f = pb.LadderFilter(mode=mode, cutoff_hz=1000, resonance=float(np.clip(p.get("resonance", 0.2), 0, 0.95)),
                        drive=float(max(1.0, p.get("drive", 1.0))))
    if not _is_curve(cut):
        f.cutoff_hz = float(np.clip(cut, 20, 20000))
        return f(x, sr).astype(np.float32)
    out, block = np.empty_like(x), 512
    for s in range(0, x.shape[1], block):
        f.cutoff_hz = float(np.clip(cut[s], 20, 20000))
        out[:, s:s + block] = f(x[:, s:s + block], sr, reset=False)
    return out


def fx_telephone(x, sr, p, ctx):
    return _board(x, sr, pb.HighpassFilter(350), pb.HighpassFilter(350), pb.LowpassFilter(3200), pb.LowpassFilter(3200),
                  pb.Distortion(drive_db=float(p.get("drive_db", 8))), pb.Gain(gain_db=-4))


def fx_wobble(x, sr, p, ctx):
    t = np.arange(x.shape[1]) / sr
    lfo = 0.5 - 0.5 * np.cos(2 * np.pi * t / (float(p.get("rate", 1)) * _beat(ctx)))
    lo, hi = float(p.get("min_hz", 200)), float(p.get("max_hz", 6000))
    cut = (lo * (hi / lo) ** lfo).astype(np.float32)
    return _filter_curve(x, sr, cut, float(p.get("resonance", 0.4)))


def _filter_curve(x, sr, cut, res):
    f = pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=1000, resonance=float(np.clip(res, 0, 0.95)))
    out, block = np.empty_like(x), 256
    for s in range(0, x.shape[1], block):
        f.cutoff_hz = float(np.clip(cut[s], 20, 20000))
        out[:, s:s + block] = f(x[:, s:s + block], sr, reset=False)
    return out


def fx_saturate(x, sr, p, ctx):
    g = 10 ** (float(p.get("drive_db", 6)) / 20)
    wet = np.tanh(x * g) / np.tanh(g) * (1 / (1 + 0.15 * np.log10(g + 1)))
    return _mix(x, wet.astype(np.float32), float(p.get("mix", 1.0)))


def fx_bitcrush(x, sr, p, ctx):
    return _board(x, sr, pb.Bitcrush(bit_depth=float(p.get("bits", 8))))


def fx_compress(x, sr, p, ctx):
    return _board(x, sr, pb.Compressor(threshold_db=float(p.get("threshold_db", -18)), ratio=float(p.get("ratio", 3)),
                                       attack_ms=float(p.get("attack_ms", 8)), release_ms=float(p.get("release_ms", 120))),
                  pb.Gain(gain_db=float(p.get("makeup_db", 0))))


# ---------------------------------------------------------------- time, space

def fx_reverb(x, sr, p, ctx):
    send = auto(p.get("send", 1.0), x.shape[1], ctx, 1.0)   # automate the send for reverb throws
    wet = _board(x * send if _is_curve(send) else x * send, sr, pb.HighpassFilter(float(p.get("hp", 250))),
                 pb.Reverb(room_size=float(np.clip(p.get("size", 0.6), 0, 1)), damping=float(p.get("damping", 0.5)),
                           wet_level=1.0, dry_level=0.0, width=1.0))
    mix = auto(p.get("mix", 0.25), x.shape[1], ctx, 0.25)
    return (x + wet * mix).astype(np.float32) if p.get("add", True) else _mix(x, wet, mix)


def fx_delay(x, sr, p, ctx):
    """Tempo echo with darkening repeats; automate `send` for delay throws on single words."""
    d = max(1, int(float(p.get("time", 0.75)) * _beat(ctx) * sr))
    fb = float(np.clip(p.get("feedback", 0.4), 0, 0.9))
    send = auto(p.get("send", 1.0), x.shape[1], ctx, 1.0)
    src = (x * send).astype(np.float32)
    src = _board(src, sr, pb.HighpassFilter(float(p.get("hp", 300))))
    wet, tap, k = np.zeros_like(x), src, 1
    while fb ** (k - 1) > 0.02 and k * d < x.shape[1]:
        tap = _board(tap, sr, pb.LowpassFilter(float(p.get("lp", 5000))))
        seg = tap[:, :x.shape[1] - k * d] * fb ** (k - 1)
        if p.get("ping_pong") and k % 2:
            seg = np.stack([seg.mean(0) * 1.4, seg.mean(0) * 0.2])
        elif p.get("ping_pong"):
            seg = np.stack([seg.mean(0) * 0.2, seg.mean(0) * 1.4])
        wet[:, k * d:] += seg
        k += 1
    return (x + wet * auto(p.get("mix", 0.3), x.shape[1], ctx, 0.3)).astype(np.float32)


def fx_reverse_reverb(x, sr, p, ctx):
    """The swell that sucks into a word: reverb on the reversed sound, reversed back."""
    wet = _board(x[:, ::-1].copy(), sr, pb.Reverb(room_size=float(p.get("size", 0.8)), wet_level=1.0, dry_level=0.0))[:, ::-1]
    return (x + wet * float(p.get("mix", 0.5))).astype(np.float32)


def fx_chorus(x, sr, p, ctx):
    return _board(x, sr, pb.Chorus(rate_hz=float(p.get("rate_hz", 0.8)), depth=float(p.get("depth", 0.3)),
                                   mix=float(p.get("mix", 0.5))))


def fx_phaser(x, sr, p, ctx):
    return _board(x, sr, pb.Phaser(rate_hz=float(p.get("rate_hz", 0.4)), depth=float(p.get("depth", 0.6)),
                                   feedback=float(p.get("feedback", 0.3)), mix=float(p.get("mix", 0.5))))


def fx_tremolo(x, sr, p, ctx):
    t = np.arange(x.shape[1]) / sr
    ph = (t / (float(p.get("rate", 0.5)) * _beat(ctx))) % 1.0
    shape = p.get("shape", "sine")
    lfo = {"square": (ph < 0.5).astype(np.float32), "saw": 1 - ph}.get(shape, 0.5 + 0.5 * np.cos(2 * np.pi * ph))
    return (x * (1 - float(p.get("depth", 0.8)) * (1 - lfo))).astype(np.float32)


def fx_gate(x, sr, p, ctx):
    """Trance gate: a 16th-note on/off pattern, looped."""
    pat = [c in "xX1" for c in str(p.get("pattern", "x.x.x.x.x.x.x.x."))] or [True]
    step = _beat(ctx) / 4 * sr
    idx = (np.arange(x.shape[1]) / step).astype(int) % len(pat)
    g = np.asarray(pat, dtype=np.float32)[idx]
    k = max(2, int(0.004 * sr))
    g = np.convolve(g, np.ones(k) / k, mode="same")
    return (x * g).astype(np.float32)


def fx_pump(x, sr, p, ctx):
    """Sidechain-style ducking on every beat, without needing a kick."""
    t = np.arange(x.shape[1]) / sr + ctx.get("pos0", 0.0) * 4 * _beat(ctx) * (ctx.get("unit") != "frac")
    ph = (t / (float(p.get("rate", 1)) * _beat(ctx))) % 1.0
    duck = 10 ** (-float(p.get("depth_db", 8)) * np.exp(-ph * 5) / 20)
    return (x * duck).astype(np.float32)


def fx_pitch(x, sr, p, ctx):
    semis = float(p.get("semis", 0))
    if abs(semis) < 1e-3:
        return x
    return pb.time_stretch(x, sr, stretch_factor=1.0, pitch_shift_in_semitones=semis, high_quality=True,
                           preserve_formants=bool(p.get("formants", True))).astype(np.float32)[:, :x.shape[1]]


def fx_octave_up(x, sr, p, ctx):
    return _mix(x, fx_pitch(x, sr, {"semis": 12, "formants": False}, ctx), float(p.get("mix", 0.4)))


def fx_octave_down(x, sr, p, ctx):
    return _mix(x, fx_pitch(x, sr, {"semis": -12, "formants": True}, ctx), float(p.get("mix", 0.4)))


def fx_reverse(x, sr, p, ctx):
    return x[:, ::-1].copy()


def fx_stutter(x, sr, p, ctx):
    """Repeat the first slice ("I-I-I-I"), for `length` beats (default: the whole clip)."""
    sl = max(64, int(float(p.get("slice", 0.25)) * _beat(ctx) * sr))
    n = min(x.shape[1], int(float(p.get("length", 1e9)) * _beat(ctx) * sr))
    piece = x[:, :sl].copy()
    k = min(sl // 4, int(0.004 * sr))
    if k > 1:
        piece[:, :k] *= np.linspace(0, 1, k, dtype=np.float32)
        piece[:, -k:] *= np.linspace(1, 0, k, dtype=np.float32)
    reps = int(np.ceil(n / sl))
    y = x.copy()
    y[:, :n] = np.tile(piece, reps)[:, :n]
    return y


def fx_tape_stop(x, sr, p, ctx):
    """The last `beats` slow to a halt."""
    n = min(x.shape[1], int(float(p.get("beats", 1)) * _beat(ctx) * sr))
    if n < 64:
        return x
    seg = x[:, -n:]
    speed = np.linspace(1, 0, n) ** 0.7
    pos = np.cumsum(speed)
    pos = pos[pos < n - 1]
    slowed = np.stack([np.interp(pos, np.arange(n), ch) for ch in seg]).astype(np.float32)
    y = x.copy()
    y[:, -n:] = 0
    m = min(n, slowed.shape[1])
    y[:, -n:-n + m if m < n else None] = slowed[:, :m] * np.linspace(1, 0.2, m, dtype=np.float32)
    return y


def fx_vocoder(x, sr, p, ctx):
    """Robot voice: the vocal's moving tone shapes a synth chord. notes=[midi...] or per_bar=[[...], ...]."""
    import librosa
    n = x.shape[1]
    bar = int(4 * _beat(ctx) * sr)
    per_bar = p.get("per_bar") or [p.get("notes") or [45, 57, 60, 64]]
    t = np.arange(n) / sr
    car = np.zeros(n, dtype=np.float32)
    for b in range(int(np.ceil(n / bar))):
        s, e = b * bar, min(n, (b + 1) * bar)
        for m in per_bar[b % len(per_bar)]:
            f = 440 * 2 ** ((float(m) - 69) / 12)
            for det in (0.997, 1.003):  # two detuned saws per note
                car[s:e] += (2 * ((t[s:e] * f * det) % 1.0) - 1).astype(np.float32)
    car /= max(1e-6, np.max(np.abs(car)))
    nfft, hop = 2048, 512
    M = np.abs(librosa.stft(x.mean(0), n_fft=nfft, hop_length=hop))
    C = librosa.stft(car, n_fft=nfft, hop_length=hop)
    bands = int(p.get("bands", 28))
    k = max(3, (nfft // 2) // bands) | 1
    from scipy.ndimage import uniform_filter1d
    smooth = lambda A: uniform_filter1d(A, size=k, axis=0, mode="nearest")
    env_m, env_c = smooth(M), smooth(np.abs(C)) + 1e-6
    y = librosa.istft(C * (env_m / env_c), hop_length=hop, length=n)
    y = y / max(1e-6, np.max(np.abs(y))) * max(1e-6, float(np.max(np.abs(x))))
    hiss = _board(x, sr, pb.HighpassFilter(5000)) * 0.6   # keep consonants so the words stay clear
    wet = np.stack([y, y]).astype(np.float32) + hiss
    return _mix(x, wet, float(p.get("mix", 1.0)))


FX = {name[3:]: fn for name, fn in list(globals().items()) if name.startswith("fx_")}


def apply_fx(x: np.ndarray, sr: int, chain: list[dict] | None, ctx: dict, warnings: list | None = None) -> np.ndarray:
    for f in chain or []:
        fn = FX.get(str(f.get("type", "")).lower()) if isinstance(f, dict) else None
        if fn is None:
            if warnings is not None:
                warnings.append(f"unknown effect {f!r}")
            continue
        try:
            y = fn(x, sr, f, ctx)
            if y.shape == x.shape and np.isfinite(y).all():
                x = y.astype(np.float32)
            elif warnings is not None:
                warnings.append(f"{f.get('type')} gave bad audio, skipped")
        except Exception as e:  # one bad parameter must not sink the song
            if warnings is not None:
                warnings.append(f"{f.get('type')} failed: {e}")
    return x


# ---------------------------------------------------------------- synth sources (transitions)

def make_synth(kind: str, bars: float, sr: int, bpm: float, seed: int = 0) -> np.ndarray:
    """Transition sounds, built from scratch, `bars` long, ending (riser) or starting (the rest) on the bar."""
    rng = np.random.default_rng(seed)
    n = max(256, int(bars * 4 * 60.0 / bpm * sr))
    t = np.arange(n) / sr
    u = t / t[-1]
    noise = rng.standard_normal((2, n)).astype(np.float32) * 0.3
    ctx = {"bpm": bpm, "unit": "frac"}
    if kind in ("riser", "noise_sweep"):
        y = fx_filter(noise, sr, {"mode": "bp" if kind == "riser" else "hp", "cutoff": [[0, 300], [1, 9000]],
                                  "resonance": 0.5}, ctx)
        if kind == "riser":
            f = 110 * 2 ** (u * 3)
            tone = np.sin(2 * np.pi * np.cumsum(f) / sr).astype(np.float32) * 0.15
            y = y + np.stack([tone, tone])
        y = y * (u ** 2)[None]
    elif kind == "downlifter":
        y = fx_filter(noise, sr, {"mode": "lp", "cutoff": [[0, 9000], [1, 150]], "resonance": 0.4}, ctx) * ((1 - u) ** 1.5)[None]
    elif kind == "impact":
        boom = np.sin(2 * np.pi * np.cumsum(40 + 60 * np.exp(-t * 8)) / sr) * np.exp(-t * 2.5)
        y = np.stack([boom, boom]).astype(np.float32) * 0.9 + noise * np.exp(-t * 6)[None]
        y = fx_reverb(y.astype(np.float32), sr, {"size": 0.9, "mix": 0.5}, ctx)
    elif kind == "sub_drop":
        s = np.sin(2 * np.pi * np.cumsum(120 * np.exp(-t * 3) + 30) / sr) * np.exp(-t * 1.5)
        y = np.stack([s, s]) * 0.9
    elif kind == "snare_roll":  # hits that double in speed towards the end: the build
        y = np.zeros((2, n), dtype=np.float32)
        beat = 60.0 / bpm
        pos, total = 0.0, bars * 4
        while pos < total:
            frac = pos / total
            step = 1.0 if frac < 0.5 else 0.5 if frac < 0.75 else 0.25 if frac < 0.9 else 0.125
            s = int(pos * beat * sr)
            L = min(n - s, int(0.12 * sr))
            if L > 0:
                tt = np.arange(L) / sr
                hit = (rng.standard_normal(L) * np.exp(-tt * 35) + 0.5 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 50))
                y[:, s:s + L] += (hit * (0.35 + 0.65 * frac)).astype(np.float32)
            pos += step
        y = fx_eq(y, sr, {"hp": 180}, ctx)
    else:
        raise ValueError(f"unknown synth {kind}")
    y = np.asarray(y, dtype=np.float32)
    return (y / max(1e-6, float(np.max(np.abs(y)))) * 0.7).astype(np.float32)


SYNTHS = ("riser", "downlifter", "impact", "sub_drop", "noise_sweep", "snare_roll")
