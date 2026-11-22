"""Song structure: give the beat real sections before ACE builds it.

A vocal on its own is one long section, so the beat ACE builds around it is one long section too: no
intro, no drop, no room for chops. Records are arranged. Here we plan that arrangement around the vocal
and open up instrumental space in its timeline - whole bars, at quiet moments on the bar grid, so the
vocal itself is never cut or moved off the beat:

    intro      8 bars before the vocal comes in (less if the song already has one)
    drop       after a chorus: 8 bars (house) / 4 (R&B, gospel) where the beat and the chops take over
    outro      8 bars after the last line

ACE then builds the beat over the whole arranged timeline, so those parts come out as real instrumental
sections that follow on from the vocal around them.

    plan = plan_structure(vocal, sr, bpm, offset, chorus_bars, style="house")
    arranged = apply_plan(vocal, plan)          # the same plan works on any aligned signal
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

DROP_BARS = {"house": 8, "rnb": 4, "gospel": 4}
MAX_SECONDS = 300.0


@dataclass
class Insert:
    at: int        # sample in the ORIGINAL timeline where silence goes in
    bars: int
    kind: str      # intro | drop | outro


@dataclass
class Plan:
    sr: int
    bpm: float
    offset: float
    inserts: list[Insert] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)  # the arranged song, for captions and the app

    @property
    def bar_samples(self) -> int:
        return int(round(4 * 60.0 / self.bpm * self.sr))

    def to_dict(self) -> dict:
        return asdict(self)


def _quiet_at(mono: np.ndarray, sr: int, t: int, win: float = 0.2) -> float:
    """Level just before t: a line that has finished (the next one may start right on the bar line)."""
    seg = mono[max(0, t - int(win * sr)):t + int(0.005 * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def plan_structure(vocal: np.ndarray, sr: int, bpm: float, offset: float, chorus_bars: list[tuple[int, int]],
                   style: str = "house", intro_bars: int = 8, outro_bars: int = 8, max_drops: int = 2,
                   max_seconds: float = MAX_SECONDS) -> Plan:
    mono = vocal.mean(0) if vocal.ndim == 2 else vocal
    plan = Plan(sr, bpm, offset)
    bar = plan.bar_samples
    off = int(round(offset * sr))
    n_bars = int(np.ceil((len(mono) - off) / bar))
    loud = np.percentile(np.abs(mono), 95) + 1e-9
    act = np.array([np.mean(np.abs(mono[off + b * bar:off + (b + 1) * bar]) > 0.08 * loud) if off + b * bar < len(mono)
                    else 0.0 for b in range(n_bars)])
    sung = act > 0.08
    if not sung.any():
        return plan
    first, last = int(np.argmax(sung)), int(n_bars - 1 - np.argmax(sung[::-1]))
    total = len(mono) / sr
    level = float(np.sqrt(np.mean(mono ** 2))) + 1e-6

    def at_bar(b: int) -> int:
        return min(len(mono), max(0, off + b * bar))

    def quiet_boundary(b: int) -> int | None:
        """The nearest bar line to bar b (within 2 bars) where the vocal is quiet."""
        thr = 0.15 * level
        for d in (0, 1, -1, 2, -2):
            c = b + d
            if first < c <= last and _quiet_at(mono, sr, at_bar(c)) < thr:
                return c
        return None

    inserts: list[Insert] = []
    if first < intro_bars:                         # an intro, unless the song already has one
        inserts.append(Insert(at_bar(first) if first else 0, intro_bars - first, "intro"))
    drops = 0
    for a, e in sorted(chorus_bars):
        if drops >= max_drops or e > last:
            continue
        if e + 2 <= n_bars and not sung[e:e + 2].any():
            continue                               # already an instrumental bit after this chorus
        c = quiet_boundary(e)
        if c is not None and all(abs(c - (i.at - off) / bar) >= 4 for i in inserts):
            inserts.append(Insert(at_bar(c), DROP_BARS.get(style, 8), "drop"))
            drops += 1
    tail = n_bars - 1 - last
    if tail < outro_bars:
        inserts.append(Insert(len(mono), outro_bars - tail, "outro"))

    # keep it a sensible length: drop the extra parts from the end until it fits
    def length(ins):
        return total + sum(i.bars for i in ins) * bar / sr
    while inserts and length(inserts) > max_seconds:
        for kind in ("outro", "drop", "intro"):
            cand = [i for i in inserts if i.kind == kind]
            if cand:
                inserts.remove(cand[-1])
                break
    plan.inserts = sorted(inserts, key=lambda i: i.at)

    # the arranged song as sections (for the caption and the app)
    sec, pos = [], 0.0
    cursor = 0
    for i in plan.inserts + [Insert(len(mono), 0, "end")]:
        if i.at > cursor:
            sec.append({"part": "vocal", "start": round(pos, 2), "end": round(pos + (i.at - cursor) / sr, 2)})
            pos += (i.at - cursor) / sr
            cursor = i.at
        if i.bars:
            sec.append({"part": i.kind, "start": round(pos, 2), "end": round(pos + i.bars * bar / sr, 2), "bars": i.bars})
            pos += i.bars * bar / sr
    plan.sections = sec
    return plan


def apply_plan(x: np.ndarray, plan: Plan) -> np.ndarray:
    """Insert the plan's silent bars into any signal aligned with the vocal ((ch, n) or (n,))."""
    two = x.ndim == 2
    x2 = x if two else x[None]
    parts, cursor = [], 0
    for i in plan.inserts:
        at = min(i.at, x2.shape[1])
        parts.append(x2[:, cursor:at])
        parts.append(np.zeros((x2.shape[0], i.bars * plan.bar_samples), dtype=x2.dtype))
        cursor = at
    parts.append(x2[:, cursor:])
    out = np.concatenate(parts, 1)
    return out if two else out[0]


def caption_hint(plan: Plan) -> str:
    kinds = [s["part"] for s in plan.sections if s["part"] != "vocal"]
    bits = []
    if "intro" in kinds:
        bits.append("a building instrumental intro")
    if "drop" in kinds:
        bits.append("big instrumental drops after the chorus")
    if "outro" in kinds:
        bits.append("an outro")
    return ("arranged like a finished record, with " + ", ".join(bits)) if bits else ""
