"""Batch-import a track list (CSV exported from the GLORE track-list sheet)
straight into a genre pack's training set.

For every row with Keep ticked: find "artist - track" on YouTube, download it,
and add it to the pack with its Tag as the mood. Rows already in the pack are
skipped, so the same list can be run again after adding tracks."""

from __future__ import annotations

import csv
import io
import re
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

from . import datasets, linkimport

TRUE_WORDS = {"true", "yes", "y", "1", "x", "✓", "☑"}


@dataclass
class Row:
    track: str
    artist: str
    tag: str = ""


@dataclass
class ImportReport:
    added: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_csv(text: str) -> list[Row]:
    """Find the header row (it has Track and Artist), then read the ticked rows.
    Rows above the header (e.g. the sheet's summary counts) are ignored.
    With no Keep column, every row counts as kept."""
    text = text.lstrip("﻿")
    rows = list(csv.reader(io.StringIO(text)))
    header_i, cols = None, {}
    for i, r in enumerate(rows):
        lower = [c.strip().lower() for c in r]
        if "track" in lower and "artist" in lower:
            header_i = i
            cols = {name: lower.index(name) for name in ("track", "artist", "tag", "keep") if name in lower}
            break
    if header_i is None:
        raise ValueError("Couldn't find a header row with 'Track' and 'Artist' columns")

    def cell(r, name):
        j = cols.get(name)
        return r[j].strip() if j is not None and j < len(r) else ""

    out = []
    for r in rows[header_i + 1:]:
        track, artist = cell(r, "track"), cell(r, "artist")
        if not track:
            continue
        if "keep" in cols and cell(r, "keep").lower() not in TRUE_WORDS:
            continue
        out.append(Row(track=track, artist=artist, tag=cell(r, "tag")))
    return out


def display_name(row: Row) -> str:
    return linkimport.safe_name(f"{row.artist} - {row.track}" if row.artist else row.track)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def search_query(row: Row) -> str:
    # First credited artist matches best; long feature lists confuse search.
    main_artist = re.split(r",|&| x | feat\.? | ft\.? ", row.artist, maxsplit=1)[0].strip()
    return f"{main_artist} - {row.track} official audio" if main_artist else f"{row.track} official audio"


def import_rows(pack: str, rows: list[Row], progress: Callable[[float, str], None] | None = None,
                download: Callable | None = None) -> ImportReport:
    """download(query, out_dir, report) -> (wav_path, info dict); injectable for tests."""
    report_fn = progress or (lambda p, m: None)
    download = download or (lambda q, d, r: linkimport._download(f"ytsearch1:{q}", d, r))
    have = {_norm(it.get("name", "")) for it in datasets.list_items(pack)}
    rep = ImportReport()
    for i, row in enumerate(rows):
        name = display_name(row)
        base = i / max(len(rows), 1)
        report_fn(base, f"{i + 1}/{len(rows)}: {row.track}")
        if _norm(name) in have:
            rep.skipped.append({"name": name, "reason": "already in pack"})
            continue
        query = search_query(row)
        try:
            with tempfile.TemporaryDirectory(prefix="glore_list_") as tmp:
                wav, info = download(query, Path(tmp), lambda p, m: report_fn(base + p / max(len(rows), 1), m))
                moods = [row.tag.lower()] if row.tag else []
                datasets.add_or_update(pack, Path(wav), name, moods, source_ref=f"list:{_norm(name)}")
            have.add(_norm(name))
            rep.added.append({"name": name, "matched": (info or {}).get("title", "")})
        except Exception as e:
            msg = linkimport.friendly_error(e, "youtube") if not isinstance(e, linkimport.LinkError) else str(e)
            rep.failed.append({"name": name, "reason": msg})
    report_fn(1.0, f"Done: {len(rep.added)} added, {len(rep.skipped)} skipped, {len(rep.failed)} failed")
    return rep
