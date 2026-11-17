"""Track-list CSV import into a pack (downloads faked)."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glore_engine import datasets, tracklist
from glore_engine.project import ProjectStore
from glore_engine.server import create_app
from test_engine import make_beat

SHEET_CSV = """Kept,Soft,DNB,Heavy,Hype,,
95,40,6,19,30,,
,,,,,,
Track,Artist,Tag,Keep,My confidence,Release,Notes
Kyle (i found you),Fred again..,Soft,TRUE,Sure,Single 2019,"Piano, voice notes"
Rumble,"Fred again.., Skrillex, Flowdan",Heavy,TRUE,Sure,USB,
Glue,Bicep,Soft,FALSE,Sure,Other artist,
trackstar,jigitz,Hype,TRUE,Listen first,50 ballerinas (2026),
"""


def test_parse_skips_summary_and_unticked():
    rows = tracklist.parse_csv(SHEET_CSV)
    assert [r.track for r in rows] == ["Kyle (i found you)", "Rumble", "trackstar"]
    assert rows[1].artist == "Fred again.., Skrillex, Flowdan" and rows[1].tag == "Heavy"


def test_parse_without_keep_column_keeps_all():
    rows = tracklist.parse_csv("Track,Artist\nA,B\nC,D\n")
    assert len(rows) == 2


def test_parse_needs_header():
    with pytest.raises(ValueError):
        tracklist.parse_csv("foo,bar\n1,2\n")


def test_search_query_uses_first_artist():
    r = tracklist.Row("Rumble", "Fred again.., Skrillex, Flowdan")
    assert tracklist.search_query(r) == "Fred again.. - Rumble official audio"


def test_import_rows_adds_skips_and_reports_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("GLORE_HOME", str(tmp_path / "home"))
    src = tmp_path / "beat.wav"
    make_beat(src, bars=2)
    queries = []

    def fake_download(query, out_dir, report):
        queries.append(query)
        if "trackstar" in query:
            raise RuntimeError("ERROR: Video unavailable")
        dst = Path(out_dir) / "a.wav"
        dst.write_bytes(src.read_bytes() + query.encode())  # unique hash per track
        return dst, {"title": query}

    rows = tracklist.parse_csv(SHEET_CSV)
    rep = tracklist.import_rows("house", rows, download=fake_download)
    assert [a["name"] for a in rep.added] == ["Fred again.. - Kyle (i found you)", "Fred again.., Skrillex, Flowdan - Rumble"]
    assert rep.failed and "available" in rep.failed[0]["reason"]
    items = datasets.list_items("house")
    assert {tuple(i["moods"]) for i in items} == {("soft",), ("heavy",)}

    # running the same list again skips what's already there
    rep2 = tracklist.import_rows("house", rows, download=fake_download)
    assert len(rep2.skipped) == 2 and not rep2.added


def test_import_list_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GLORE_HOME", str(tmp_path / "home"))
    c = TestClient(create_app(ProjectStore(tmp_path / "p")))
    assert c.post("/datasets/house/import-list", json={"csv_text": "Track,Artist,Keep\nA,B,FALSE\n"}).status_code == 400
    assert c.post("/datasets/house/import-list", json={"csv_path": str(tmp_path / "nope.csv")}).status_code == 400
    monkeypatch.setattr(tracklist, "import_rows", lambda pack, rows, r: tracklist.ImportReport(added=[{"name": x.track} for x in rows]))
    r = c.post("/datasets/house/import-list", json={"csv_text": SHEET_CSV}).json()
    assert r["tracks"] == 3
    for _ in range(50):
        j = c.get(f"/jobs/{r['job_id']}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done" and len(j["result"]["added"]) == 3
