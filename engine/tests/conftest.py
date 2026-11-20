import pytest


@pytest.fixture(autouse=True)
def _isolated_glore_home(tmp_path, monkeypatch):
    """Never touch the real %APPDATA%/GLORE (cloud keys, ledger) from a test."""
    monkeypatch.setenv("GLORE_HOME", str(tmp_path / "home"))
