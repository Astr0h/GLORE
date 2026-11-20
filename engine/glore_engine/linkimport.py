"""Import audio from a link.

- YouTube / SoundCloud: downloaded directly with yt-dlp.
- Spotify / Apple Music: their audio is DRM-protected, so GLORE reads the
  track's title + artist from the page and finds the same song on YouTube.

ffmpeg comes bundled via imageio-ffmpeg, so nothing extra to install."""

from __future__ import annotations

import html
import json
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


class LinkError(RuntimeError):
    pass


@dataclass
class LinkInfo:
    service: str            # youtube | soundcloud | spotify | apple
    title: str = ""
    artist: str = ""
    matched: str = ""       # what was actually downloaded (for Spotify/Apple matches)


def detect_service(url: str) -> str:
    host = (urlparse(url.strip()).hostname or "").lower()
    if host.endswith("youtube.com") or host.endswith("youtu.be") or host == "music.youtube.com":
        return "youtube"
    if host.endswith("soundcloud.com"):
        return "soundcloud"
    if host.endswith("spotify.com"):
        return "spotify"
    if host.endswith("music.apple.com"):
        return "apple"
    raise LinkError("Paste a YouTube, SoundCloud, Spotify or Apple Music link")


# ---------------------------------------------------------------- metadata (Spotify / Apple)

def _meta(page: str, prop: str) -> str:
    m = re.search(rf'<meta[^>]+(?:property|name)="{re.escape(prop)}"[^>]+content="([^"]*)"', page)
    if not m:
        m = re.search(rf'<meta[^>]+content="([^"]*)"[^>]+(?:property|name)="{re.escape(prop)}"', page)
    return html.unescape(m.group(1)).strip() if m else ""


def parse_spotify(page: str) -> tuple[str, str]:
    """og:title = song; og:description = 'Artist · Album · Song · 2024'.
    Falls back to <title>Song - song and lyrics by Artist | Spotify</title>, then to the
    JSON in the embed page ("name" + "artists"), since Spotify serves different pages
    to different clients."""
    title = _meta(page, "og:title")
    desc = _meta(page, "og:description")
    artist = desc.split("·")[0].strip() if "·" in desc else ""
    if not artist:
        artist = _meta(page, "music:musician_description")
    if not title or not artist:
        m = re.search(r"<title>(.+?)\s+-\s+song(?:\s+and\s+lyrics)?\s+by\s+(.+?)\s*\|\s*Spotify</title>", page, re.S)
        if m:
            title = title or html.unescape(m.group(1)).strip()
            artist = artist or html.unescape(m.group(2)).strip()
    if not title:
        m = re.search(r'"type"\s*:\s*"track"[^{}]*?"name"\s*:\s*"([^"]+)"', page) or \
            re.search(r'"name"\s*:\s*"([^"]+)"[^{}]*?"type"\s*:\s*"track"', page)
        if m:
            title = json.loads(f'"{m.group(1)}"')
    if not artist:
        m = re.search(r'"artists"\s*:\s*\[\s*\{[^{}]*?"name"\s*:\s*"([^"]+)"', page)
        if m:
            artist = json.loads(f'"{m.group(1)}"')
    return title, artist


def parse_apple(page: str) -> tuple[str, str]:
    """og:title like 'Song by Artist on Apple Music' or 'Song - Single by Artist on Apple Music'."""
    t = _meta(page, "og:title") or _meta(page, "apple:title")
    t = re.sub(r"\s+on Apple\s+Music$", "", t)
    m = re.match(r"(.+?)\s+by\s+(.+)$", t)
    if not m:
        return t, ""
    title = re.sub(r"\s+-\s+(Single|EP)$", "", m.group(1)).strip()
    return title, m.group(2).strip()


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


def lookup_track(url: str, service: str) -> tuple[str, str]:
    page = _fetch(url)
    title, artist = (parse_spotify if service == "spotify" else parse_apple)(page)
    if service == "spotify" and not (title and artist):
        m = re.search(r"/track/([A-Za-z0-9]+)", url)
        if m:  # the embed page carries the track JSON even when the main page is a bare app shell
            try:
                t2, a2 = parse_spotify(_fetch(f"https://open.spotify.com/embed/track/{m.group(1)}"))
                title, artist = title or t2, artist or a2
            except Exception:
                pass
    if not title:
        raise LinkError("Couldn't read the track name from that link (is it a single track, not a playlist?)")
    return title, artist


# ---------------------------------------------------------------- download

def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise LinkError("ffmpeg missing - run scripts\\setup.ps1 again") from e


def _download(target: str, out_dir: Path, report: Callable[[float, str], None]) -> tuple[Path, dict]:
    try:
        import yt_dlp
    except ImportError as e:
        raise LinkError("yt-dlp isn't installed - run scripts\\setup.ps1 again") from e

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                report(0.15 + 0.6 * d.get("downloaded_bytes", 0) / total, "Downloading")
        elif d.get("status") == "finished":
            report(0.8, "Converting")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg_path(),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(target, download=True)
    if info and info.get("entries"):
        info = info["entries"][0]
    wavs = sorted(out_dir.glob("*.wav"))
    if not wavs:
        raise LinkError("Download finished but no audio came out")
    return wavs[0], info or {}


def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip()
    return s[:80] or "Imported"


def friendly_error(e: Exception, service: str) -> str:
    import urllib.error
    name = {"youtube": "YouTube", "soundcloud": "SoundCloud", "spotify": "Spotify", "apple": "Apple Music"}[service]
    msg = str(e)
    if isinstance(e, urllib.error.HTTPError) and e.code == 404:
        return f"{name} says that track doesn't exist - check the link"
    if isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError)) or "Tunnel connection" in msg:
        return f"Couldn't reach {name} - check your internet connection"
    if "DRM" in msg:
        return (f"{name} has this track copy-protected (DRM, common on label releases), so it can't be "
                "downloaded from anywhere. Upload the file instead")
    if "Private video" in msg or "Sign in" in msg:
        return "That video is private or needs a login"
    if "not available" in msg.lower() or "unavailable" in msg.lower():
        return "That track isn't available (removed or region-locked)"
    return f"Couldn't get audio from that link ({msg.splitlines()[0][:120]})"


def fetch_link(url: str, progress: Callable[[float, str], None] | None = None) -> tuple[Path, LinkInfo, tempfile.TemporaryDirectory]:
    """Returns (wav path, info, temp dir handle - caller cleans up)."""
    report = progress or (lambda p, m: None)
    service = detect_service(url)
    info = LinkInfo(service=service)
    tmp = tempfile.TemporaryDirectory(prefix="glore_link_")
    out_dir = Path(tmp.name)
    try:
        if service in ("spotify", "apple"):
            report(0.05, f"Reading {service.title()} track")
            info.title, info.artist = lookup_track(url, service)
            query = f"{info.artist} - {info.title}" if info.artist else info.title
            report(0.1, f"Finding “{query}” on YouTube")
            wav, yinfo = _download(f"ytsearch1:{query} official audio", out_dir, report)
            info.matched = yinfo.get("title", "")
        else:
            report(0.05, "Connecting")
            wav, yinfo = _download(url, out_dir, report)
            info.title = yinfo.get("track") or yinfo.get("title", "")
            info.artist = yinfo.get("artist") or yinfo.get("uploader", "")
            info.matched = yinfo.get("title", "")
    except LinkError:
        tmp.cleanup()
        raise
    except Exception as e:
        tmp.cleanup()
        raise LinkError(friendly_error(e, service)) from e
    return wav, info, tmp
