# GLORE

*Glór* — voice. Build house and melodic rap tracks around your own vocals and samples, locally on your GPU.

Full feature spec: [docs/SPEC.md](docs/SPEC.md)

## First-time setup (Windows)

You need **Python 3.12**, **Node.js LTS**, **Git**, and an up-to-date **NVIDIA driver**.

1. Clone the repo:
   ```
   git clone https://github.com/Astr0h/GLORE.git
   cd GLORE
   ```
2. Double-click **`Setup GLORE.bat`** in the GLORE folder. It takes a while the first time, since it downloads ~3 GB of PyTorch and models. It ends with a self-test and waits for you to press Enter. Everything it prints is also saved to `setup.log`; if anything goes red, send that file to Claude.
3. Launch: double-click **`Start GLORE.bat`**. If setup hasn't finished, it runs setup for you first.

## Updating after a new build session
```
git pull
```
Then double-click `Setup GLORE.bat` again. It skips what's already installed, so re-running it is quick.

## What works right now (v0.1)
- **Import:** drag files anywhere onto the window. GLORE auto-detects Vocal / Full song / Beat.
- **Extract vocal:** on any full song (GPU, roughly 1–2 min per track on an RTX 4060).
- **Analyse:** BPM, key, Camelot, beat grid, vocal phrases, with ½× / 2× fixes and manual overrides.
- **Vocal Lock:** stretches the vocal to your BPM without changing pitch, with an optional pitch shift and an A/B against the original.
- **Player:** every layer in sync, M / S / volume, click a waveform to seek, Space to play/pause.
- **Headphones / Sound System:** monitoring only. Exports are never affected.

## New in session 2
- **Link import:** paste a YouTube, SoundCloud, Spotify or Apple Music link under the drop zone. Spotify and Apple Music are matched on YouTube.
- **Raw vocal:** on any vocal, choose what to strip (backing vocals, echo, reverb, noise), then click *Make raw*. The first use of each option downloads its model (about 0.1–0.9 GB each).
- **Training tags:** *+ Train* on any sample adds it to a genre pack (House, R&B / Rap, Gospel / Soul) with mood tags.
- **Import list:** on the LoRA Packs page. In the track-list sheet, download it as CSV, then click *Import list* on the pack. GLORE finds and downloads every ticked track and tags it with its Tag. Tracks it can't find are listed under the pack. Running it again skips what's already there.
- **LoRA Packs page** (bottom of the sidebar): track counts per pack, plus *Prepare*, which strips vocals ready for training. *Train LoRA* arrives with Generate.

## If something goes wrong
- **Stuck on "Starting engine…":** click *Open engine log* and send Claude what's in it.
- **"CPU only — CUDA not found"** in the top right: update your NVIDIA driver and re-run setup.
- Projects live in `%APPDATA%\GLORE\projects\`.

## Layout
```
app/       Electron desktop app (UI)   - theme colours live at the top of app/renderer/styles.css
engine/    Python audio engine          - analysis, separation, time-stretch
scripts/   setup.ps1
docs/      SPEC.md
```
