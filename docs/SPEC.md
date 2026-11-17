# GLORE — locked spec

*Glór* (Irish): voice. A local Windows app that builds tracks around **your own** vocals and samples, running on an RTX 4090.

## Core principles
- **The vocal is never regenerated.** AI only makes the instrumental; the original vocal is stretched/processed and mixed on top.
- **Step-by-step flow** — approve each stage, jump back to any step without losing work.
- **Everything local.** Nothing leaves the machine.
- **Monitoring mode changes what you hear, never what you export.**

## Flow
Import → Analyse → Vocal Lock → Arrange → Generate → Chops → Vocal FX → Mix → Export

| # | Step | What it does |
|---|------|--------------|
| 1 | **Import** | Drag & drop WAV/MP3/FLAC/AIFF/OGG. Auto-tags each as Vocal / Full song / Beat; user confirms. Full songs → "Extract vocal" (Demucs HTDemucs FT). |
| 2 | **Analyse** | BPM, key (+Camelot), beat grid, bar starts, vocal phrases. All overridable. Extracted vocals inherit tempo/key from the full mix. |
| 3 | **Vocal Lock** | Time-stretch vocal to project BPM, pitch unchanged (Rubber Band, formant-preserving). Optional ±12 semitone shift, off by default. Project BPM follows the vocal's natural tempo until changed. |
| 4 | **Arrange** | Auto-places vocal by analysis (hook vs verse from repetition/energy) into the genre pack's template. Blocks are draggable/cuttable/duplicable. |
| 5 | **Generate** | ACE-Step 1.5 instrumental in vocal's key + BPM, following the arrangement. Style prompt + genre LoRA. 2–4 variations; regenerate single sections. Beat can be used as a **reference**. |
| 6 | **Chops** | Auto-slice phrases at transients; per-slice pitch/reverse/stutter/gate/filter; auto drop patterns, reshufflable. **Select-to-loop**: drag a region of drums → snaps to bars, rounds to 1/2/4/8 bars, crossfades the seam, stretches to project BPM, preview before commit. |
| 7 | **Vocal FX** | EQ, compression, de-ess, reverb, delay (pedalboard). Presets: Clean / Airy / Club + sliders. |
| 8 | **Mix** | Kick sidechain, vocal pocket EQ carve on the instrumental, master limiter. |
| 9 | **Export** | WAV 24-bit, MP3 320 kbps, MIDI from melodic stems (basic-pitch), stems ZIP for Ableton. Final instrumental split into drums / bass / other. |

## Always available
- **Player:** synced multi-layer playback, mute/solo/volume per layer, click-to-seek waveforms, Space = play/pause.
- **Monitor mode:** Headphones (flat) / Sound System (low-end lift, tamed presence, safety compressor). Playback only.
- **Genre Packs:** LoRA + arrangement template + BPM range + drum/chop style + mix preset, swappable.
  - *Emotive House* — UK garage/house drums, pitched/filtered vocal chops, emotional chords, ~128–134 BPM. Intro → Build → Drop → Break → Drop → Outro.
  - *Melodic Rap* — 808s, ambient pads, rolling hats, half-time feel, ~140–160 BPM. Intro → Hook → Verse → Hook → Verse → Hook → Outro.
  - Each LoRA needs ~30–100 reference tracks in that style.

## Architecture
- `app/` — Electron desktop shell + UI (plain HTML/CSS/JS; theme in CSS variables).
- `engine/` — Python engine (FastAPI on 127.0.0.1), started/stopped by the app.
- Projects saved in `%APPDATA%\GLORE\projects\`.

## Build log
- **Session 1** (2026-11-17): Engine, desktop app, Import, Analyse, Extract vocal, Vocal Lock, player, monitor modes, Windows setup.
- Next: Arrange → Chops (incl. select-to-loop) → Vocal FX → Mix → Export → Generate (ACE-Step) → LoRA training → installer (.exe).
