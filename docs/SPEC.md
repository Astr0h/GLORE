# GLORE — locked spec

*Glór* (Irish): voice. A local Windows app that builds tracks around **your own** vocals and samples, running locally on an NVIDIA GPU (Seos's PC: RTX 4060, 8 GB), with heavy jobs optionally sent to a rented cloud GPU.

## Core principles
- **The vocal is never regenerated.** AI only makes the instrumental; the original vocal is stretched/processed and mixed on top.
- **Step-by-step flow** — approve each stage, jump back to any step without losing work.
- **Local by default.** Nothing leaves the machine except cloud jobs you start, which go to your own RunPod storage.
- **Monitoring mode changes what you hear, never what you export.**
- **Clean exports.** GLORE writes only its own metadata (title, artist, BPM, key). The local model adds no watermark. GLORE does not strip third-party watermarks.
- **UI: no on-screen explanations.** Anything that needs explaining gets a hover tip. Typed inputs instead of preset buttons; sliders where a range makes sense (e.g. pitch).

## Flow
Import → Analyse → Vocal Lock → Arrange → Generate → Chops → Vocal FX → Mix → Export

| # | Step | What it does |
|---|------|--------------|
| 1 | **Import** | Drag & drop WAV/MP3/FLAC/AIFF/OGG. Auto-tags each as Vocal / Full song / Beat; user confirms. Full songs → "Extract vocal" (Demucs HTDemucs FT). |
| 1b | **Link import** | Paste a YouTube / SoundCloud link → audio pulled with yt-dlp. Spotify / Apple Music links are DRM-protected, so GLORE reads the track's metadata and finds the matching audio on YouTube (the spotDL approach). |
| 1c | **Genre + mood tags → LoRA dataset** | On import, optionally tag a track with a genre pack **and mood/style tags** (e.g. introspective, hype, R&B, gospel). Tags become the training captions, so one LoRA per pack learns every sub-style and the prompt picks which one. Once a set is big enough (~30+), "Train LoRA". Split into a separate LoRA only if a sub-style keeps bleeding into others. |
| 1d | **Raw vocal** | After extraction, an optional GPU clean-up chain: de-reverb → de-echo/delay → lead vs backing/doubles split → de-noise (UVR-style Roformer/MDX models via audio-separator). Result: a dry, raw-sounding lead. Each stage can be toggled. Pitch correction (autotune) can't be truly undone, only softened. |
| 2 | **Analyse** | BPM, key (+Camelot), beat grid, bar starts, vocal phrases. All overridable. Extracted vocals inherit tempo/key from the full mix. |
| 3 | **Vocal Lock** | Time-stretch vocal to project BPM, pitch unchanged (Rubber Band, formant-preserving). Optional ±12 semitone shift, off by default. Project BPM follows the vocal's natural tempo until changed. |
| 4 | **Arrange** | Auto-places vocal by analysis (hook vs verse from repetition/energy) into the genre pack's template. Blocks are draggable/cuttable/duplicable. |
| 5 | **Generate** | ACE-Step 1.5 instrumental in vocal's key + BPM, following the arrangement. Style prompt + genre LoRA. 2–4 variations; regenerate single sections. Beat can be used as a **reference**. |
| 5b | **Combined mode** | Build around a vocal **and** follow a reference song at once: key/BPM/arrangement come from the vocal, sound/style from the reference. Reference-strength slider (too high = copies the reference). |
| 6 | **Chops** | Auto-slice phrases at transients; per-slice pitch/reverse/stutter/gate/filter; auto drop patterns, reshufflable. **Select-to-loop**: drag a region of drums → snaps to bars, rounds to 1/2/4/8 bars, crossfades the seam, stretches to project BPM, preview before commit. |
| 6b | **Chop style per pack** | Each genre pack has a chop style: slice length, density, repetition, grid placement, pitch, reverse/gate, tails. Starting presets: House = short 1/16–1/8 rhythmic stabs, pitched up, repeated; R&B / Rap = half-bar-to-phrase cuts, sparse, often pitched down with tails; Gospel / Soul = long held vowels, stretched and stacked. **Learned:** Prepare keeps each training track's vocal stem; GLORE measures its chops (lengths, repeats, grid positions, pitch shifts) and tunes the pack's style from them. User sliders (length, density, pitch spread) on top. **Reference tracks:** star a training track as a chop reference and its measurements count extra (House: *Cmon (LATIN MAFIA & Fred edit)*, a full song built from vocal chops). |
| 7 | **Vocal FX** | EQ, compression, de-ess, reverb, delay (pedalboard). Presets: Clean / Airy / Club + sliders. |
| 8 | **Mix** | Kick sidechain, vocal pocket EQ carve on the instrumental, master limiter. |
| 9 | **Export** | WAV 24-bit, MP3 320 kbps, MIDI from melodic stems (basic-pitch), stems ZIP for Ableton. Final instrumental split into drums / bass / other. |

## Always available
- **Player:** synced multi-layer playback, mute/solo/volume per layer, click-to-seek waveforms, Space = play/pause.
- **Monitor mode:** Headphones (flat) / Sound System (low-end lift, tamed presence, safety compressor). Playback only.
- **Genre Packs** (one LoRA each; LoRA + arrangement template + BPM range + drum/chop style + mix preset):
  - *House* — hype sets and introspective, beautiful tracks (Fred again.. lane). UK garage/house drums, pitched/filtered vocal chops, emotional chords, ~128–134 BPM.
  - *R&B / Melodic Rap* — Drake-lane: 808s, ambient pads, rolling hats, half-time feel, ~140–160 BPM.
  - *Gospel / Soul* — ethereal, gospel and soul-sample beats (Kanye lane): choirs, chopped soul samples, organ, warm drums.
  - Mood tags inside each pack (e.g. hype vs introspective) steer the prompt.
  - **Training audio per pack:** import full songs; GLORE splits them automatically. House and R&B/Rap train on *instrumental only* (all vocals removed). Gospel/Soul uses *keep backing vocals*: the lead is removed, choirs/hums/ad-lib pads stay, so it can generate those textures but never a lead vocal.
  - Each LoRA needs ~30–100 reference tracks in that style. No Suno (or other AI-service) output in training sets: their terms forbid it.

## Community LoRAs
- Each genre pack can also load downloaded ACE-Step 1.5 LoRAs (e.g. from Hugging Face), not only ones trained in GLORE.
- On import, GLORE checks the base model + variant (standard/XL, base/turbo/sft) and the license, and warns about mismatches.
- A downloaded LoRA can be stacked with your own at adjustable strengths (e.g. 70% yours, 30% community).

## Hardware tiers
- GLORE detects the GPU's VRAM and picks model sizes automatically.
- **≥16 GB (e.g. RTX 4090):** ACE-Step 1.5 XL (4B) available; everything runs at full quality.
- **8–12 GB (e.g. RTX 4060):** standard ACE-Step 1.5 (XL needs ≥12 GB with offload); separation models run with smaller chunks.
- **LoRAs only work on the base model they were trained on**, so genre packs ship trained on the standard model (works on every tier), plus optional XL versions.

## Remote mode (laptop / phone)
- The engine is already a separate HTTP server, so the UI can talk to an engine on another machine.
- **Home PC:** the home PC runs the engine, and the laptop/phone connects over Tailscale (private network, no open ports). Free apart from electricity.
- **Cloud GPU:** the same engine in a Docker image on a rented GPU (e.g. RunPod). Models, LoRAs and projects live on a network volume, so pods can be started and stopped freely.
- **Hybrid (default plan):** the local 4060 does everything that fits (analysis, separation, standard ACE-Step generation). Heavy jobs (LoRA training, XL generation, batch Prepare) go to the cloud GPU.
- **Auto on/off:** GLORE starts the cloud GPU through the RunPod API when a cloud job is queued or "Connect to cloud" is pressed. The pod stops itself after N idle minutes (default 10), with a hard session cap as a safety net. The app shows live cost per session and per month, with an optional monthly spending limit.
- Needed: an engine URL + access token setting, file upload (remote engines can't read local paths), the UI served by the engine for phone browsers, a mobile layout, and the Docker image.

## Cloud LoRA training (built in session 3)
- **Where:** RunPod, REST API v2 (`api.runpod.io/v2`; v1 was retired on 15 Nov 2026). One **network volume** in an S3-API datacenter holds everything; GLORE reads and writes it from the PC through RunPod's S3-compatible API, so uploads and status checks never need a running pod or an open port.
- **Keys:** RunPod API key (pods) + S3 API key (`user_…` id + secret). Stored in `%APPDATA%\GLORE\cloud\settings.json`, masked in the UI.
- **Volume layout** (`/workspace` on a pod = the bucket root over S3): `glore/` (pod scripts, `queue/`, `runs/<job>/status.json` + `log.txt`, `pods/<id>.json` heartbeats), `datasets/<pack>/` (FLAC + `dataset.json`), `tensors/<pack>/<hash>/`, `loras/<pack>/<run>/`, `ace/` (ACE-Step 1.5 + venv + checkpoints, installed on first run).
- **No custom Docker image:** pods use the stock `nvidia/cuda:12.8.1-runtime-ubuntu22.04` image and run `glore/pod_start.sh`, which GLORE uploads before each job. ACE-Step (pinned commit) and its venv live on the volume, so the second run starts in about a minute. Nothing to build or publish; the pod code ships with the app.
- **Trainer:** ACE-Step 1.5's built-in Side-Step CLI, corrected ("fixed") mode, PEFT LoRA. Default: standard 2B **turbo** variant (what runs locally on 8 GB), rank 64 / alpha 128, lr 1e-4, batch 1 × grad-accum 4, checkpoint every 10 epochs. Epochs from track count (≤10: 300, ≤50: 150, 50+: 80), overridable. Captions = pack + moods + BPM + key, trigger word `glore_<pack>`, all tracks marked instrumental.
- **Flow:** prepared WAV → 24-bit FLAC (lossless) → upload (files already there are skipped, so an interrupted upload resumes) → job file in `glore/queue/` → start the cheapest in-stock GPU matching the GPU setting (default A100 80GB, ~$2.19/h) in the volume's datacenter, or reuse a GLORE pod that's idling → follow `status.json` (epoch, loss, ETA, cost) → download `adapter_config.json` + `adapter_model.safetensors` + a `glore_lora.json` manifest (base model, variant, inference shift/steps, trigger) to `%APPDATA%\GLORE\loras\<pack>\<run>\`.
- **Real API notes (checked 18 Nov 2026):** `/catalog/gpus` needs `product=POD` with `include=AVAILABILITY`; per-datacenter stock is `gpu.dataCenters[]`. Only datacenters whose `networkVolumeTypes` include `STANDARD` can hold the volume. Secure H100 SXM was $4.39/h and only in US-GA-2 among S3 datacenters, so the default is an A100 80GB in EUR-IS-1 (Iceland).
- **Run id** = hash of tracks + settings. The same run started again resumes from its last checkpoint; an identical finished run is refused.
- **Stopping:** the pod terminates itself after `idle_minutes` with no job (default 10) and at `hard_cap_hours` (default 4); `pod_start.sh` also terminates on any exit. GLORE watches every pod it started (survives app restarts) and terminates it at the hard cap, at the monthly limit, if its heartbeat stops for 20 min, or on **Stop**. Pods are *terminated*, not stopped, so nothing keeps billing; the data is on the volume.
- **Cost:** live per pod and per month from a local ledger (pod price × time). A run is refused if the worst case (price × hard cap) would pass the monthly limit (default $20). Network volume storage is separate: $0.07/GB/month.
- **Connection test:** writes and reads a file over S3, then runs the cheapest in-stock GPU for ~2 min (hard cap 4 min) to prove pod start, volume mount and self-shutdown.
- **8 GB test (local):** installs ACE-Step in `%APPDATA%\GLORE\ace` with `uv`, trains 3 tracks for 5 min with low-VRAM settings (rank 16, AdamW8bit, encoder offload), reports peak VRAM, s/step and a full-run estimate for the PC.

## Architecture
- `app/` — Electron desktop shell + UI (plain HTML/CSS/JS; theme in CSS variables).
- `engine/` — Python engine (FastAPI on 127.0.0.1), started/stopped by the app.
- Projects saved in `%APPDATA%\GLORE\projects\`.

## Build log
- **Session 1** (2026-11-17): Engine, desktop app, Import, Analyse, Extract vocal, Vocal Lock, player, monitor modes, Windows setup. UI tidy: hover tips, no preset BPM buttons.
- **Session 2** (2026-11-17): Link import, raw vocal chain (backing / echo / reverb / noise), genre + mood tags, LoRA Packs page with Prepare.
- **Session 3** (2026-11-17): Cloud LoRA training on RunPod (settings, network volume, S3 upload, H100 pods with self-shutdown, live cost, monthly limit, connection test, resume), local 8 GB fit test.
- **Session 4** (2026-11-20): Cloud without the PC (GitHub Actions + RunPod), XL-SFT training, **GLORE Web** (astr0h.github.io/GLORE): Song (vocal → beat built around it with ACE *complete*), **Library** (encrypted uploads, links: SoundCloud in the cloud, YouTube/Spotify/Apple downloaded by the PC app), cloud tools (stems, clean vocal, analyse, tempo/pitch, chops), **Chops** (pack styles, ducked under the vocal), **Vocal FX** (Clean/Airy/Club), **Mix** (kick sidechain, vocal pocket EQ, limiter), **Export** (MP3 320, beat, stems ZIP of 24-bit WAVs).
- **Session 5** (2026-11-22/23): chops rebuilt to replay the song's own melody (any vocal: sung, harmonies, rap), then the **Producer overhaul**: `session.py` (multitrack plans as JSON: clips, track FX, sends, sidechain), `fxrack.py` (26 effects with automation, vocoder, transition synths), `songmap.py` (the song described for a producer), `producer.py` (Claude writes the plan from a brief; rule-based fallback; measures its own render and revises once), cloud `produce` command and Producer mode in GLORE Web. Needs the `GLORE_ANTHROPIC_KEY` repo secret.
- Next: Arrange → Chops (incl. select-to-loop) → Vocal FX → Mix → Export → Generate (ACE-Step, combined mode, loads these LoRAs) → community LoRA import → Remote mode → installer (.exe).
