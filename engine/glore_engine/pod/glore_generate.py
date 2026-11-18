"""GLORE generator - runs inside ACE-Step 1.5's own venv (not the GLORE engine's).

Usage: python glore_generate.py job.json
Prints "GLORE_PROGRESS <0-1> <message>" lines while working and one
"GLORE_RESULT <json>" line at the end; exit code 0 on success.

job.json:
  ace_dir      ACE-Step checkout (holds checkpoints/)
  out_dir      where the audio goes
  variant      turbo | base | sft            (default turbo)
  caption      text prompt, without the LoRA trigger word
  lora         {"path": ..., "trigger": "glore_house", "scale": 1.0} or null
  bpm, key, duration (s), seed (-1 = random), count (takes per run), steps, shift
  offload      move the parts not in use to system RAM (needed on 8 GB cards)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback


def say(p: float, msg: str) -> None:
    print(f"GLORE_PROGRESS {p:.3f} {msg}", flush=True)


def main(job_path: str) -> int:
    job = json.loads(open(job_path, encoding="utf-8").read())
    ace_dir = job["ace_dir"]
    os.chdir(ace_dir)
    sys.path.insert(0, ace_dir)

    say(0.02, "Loading ACE-Step")
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    variant = job.get("variant", "turbo")
    dit = AceStepHandler()
    say(0.08, "Loading the model onto your GPU")
    t0 = time.time()
    msg, ok = dit.initialize_service(
        project_root=ace_dir,
        config_path=f"acestep-v15-{variant}",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=bool(job.get("offload", True)),
        offload_dit_to_cpu=bool(job.get("offload_dit", False)),
        quantization=None,  # LoRAs can't load into a quantized model
    )
    if not ok:
        raise RuntimeError(f"Model didn't load: {msg}")

    caption = (job.get("caption") or "").strip()
    lora = job.get("lora")
    if lora:
        say(0.25, "Loading the LoRA")
        m = dit.load_lora(lora["path"])
        if not str(m).startswith("✅"):
            raise RuntimeError(f"LoRA didn't load: {m}")
        dit.set_use_lora(True)
        dit.set_lora_scale(float(lora.get("scale", 1.0)))
        trig = lora.get("trigger")
        if trig and not caption.startswith(trig):
            caption = f"{trig}, {caption}" if caption else trig  # same as training (tag_position=prepend)

    common = dict(
        caption=caption,
        lyrics="[Instrumental]",
        bpm=int(round(job["bpm"])) if job.get("bpm") else None,
        keyscale=job.get("key") or "",
        timesignature="4",
        inference_steps=int(job.get("steps", 8 if variant == "turbo" else 50)),
        shift=float(job.get("shift", 3.0 if variant == "turbo" else 1.0)),
        seed=int(job.get("seed", -1)),
        thinking=False,          # no 5Hz LM: keeps it inside 8 GB
        use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
    )
    if job.get("mode") == "complete":
        # Build the other instruments around an existing part (e.g. your vocal). Base model only.
        tracks = ", ".join(job.get("tracks") or ["drums", "bass", "keyboard", "synth"])
        params = GenerationParams(task_type="complete", src_audio=job["src_audio"],
                                  instruction=f"Complete the input track with {tracks}:",
                                  guidance_scale=float(job.get("guidance", 7.0)),
                                  duration=float(job.get("duration") or -1), **common)
    else:
        params = GenerationParams(task_type="text2music", instrumental=True,
                                  duration=float(job.get("duration", 60)), **common)
    seed = int(job.get("seed", -1))
    count = max(1, int(job.get("count", 1)))
    config = GenerationConfig(batch_size=count, audio_format="wav",
                              use_random_seed=seed < 0, seeds=None if seed < 0 else [seed + i for i in range(count)])
    say(0.35, "Building around your vocal" if job.get("mode") == "complete" else f"Generating {params.duration:.0f} s")
    os.makedirs(job["out_dir"], exist_ok=True)
    result = generate_music(dit, None, params, config, save_dir=job["out_dir"])
    if not result.success:
        raise RuntimeError(result.error or result.status_message or "Generation failed")
    files = [{"path": a.get("path"), "seed": (a.get("params") or {}).get("seed")} for a in result.audios]
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
    print("GLORE_RESULT " + json.dumps({"files": files, "caption": caption, "seconds": round(time.time() - t0, 1),
                                        "peak_vram_gb": round(peak, 2)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:
        traceback.print_exc()
        print("GLORE_ERROR " + json.dumps({"error": str(e)}), flush=True)
        sys.exit(1)
