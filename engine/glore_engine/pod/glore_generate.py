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
  mode         text2music (default) | cover (src_audio, strength 0-1) | complete (src_audio, tracks; base only)
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

    llm = None
    lm_name = job.get("lm")
    if lm_name:
        # The 5Hz LM plans the song (structure, rhythm, melody as audio codes) before the DiT renders it.
        # Without it the DiT has no plan, which sounds like a shapeless wash.
        say(0.18, "Loading the song planner")
        from acestep.llm_inference import LLMHandler
        llm = LLMHandler()
        lm_msg, lm_ok = llm.initialize(checkpoint_dir=os.path.join(ace_dir, "checkpoints"), lm_model_path=lm_name,
                                       backend="pt", device="cuda" if torch.cuda.is_available() else "cpu",
                                       offload_to_cpu=True)
        if not lm_ok:
            raise RuntimeError(f"Song planner didn't load: {lm_msg}")

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
        thinking=bool(llm),
        use_cot_metas=False, use_cot_caption=False, use_cot_language=False,  # keep our BPM/key/caption
    )
    if job.get("mode") == "complete":
        # Build the other instruments around an existing part (e.g. your vocal). Base model only.
        tracks = ", ".join(job.get("tracks") or ["drums", "bass", "keyboard", "synth"])
        params = GenerationParams(task_type="complete", src_audio=job["src_audio"],
                                  instruction=f"Complete the input track with {tracks}:",
                                  guidance_scale=float(job.get("guidance", 7.0)),
                                  duration=float(job.get("duration") or -1), **common)
    elif job.get("mode") == "cover":
        # Remake a song in a new style, keeping its bars, rhythm, chords and structure (works on turbo + LoRA).
        # Fed the instrumental a vocal was written to, the new beat lines up with that vocal.
        params = GenerationParams(task_type="cover", src_audio=job["src_audio"], instrumental=True,
                                  audio_cover_strength=float(job.get("strength", 0.7)),
                                  duration=float(job.get("duration") or -1), **common)
    else:
        params = GenerationParams(task_type="text2music", instrumental=True,
                                  duration=float(job.get("duration", 60)), **common)
    seed = int(job.get("seed", -1))
    count = max(1, int(job.get("count", 1)))
    config = GenerationConfig(batch_size=count, audio_format="wav",
                              use_random_seed=seed < 0, seeds=None if seed < 0 else [seed + i for i in range(count)])
    say(0.35, {"complete": "Building around your vocal", "cover": "Rebuilding the song as a new beat"}
        .get(job.get("mode"), f"Generating {params.duration:.0f} s"))
    os.makedirs(job["out_dir"], exist_ok=True)
    result = generate_music(dit, llm, params, config, save_dir=job["out_dir"])
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
