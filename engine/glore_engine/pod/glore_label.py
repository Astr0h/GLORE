"""GLORE auto-labeller - runs inside ACE-Step 1.5's own venv, on the cloud GPU.

Usage: python glore_label.py job.json
job.json: ace_dir, dataset_json (in), out_json (out), cache_json, lm (default acestep-5Hz-lm-1.7B),
          genre_ratio (0-100)

ACE-Step's song planner listens to each training track (as audio codes) and writes the
caption and genre tags it would itself use when generating. Training on those captions
teaches the LoRA your sound in ACE-Step's own vocabulary, so prompts at generation time
line up with what the LoRA learned. Results are cached per track so a re-run is free.
Prints GLORE_PROGRESS <0-1> <message> lines.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def say(p: float, msg: str) -> None:
    print(f"GLORE_PROGRESS {p:.3f} {msg}", flush=True)


def clean_caption(caption: str, trigger: str | None) -> str:
    c = " ".join(str(caption or "").split())
    if trigger and c.lower().startswith(trigger.lower()):
        c = c[len(trigger):].lstrip(" ,")
    return c


def merge(doc: dict, labels: dict, genre_ratio: int) -> dict:
    """Put the planner's captions/genres into a Side-Step dataset.json (BPM/key from GLORE stay)."""
    trig = (doc.get("metadata") or {}).get("custom_tag")
    out = json.loads(json.dumps(doc))
    for s in out.get("samples", []):
        lab = labels.get(s.get("id")) or {}
        if lab.get("caption"):
            s["caption"] = clean_caption(lab["caption"], trig)
        if lab.get("genre"):
            s["genre"] = str(lab["genre"])
    out.setdefault("metadata", {})["genre_ratio"] = int(genre_ratio)
    return out


def main(job_path: str) -> int:
    job = json.loads(open(job_path, encoding="utf-8").read())
    ace_dir = job["ace_dir"]
    os.chdir(ace_dir)
    sys.path.insert(0, ace_dir)
    doc = json.loads(open(job["dataset_json"], encoding="utf-8").read())
    base = os.path.dirname(os.path.abspath(job["dataset_json"]))
    cache_path = job["cache_json"]
    labels = json.loads(open(cache_path, encoding="utf-8").read()) if os.path.exists(cache_path) else {}
    todo = [s for s in doc["samples"] if s["id"] not in labels]

    if todo:
        say(0.02, "Loading ACE-Step to listen to your tracks")
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dit = AceStepHandler()
        msg, ok = dit.initialize_service(project_root=ace_dir, config_path="acestep-v15-turbo", device=dev,
                                         use_flash_attention=False, compile_model=False, offload_to_cpu=False,
                                         offload_dit_to_cpu=False, quantization=None)
        if not ok:
            raise RuntimeError(f"Model didn't load: {msg}")
        llm = LLMHandler()
        lm_msg, lm_ok = llm.initialize(checkpoint_dir=os.path.join(ace_dir, "checkpoints"),
                                       lm_model_path=job.get("lm", "acestep-5Hz-lm-1.7B"), backend="pt",
                                       device=dev, offload_to_cpu=False)
        if not lm_ok:
            raise RuntimeError(f"Song planner didn't load: {lm_msg}")

        for i, s in enumerate(todo):
            say(0.05 + 0.9 * i / len(todo), f"Describing track {i + 1}/{len(todo)}")
            path = os.path.normpath(os.path.join(base, s["audio_path"]))
            codes = dit.convert_src_audio_to_codes(path)
            if not codes or str(codes).startswith("❌"):
                print(f"[label] skipped {s['id']}: {str(codes)[:200]}", flush=True)
                labels[s["id"]] = {"caption": "", "genre": ""}
            else:
                meta, status = llm.understand_audio_from_codes(audio_codes=codes, temperature=0.7,
                                                               use_constrained_decoding=True)
                meta = meta or {}
                labels[s["id"]] = {"caption": meta.get("caption", ""), "genre": meta.get("genres", ""),
                                   "bpm": meta.get("bpm"), "keyscale": meta.get("keyscale")}
                print(f"[label] {s['id']}: {labels[s['id']]['caption'][:300]}", flush=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2)

    out = merge(doc, labels, int(job.get("genre_ratio", 30)))
    with open(job["out_json"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    say(1.0, f"Described {len(doc['samples'])} tracks")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:
        traceback.print_exc()
        print("GLORE_ERROR " + json.dumps({"error": str(e)}), flush=True)
        sys.exit(1)
