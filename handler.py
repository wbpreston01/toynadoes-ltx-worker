"""
handler.py — Toynadoes OWNED text-to-video worker (RunPod serverless).

Runs LTX-Video (Lightricks/LTX-Video, open weights) via diffusers and returns the
clip as a base64 data URI. Pure text-to-video: a prompt, no image. This is the
reliable replacement for the broken community Mochi worker.

Job contract (matches the app's lib/providers/selfhost-t2v.ts exactly):
  input: {
    positive_prompt | prompt : str   (REQUIRED)
    negative_prompt          : str
    width, height            : int    (rounded to /32)
    num_frames               : int    (rounded to 8k+1)
    steps                    : int    (num_inference_steps)
    cfg                      : float  (guidance_scale)
    seed                     : int
  }
  output (COMPLETED): { "video": "data:video/mp4;base64,..." , "seconds": <float> }
  output (FAILED):    { "error": "..." }

Heavy imports live INSIDE the function so `python3 -m py_compile handler.py`
passes on a CPU box with no torch/diffusers installed.
"""

import base64
import os
import tempfile
import time

import runpod

# Lazy singleton so a warm worker loads the pipeline only once.
_pipe = None

MODEL_ID = os.environ.get("LTX_MODEL_ID", "Lightricks/LTX-Video")
# Cache weights on the network volume when one is attached, so cold starts after
# the first download are fast. Falls back to the container's default HF cache.
if os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/hf")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _round32(v, default):
    try:
        v = int(v)
    except Exception:
        v = default
    v = max(256, min(1280, v))
    return (v // 32) * 32


def _round_frames(v, default=97):
    """LTX needs (num_frames - 1) divisible by 8."""
    try:
        v = int(v)
    except Exception:
        v = default
    v = max(25, min(257, v))
    return ((v - 1) // 8) * 8 + 1


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _pipe = pipe
    return _pipe


def handler(job):
    started = time.time()
    job_input = job.get("input", {}) or {}

    prompt = (job_input.get("positive_prompt") or job_input.get("prompt") or "").strip()
    if not prompt:
        return {"error": "No prompt provided (expected 'positive_prompt' or 'prompt')."}

    negative_prompt = (
        job_input.get("negative_prompt")
        or "worst quality, blurry, distorted, deformed, watermark, text, subtitles"
    )
    width = _round32(job_input.get("width", 832), 832)
    height = _round32(job_input.get("height", 480), 480)
    num_frames = _round_frames(job_input.get("num_frames", 97))
    steps = int(job_input.get("steps", 40) or 40)
    guidance = float(job_input.get("cfg", 3.0) or 3.0)
    seed = int(job_input.get("seed", 0) or 0)

    try:
        import torch
        from diffusers.utils import export_to_video

        pipe = _load()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        frames = result.frames[0]

        out_path = os.path.join(tempfile.gettempdir(), f"ltx_{int(time.time()*1000)}.mp4")
        export_to_video(frames, out_path, fps=24)

        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            os.remove(out_path)
        except Exception:
            pass

        return {
            "video": f"data:video/mp4;base64,{b64}",
            "seconds": round(time.time() - started, 1),
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }
    except Exception as e:
        return {"error": f"LTX generation failed: {type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
