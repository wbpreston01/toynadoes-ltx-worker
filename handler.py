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


def _ltx_resolution(width, height):
    """LTX-Video looks best at its native ~1216x704 (30fps) class of sizes, all
    /32. We ignore the app's small incoming dims and snap to the LTX-native size
    matching the requested aspect — low resolution is a top cause of the
    'mutant' look, so we never render tiny."""
    try:
        w = int(width)
        h = int(height)
    except Exception:
        w, h = 16, 9
    if w > h:               # landscape (16:9)
        return 1216, 704
    if h > w:               # portrait (9:16)
        return 704, 1216
    return 768, 768         # square


# LTX is trained on long, detailed, single-paragraph prompts. A one-line prompt
# leaves the model free to hallucinate anatomy/texture (the 'nightmare doll'
# failure). For short prompts we wrap the user's idea in a descriptive,
# physically-grounded scaffold; detailed prompts are left alone.
def _enrich_prompt(p):
    p = p.strip().rstrip(".")
    if len(p.split()) >= 30:
        return p
    return (
        f"{p}. The scene is filmed as a real, physically plausible shot with "
        "natural, correct anatomy and proportions and smooth, believable motion. "
        "Soft cinematic lighting, shallow depth of field, a gentle slow camera "
        "move, warm color grade, highly detailed realistic textures, sharp focus, "
        "professional cinematography, coherent and stable throughout."
    )


# Strong default negative prompt aimed squarely at the body-horror artifacts.
_NEG = (
    "deformed, mutated, mutation, extra limbs, fused limbs, missing limbs, "
    "distorted anatomy, malformed, disfigured, grotesque, body horror, melting, "
    "warped face, two heads, uncanny, jittery, flickering, morphing, low quality, "
    "worst quality, blurry, watermark, text, subtitles, duplicate"
)


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

    prompt = _enrich_prompt(prompt)
    # Always fold the strong anti-body-horror terms into any caller-supplied negative.
    caller_neg = (job_input.get("negative_prompt") or "").strip()
    negative_prompt = f"{caller_neg}, {_NEG}" if caller_neg else _NEG

    width, height = _ltx_resolution(job_input.get("width", 1216), job_input.get("height", 704))
    num_frames = _round_frames(job_input.get("num_frames", 97))
    # LTX (non-distilled) wants more steps and higher guidance than we first shipped;
    # cfg ~5 sharply improves prompt adherence and kills hallucinated anatomy.
    steps = int(job_input.get("steps", 50) or 50)
    guidance = float(job_input.get("cfg", 5.0) or 5.0)
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
        export_to_video(frames, out_path, fps=30)

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
