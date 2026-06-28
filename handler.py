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
    """LTX-Video can't render 4K natively (it works best under ~1280x720). We
    GENERATE at LTX's max native size for the requested aspect, then upscale the
    finished clip to 4K in a separate ffmpeg pass (_upscale_4k). Rendering tiny is
    the top cause of the 'mutant' look, so we always render at the native ceiling."""
    try:
        w = int(width)
        h = int(height)
    except Exception:
        w, h = 16, 9
    if w > h:               # landscape (16:9) — LTX native ceiling
        return 1280, 704
    if h > w:               # portrait (9:16)
        return 704, 1280
    return 704, 704         # square


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


def _upscale_4k(src, dst, width, height):
    """Upscale the native LTX clip to 4K (UHD class) with a high-quality Lanczos
    scale plus a mild sharpen, re-encoded H.264. LTX can't generate 4K directly,
    so this is the real 'add 4K' step. Aspect is preserved (no stretch): the long
    edge is taken to 3840 (landscape) / the tall edge to 3840 (portrait), square
    to 2160x2160. Returns True on success, False to fall back to the native clip."""
    import subprocess

    if width > height:
        scale = "scale=3840:-2:flags=lanczos"      # ~3840x2112, 4K width
    elif height > width:
        scale = "scale=-2:3840:flags=lanczos"      # ~2112x3840, 4K height
    else:
        scale = "scale=2160:2160:flags=lanczos"    # square UHD-class
    vf = f"{scale},unsharp=5:5:0.8:5:5:0.0"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vf", vf,
             "-c:v", "libx264", "-crf", "16", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


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


def _interpolate(src, dst, fps):
    """Polish a clip: motion-interpolate to `fps` (fixes Wan's choppy 16 fps),
    then crisp it up — 1.5x Lanczos upscale for more apparent resolution, a light
    unsharp for clarity, and a small saturation/contrast lift for punchier color —
    encoded at a low CRF (less compression mush). Returns True on success."""
    import subprocess
    vf = (
        f"minterpolate=fps={int(fps)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        "scale=iw*1.5:ih*1.5:flags=lanczos,"
        "unsharp=5:5:0.7:5:5:0.0,"
        "eq=saturation=1.08:contrast=1.05"
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vf", vf,
             "-c:v", "libx264", "-crf", "15", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def _polish(job_input):
    """POLISH MODE: when given `video_base64`, smooth it via frame interpolation
    instead of generating. Lets the LTX endpoint double as our owned 'make-it-
    smooth' service for clips from any model (e.g. Wan i2v at 16 fps)."""
    b64 = job_input.get("video_base64")
    if b64 and b64.startswith("data:"):
        b64 = b64.split("base64,", 1)[-1]
    if not b64:
        return None  # not a polish request
    target_fps = int(job_input.get("interpolate_fps", 32) or 32)
    stamp = int(time.time() * 1000)
    src = os.path.join(tempfile.gettempdir(), f"polish_{stamp}_in.mp4")
    dst = os.path.join(tempfile.gettempdir(), f"polish_{stamp}_out.mp4")
    try:
        with open(src, "wb") as f:
            f.write(base64.b64decode(b64))
        ok = _interpolate(src, dst, target_fps)
        final = dst if ok else src
        with open(final, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"video": f"data:video/mp4;base64,{out_b64}", "interpolated_fps": target_fps if ok else None}
    except Exception as e:
        return {"error": f"Polish/interpolation failed: {type(e).__name__}: {e}"}
    finally:
        for p in (src, dst):
            try:
                os.remove(p)
            except Exception:
                pass


def handler(job):
    started = time.time()
    job_input = job.get("input", {}) or {}

    # Polish/interpolation request? Smooth the given clip and return (no LTX gen).
    polished = _polish(job_input)
    if polished is not None:
        return polished

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
    # Only the "Max" quality tier asks for the 4K upscale pass.
    want_4k = bool(job_input.get("upscale_4k", False))

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

        stamp = int(time.time() * 1000)
        native_path = os.path.join(tempfile.gettempdir(), f"ltx_{stamp}.mp4")
        export_to_video(frames, native_path, fps=30)

        # 4K pass: upscale the native LTX clip to UHD-class. Falls back to the
        # native clip if ffmpeg is unavailable or the scale fails.
        uhd_path = os.path.join(tempfile.gettempdir(), f"ltx_{stamp}_4k.mp4")
        final_path = native_path
        out_w, out_h = width, height
        if _upscale_4k(native_path, uhd_path, width, height):
            final_path = uhd_path
            if width > height:
                out_w, out_h = 3840, round(3840 * height / width / 2) * 2
            elif height > width:
                out_w, out_h = round(3840 * width / height / 2) * 2, 3840
            else:
                out_w, out_h = 2160, 2160

        with open(final_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        for p in (native_path, uhd_path):
            try:
                os.remove(p)
            except Exception:
                pass

        return {
            "video": f"data:video/mp4;base64,{b64}",
            "seconds": round(time.time() - started, 1),
            "width": out_w,
            "height": out_h,
            "render_width": width,
            "render_height": height,
            "upscaled_4k": final_path == uhd_path,
            "num_frames": num_frames,
        }
    except Exception as e:
        return {"error": f"LTX generation failed: {type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
