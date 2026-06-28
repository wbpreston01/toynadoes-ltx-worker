# Toynadoes — owned text-to-video worker (LTX-Video)

A RunPod serverless worker that turns a prompt into an mp4 clip using
[LTX-Video](https://huggingface.co/Lightricks/LTX-Video) (open weights) through
`diffusers`. This is the OWNED, self-contained replacement for the broken
community Mochi worker — no Veo, no fal, no third-party API.

It speaks the exact job contract the app already sends from
`lib/providers/selfhost-t2v.ts`, so once it's deployed the only change is
pointing `SELFHOST_T2V_ENDPOINT_ID` at the new endpoint.

## Job contract

```jsonc
// input
{
  "positive_prompt": "a felt fox astronaut hops across a snowy quilt mountain at dawn",
  "negative_prompt": "blurry, watermark, text",
  "width": 832, "height": 480,   // rounded to /32 by the worker
  "num_frames": 97,              // rounded to 8k+1 by the worker
  "steps": 40, "cfg": 3.0, "seed": 12345
}
// output (success)
{ "video": "data:video/mp4;base64,AAAA...", "seconds": 41.2 }
// output (failure)
{ "error": "..." }
```

## Deploy (RunPod, GitHub integration)

1. Push this folder to its own GitHub repo (commands below).
2. RunPod → Serverless → New Endpoint → **GitHub repo** → pick the repo.
3. Container disk 20 GB. Attach a **network volume** (~40 GB) mounted at
   `/runpod-volume` so the LTX weights persist between cold starts.
4. GPU: any 24 GB+ card (A5000 / L40S / A100 all fine — LTX is small and fast).
5. Build, then copy the **Endpoint ID** into `.env.local`:
   `SELFHOST_T2V_ENDPOINT_ID=<new id>`.

First request downloads the weights to the volume (a few minutes); after that
cold starts are quick and every later clip is fast.

## Local sanity check

`python3 -m py_compile handler.py` compiles with no GPU/torch present, because
the heavy imports live inside the functions.
