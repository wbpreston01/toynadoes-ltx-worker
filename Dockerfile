# Toynadoes OWNED text-to-video worker — LTX-Video on RunPod serverless.
# Built on the CUDA-enabled PyTorch base so torch sees the GPU out of the box.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# ffmpeg is needed by imageio to mux frames into an mp4.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch is already in the base image; install the rest without re-pulling torch.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
        runpod==1.7.7 \
        diffusers==0.32.2 \
        transformers==4.46.3 \
        accelerate==1.1.1 \
        sentencepiece==0.2.0 \
        imageio==2.36.1 \
        imageio-ffmpeg==0.5.1 \
        hf_transfer==0.1.8

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
