FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.10 -m venv .venv
ENV PATH="/app/.venv/bin:${PATH}"
RUN pip install --upgrade pip

# PyTorch (cu121) – CPU stub için de kalabilir
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Uygulama dosyaları
COPY entrypoint.sh   /app/entrypoint.sh
COPY bom_guard.py    /app/bom_guard.py
COPY validate_hf.py  /app/validate_hf.py
COPY run_vllm.sh     /app/run_vllm.sh
COPY cpu_api.py      /app/cpu_api.py

RUN chmod +x /app/entrypoint.sh /app/run_vllm.sh

ENV MODEL_DIR=/app/models/APRIL_CAUSAL_LM \
    VLLM_NO_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
