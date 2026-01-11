#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_DIR:?MODEL_DIR gerekli}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90