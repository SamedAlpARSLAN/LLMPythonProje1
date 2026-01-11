#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt

python scripts/run_inference.py \
  --checkpoint checkpoint/april_step132810.pth \
  --tokenizer model/tokenizer.json \
  --num_heads 3 \
  --prompt_file data/prompts/ornek.txt \
  --max_new_tokens 64 \
  --temperature 1.0 \
  --top_k 32 \
  --top_p 0.9 \
  --out_txt outputs/runs/$(date +%Y%m%d_%H%M%S)/out.txt
