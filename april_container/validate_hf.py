#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
from transformers import AutoConfig, AutoTokenizer

def _load_cfg(model_path: str, trc: bool):
    return AutoConfig.from_pretrained(model_path, trust_remote_code=trc)

def _load_tok(model_path: str, trc: bool):
    return AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model_path = args.model
    cpu_dry = os.environ.get("CPU_DRY_RUN", "0") == "1"

    print(f"[validate_hf] Model dizini  : {model_path}")
    print(f"[validate_hf] CPU_DRY_RUN   : {cpu_dry}")

    try:
        cfg = _load_cfg(model_path, trc=False)
    except Exception:
        print("[validate_hf] Bilgi: config TRC=False başarısız, TRC=True ile yeniden denenecek...")
        cfg = _load_cfg(model_path, trc=True)
    print(f"[validate_hf] model_type    : {getattr(cfg, 'model_type', None)}")
    print(f"[validate_hf] architectures : {getattr(cfg, 'architectures', None)}")

    try:
        tok = _load_tok(model_path, trc=False)
    except Exception:
        print("[validate_hf] Bilgi: tokenizer TRC=False başarısız, TRC=True ile yeniden denenecek...")
        tok = _load_tok(model_path, trc=True)
    _ = tok("Merhaba dünya!")

    print("[validate_hf] Hafif kontrol OK (config+tokenizer).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
