#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, time, json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()
MODEL_ID = "APRIL_CAUSAL_LM"
DRY = os.environ.get("CPU_DRY_RUN","1") == "1"  # varsayılan DRY RUN

try:
    if not DRY:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        HF_MODEL_PATH = os.environ.get("MODEL_DIR")
        tok = AutoTokenizer.from_pretrained(HF_MODEL_PATH, use_fast=True, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(HF_MODEL_PATH, trust_remote_code=True, device_map=None)
        device = "cpu"
        mdl.to(device)
    else:
        tok = None
        mdl = None
except Exception:
    # Model yüklenemiyorsa otomatik DRY moduna düş
    tok = None
    mdl = None
    DRY = True

@app.get("/v1/models")
def list_models():
    return {"data":[{"id": MODEL_ID, "object":"model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    # PowerShell/curl gövde sorunlarına tolerans
    try:
        body = await req.json()
    except Exception:
        raw = await req.body()
        try:
            body = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return JSONResponse({"error":"invalid_json","detail":"Body JSON parse edilemedi."}, status_code=400)

    messages    = body.get("messages", [])
    max_tokens  = int(body.get("max_tokens", 64))
    temperature = float(body.get("temperature", 0.7))
    top_p       = float(body.get("top_p", 0.9))

    user_text = "\n".join([m.get("content","") for m in messages if m.get("role")=="user"]).strip()

    if DRY or mdl is None or tok is None:
        reply = f"[CPU-STUB] Merhaba! (DRY RUN) Son kullanıcı mesajı: {user_text[:200]}"
        comp_tokens = 10
        prompt_tokens = len(user_text.split())
    else:
        import torch
        enc = tok(user_text, return_tensors="pt")
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = mdl.generate(
                input_ids=input_ids.to(mdl.device),
                do_sample=True,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=getattr(tok, "eos_token_id", None)
            )
        gen = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
        reply = gen.strip()
        comp_tokens = len(tok.encode(gen))
        prompt_tokens = int(input_ids.numel())

    now = int(time.time())
    return JSONResponse({
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role":"assistant","content": reply},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_tokens + comp_tokens
        }
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False)  # entrypoint geçiriyor
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
