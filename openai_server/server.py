import os, sys, uuid, time
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

# ====== Model klasörü ======
MODEL_ROOT = Path(os.getenv("APRIL_MODEL_DIR", "/model")).resolve()
for cand in [MODEL_ROOT, MODEL_ROOT / "src", MODEL_ROOT / "src" / "april"]:
    if cand.exists():
        sys.path.insert(0, str(cand))

from april_model import AprilModel            # sizin dosyalar
from april_tokenizer import AprilTokenizer    # sizin dosyalar

def find_checkpoint(root: Path) -> Path:
    for d in ["checkpoint", "checkpoints"]:
        p = root / d
        if p.exists():
            cands = sorted(p.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
            if cands:
                return cands[0]
    raise FileNotFoundError("Checkpoint (.pth) bulunamadı")

def find_tokenizer(root: Path) -> Path:
    for name in ["april_vocab.json", "tokenizer.json", "vocab.json"]:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError("Tokenizer/Vocab (.json) bulunamadı")

def load_checkpoint(checkpoint_path: Path):
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if "model_state" not in payload or "config" not in payload:
        raise ValueError("Checkpoint formatı beklenenden farklı")
    return payload["model_state"], payload["config"]

# ====== Start-up: Modeli yükle ======
torch.set_num_threads(1)
CKPT = Path(os.getenv("APRIL_CKPT", ""))
TOK  = Path(os.getenv("APRIL_TOK", ""))

if not CKPT.exists():
    CKPT = find_checkpoint(MODEL_ROOT)
if not TOK.exists():
    TOK = find_tokenizer(MODEL_ROOT)

model_state, cfg = load_checkpoint(CKPT)
vocab_size      = int(cfg["vocab_size"])
embedding_dim   = int(cfg["embedding_dim"])
context_length  = int(cfg["context_length"])
num_layers      = int(cfg["num_layers"])
num_heads       = int(cfg.get("num_heads", 2))

device = "cpu"
MODEL = AprilModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    context_length=context_length,
    num_layers=num_layers,
    device=device,
).to(device)
MODEL.load_state_dict(model_state, strict=True)
MODEL.eval()

TOKENIZER = AprilTokenizer(str(TOK))

# ====== API şemaları ======
class CompletionRequest(BaseModel):
    model: Optional[str] = "april"
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 32
    top_p: float = 0.9

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "april"
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 32
    top_p: float = 0.9

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status":"ok","model":"april","vocab_size":vocab_size,"ctx":context_length}

def _generate(prompt: str, max_new_tokens=128, temperature=1.0, top_k=32, top_p=0.9):
    ids = TOKENIZER.encode(prompt)
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long)
    else:
        ids = ids.to(dtype=torch.long)
    if ids.shape[0] > context_length:
        ids = ids[-context_length:]
    with torch.no_grad():
        gen = MODEL.generate(
            x=ids.to("cpu"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    txt = TOKENIZER.decode(gen).replace("<unk>", "")
    return txt

# ---- OpenAI benzeri uçlar ----
@app.post("/v1/completions")
def completions(req: CompletionRequest):
    out = _generate(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index":0, "text": out, "finish_reason":"stop"}],
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # Basit birleştirme: system+user+assistant içerikleri alt alta
    prompt = "\n".join([m.content for m in req.messages])
    out = _generate(
        prompt=prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index":0,
            "message":{"role":"assistant","content": out},
            "finish_reason":"stop"
        }],
    }
