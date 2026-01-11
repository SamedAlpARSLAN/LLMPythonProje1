#!/usr/bin/env python
# -*- coding: utf-8 -*-
# scripts/run_inference.py
#
# Checkpoint'i yükler, tokenizer'ı yükler,
# prompt'u kurar (metin + opsiyonel PDF içerikleri),
# modelden cevap üretir,
# çıktıyı .txt (ve istersen .pdf) olarak kaydeder.

import argparse
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import List

import torch
from PyPDF2 import PdfReader

# -----------------------------------------
# sys.path ayarı (src/april modüllerini bulmak için)
# -----------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIRS = [ROOT / "src" / "april", ROOT / "src", ROOT]
for d in CANDIDATE_DIRS:
    if d.exists():
        sys.path.insert(0, str(d))

try:
    from april_model import AprilModel
    from april_tokenizer import AprilTokenizer
except Exception as e:
    raise SystemExit(
        f"Modül import hatası: {e}\n"
        f"Lütfen src\\april\\ içindeki dosyaların adlarının (april_*.py) doğru olduğundan emin ol."
    )

# -----------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------
def read_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt.strip())
    return "\n\n".join(pages).strip()


def gather_pdfs(items: List[str]) -> list[Path]:
    paths = []
    for it in items or []:
        s = str(it)
        if any(ch in s for ch in ("*", "?")):
            paths += [Path(p) for p in glob.glob(s)]
        else:
            p = Path(s)
            if p.is_dir():
                paths += sorted(p.glob("*.pdf"))
            else:
                paths.append(p)

    uniq, seen = [], set()
    for p in paths:
        if p.exists() and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def to_pdf(text_path: Path, pdf_path: Path, font_ttf: Path | None = None, title: str | None = None):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    if font_ttf and Path(font_ttf).exists():
        pdfmetrics.registerFont(TTFont("DocFont", str(font_ttf)))
        font_name = "DocFont"
    else:
        font_name = "Helvetica"

    c.setFont(font_name, 12)
    y = height - 20 * mm

    if title:
        c.setFont(font_name, 14)
        c.drawString(20 * mm, y, title)
        c.setFont(font_name, 12)
        y -= 10 * mm

    left = 20 * mm
    right = width - 20 * mm
    max_width = right - left

    text = Path(text_path).read_text(encoding="utf-8")

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            y -= 6 * mm
            if y < 20 * mm:
                c.showPage(); c.setFont(font_name, 12); y = height - 20 * mm
            continue

        words = line.split(" ")
        buf = ""
        for w in words:
            test = (buf + " " + w).strip()
            if c.stringWidth(test, font_name, 12) <= max_width:
                buf = test
            else:
                c.drawString(left, y, buf); y -= 6 * mm; buf = w
                if y < 20 * mm:
                    c.showPage(); c.setFont(font_name, 12); y = height - 20 * mm
        if buf:
            c.drawString(left, y, buf); y -= 6 * mm
            if y < 20 * mm:
                c.showPage(); c.setFont(font_name, 12); y = height - 20 * mm
    c.save()


def load_checkpoint(checkpoint_path: Path) -> tuple[dict, dict]:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if "model_state" not in payload or "config" not in payload:
        raise SystemExit(
            "Checkpoint formatı beklenen yapıda değil. "
            "train_april.py ile üretilmiş bir .pth vermelisin."
        )
    return payload["model_state"], payload["config"]


# -----------------------------------------
# main
# -----------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--override_num_heads", type=int, default=None)

    # prompt kaynakları
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--pdf_in", type=str, default=None)
    ap.add_argument("--pdfs", nargs="+", default=None)

    # sampling ayarları
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--top_p", type=float, default=0.9)

    # çıktı ayarları
    ap.add_argument("--out_txt", type=str, default=None)
    ap.add_argument("--out_pdf", type=str, default=None)
    ap.add_argument("--font_ttf", type=str, default=None)

    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)

    # 1) checkpoint yükle
    model_state, cfg = load_checkpoint(ckpt_path)

    vocab_size = int(cfg["vocab_size"])
    embedding_dim = int(cfg["embedding_dim"])
    context_length = int(cfg["context_length"])
    num_layers = int(cfg["num_layers"])
    num_heads = int(args.override_num_heads) if args.override_num_heads is not None else int(cfg["num_heads"])

    # 2) prompt_text derle
    prompt_text = ""
    if args.prompt is not None:
        prompt_text = args.prompt
    elif args.prompt_file is not None:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8", errors="ignore")

    # PDF içeriğini prompt'a ekle
    pdf_paths = []
    if args.pdf_in:
        pdf_paths += gather_pdfs([args.pdf_in])
    if args.pdfs:
        pdf_paths += gather_pdfs(args.pdfs)

    if pdf_paths:
        parts = []
        for p in pdf_paths:
            pdf_txt = read_text_from_pdf(p)
            parts.append(f"\n\n===== {p.name} =====\n\n{pdf_txt}")
        pdf_block = "".join(parts).strip()
        prompt_text = (prompt_text + "\n\n" + pdf_block).strip() if prompt_text else pdf_block

    if not (prompt_text or "").strip():
        raise SystemExit("Prompt boş. --prompt, --prompt_file veya --pdf_in/--pdfs vermen lazım.")

    # 3) tokenizer & encode
    tokenizer = AprilTokenizer(str(tok_path))
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size != vocab_size:
        raise SystemExit(
            f"Tokenizer vocab_size ({getattr(tokenizer, 'vocab_size', '???')}) != "
            f"checkpoint vocab_size ({vocab_size}). Yanlış tokenizer.json olabilir."
        )

    token_ids = tokenizer.encode(prompt_text)
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    else:
        token_ids = token_ids.to(dtype=torch.long)

    if token_ids.shape[0] > context_length:
        token_ids = token_ids[-context_length:]

    prompt_len = token_ids.shape[0]
    max_can_generate = max(context_length - prompt_len, 0)
    max_new_tokens_effective = min(int(args.max_new_tokens), max_can_generate) if max_can_generate > 0 else 0

    if max_new_tokens_effective == 0:
        print("[uyarı] prompt context_length'i dolduruyor; yeni token üretilemeyecek.")

    # 4) model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AprilModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        context_length=context_length,
        num_layers=num_layers,
        device=device,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # 5) generate
    with torch.no_grad():
        gen_ids = model.generate(
            x=token_ids.to(device),
            max_new_tokens=max_new_tokens_effective,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )

    generated_text_raw = tokenizer.decode(gen_ids)
    generated_text = generated_text_raw.replace("<unk>", "")

    # 6) çıktı
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "runs" / ts
    out_txt_path = Path(args.out_txt) if args.out_txt else (run_dir / "out.txt")
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text(generated_text, encoding="utf-8")

    if args.out_pdf:
        out_pdf_path = Path(args.out_pdf)
        out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        to_pdf(
            text_path=out_txt_path,
            pdf_path=out_pdf_path,
            font_ttf=Path(args.font_ttf) if args.font_ttf else None,
            title="TR-LLM Çıktı (April)",
        )

    print(generated_text)


if __name__ == "__main__":
    main()
