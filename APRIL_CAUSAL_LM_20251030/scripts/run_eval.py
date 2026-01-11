# scripts/run_eval.py
#
# Bu script, başka bir programın pipe ile "prompt" göndermesi için kolay arayüz.
#
# Örnek:
# echo "{\"prompt\": \"Bu PDF'i maddeler halinde özetle\"}" |
#   .venv/Scripts/python.exe scripts/run_eval.py
#
# Ortam değişkenleri:
#   TRLLM_CKPT             -> checkpoint .pth yolu
#   TRLLM_TOK              -> tokenizer.json yolu
#   TRLLM_OVERRIDE_HEADS   -> (opsiyonel) num_heads override
#
# Çıktı:
#   {"output": "model cevabı ..."}
#

import sys
import json
import os
from pathlib import Path
from subprocess import check_output, CalledProcessError


def main():
    # -------------------------------------------------
    # stdin'den JSON çek
    # -------------------------------------------------
    raw_in = sys.stdin.read().strip()
    try:
        req = json.loads(raw_in) if raw_in else {}
    except Exception:
        print(json.dumps({"error": "invalid_json"}))
        return

    prompt = req.get("prompt", "")
    if not prompt:
        print(json.dumps({"error": "missing_prompt"}))
        return

    # -------------------------------------------------
    # Prompt'ı geçici dosyaya yaz (run_inference --prompt_file ile kullanacağız)
    # -------------------------------------------------
    tmp_in = Path("outputs/runs/eval_prompt.txt")
    tmp_in.parent.mkdir(parents=True, exist_ok=True)
    tmp_in.write_text(prompt, encoding="utf-8")

    # -------------------------------------------------
    # Ortam değişkenlerinden model/tokenizer yollarını al
    # -------------------------------------------------
    ckpt = os.environ.get("TRLLM_CKPT", "").strip()
    tok = os.environ.get("TRLLM_TOK", "").strip()
    override_heads = os.environ.get("TRLLM_OVERRIDE_HEADS", "").strip()  # opsiyonel

    if not ckpt or not tok:
        print(json.dumps({"error": "missing_env",
                          "detail": "TRLLM_CKPT ve TRLLM_TOK env değişkenlerini ayarla."},
                         ensure_ascii=False))
        return

    # -------------------------------------------------
    # run_inference.py komutunu hazırla
    # -------------------------------------------------
    cmd = [
        sys.executable,
        "scripts/run_inference.py",
        "--checkpoint", ckpt,
        "--tokenizer", tok,
        "--prompt_file", str(tmp_in),
        "--max_new_tokens", "128",
        "--temperature", "1.0",
        "--top_k", "32",
        "--top_p", "0.9",
    ]

    if override_heads:
        cmd += ["--override_num_heads", override_heads]

    # -------------------------------------------------
    # Dış scripti çalıştır
    # -------------------------------------------------
    try:
        out = check_output(cmd, text=True)
    except CalledProcessError as e:
        # Eğer run_inference içinde SystemExit vs atarsa buraya düşebilir
        print(json.dumps({
            "error": "inference_failed",
            "detail": e.stderr if hasattr(e, "stderr") else str(e),
        }, ensure_ascii=False))
        return

    # -------------------------------------------------
    # Başarı -> cevabı JSON olarak ver
    # -------------------------------------------------
    print(json.dumps({"output": out.strip()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
