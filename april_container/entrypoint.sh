#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR env değişkeni set edilmeli}"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "[entrypoint] MODEL_DIR = ${MODEL_DIR}"
if [ ! -d "${MODEL_DIR}" ]; then
  echo "[entrypoint] HATA: MODEL_DIR yok: ${MODEL_DIR}" >&2
  exit 2
fi

echo "[entrypoint] BOM kontrol & temizlik..."
if [ -w "${MODEL_DIR}" ]; then
  python /app/bom_guard.py --root "${MODEL_DIR}" || true
else
  echo "[entrypoint] Not writable (read-only). BOM guard dry-run ile çalışıyor."
  python /app/bom_guard.py --root "${MODEL_DIR}" --dry || true
fi

echo "[entrypoint] HF config/tokenizer kontrolü..."
python /app/validate_hf.py --model "${MODEL_DIR}" || true

# GPU var mı?
if [ -e /dev/nvidiactl ] || ls /dev/nvidia* >/dev/null 2>&1; then
  echo "[entrypoint] GPU bulundu -> vLLM OpenAI API başlatılıyor..."
  exec /app/run_vllm.sh
else
  echo "[entrypoint] GPU yok -> CPU STUB OpenAI API başlatılıyor..."
  exec python /app/cpu_api.py --model "${MODEL_DIR}" --port 8000
fi
