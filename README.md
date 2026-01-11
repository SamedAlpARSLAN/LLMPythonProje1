# LLMPythonProje1 — APRIL_CAUSAL_LM (Causal Language Model) | Offline Showcase (Code + Tokenizer + Checkpoint + Docker)

Bu repo, **APRIL_CAUSAL_LM_20251030** isimli Causal Language Model (autoregressive / next-token prediction) çalışmasının **offline çalıştırılabilir** ve **akademik sunum/teslim** odaklı paketlenmiş halidir. Amaç; model kodu, tokenizer/vocab, checkpoint/weights, inference–eval script’leri ve Docker araçlarını tek repoda toplayarak **tekrar üretilebilir bir demo** sunmaktır.

## İçerik Özeti
- **Model Türü:** Causal Language Model (next-token prediction)
- **Kullanım:** CPU ortamında prompt → metin üretimi (inference), deney/eval akışı
- **Paket İçeriği:** Model bileşenleri + tokenizer dosyaları + checkpoint/weights + script’ler + Docker yardımcıları
- **Arşiv:** `APRIL_CAUSAL_LM_EnGüncel.zip` (showcase/backup amaçlı)

## Repo Yapısı
- `APRIL_CAUSAL_LM_20251030/` → ana proje paketi  
  - `april_*.py` → modelin çekirdek bileşenleri (attention, decoder block, embedding, layer norm, MLP, tokenizer vb.)
  - `checkpoint/` → eğitim çıktısı checkpoint’ler (örn. `april_step132810.pth`)
  - `scripts/` → `run_inference.py`, `run_eval.py`, `train_april.py`, `requirements.txt`, `prompts.txt` vb.
  - `april_vocab.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json` → tokenizer/config dosyaları
  - `outputs/` → önceki koşumların metin çıktıları (run çıktıları)
- `april_container/` → container çalıştırma/validasyon yardımcıları
- `openai_server/` → (opsiyonel) OpenAI-benzeri yerel servis altyapısı
- `tools/` → Docker CLI ve PowerShell çalışma script’leri
- `APRIL_CAUSAL_LM_EnGüncel.zip` → aynı içeriğin zip arşivi (backup/showcase)

## Tek Komut Bloğunda Kurulum + Çalıştırma (Windows / PowerShell)
Aşağıdaki bloktaki adımlar **sırayla** çalıştırılır. (Repo kök dizininde olun.)


# =========================
# 0) (Opsiyonel) Repo'yu klonla
# =========================
# git clone https://github.com/SamedAlpARSLAN/LLMPythonProje1.git
# cd LLMPythonProje1

# =========================
# 1) Sanal ortam (venv)
# =========================
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# =========================
# 2) Bağımlılıkları yükle
# =========================
pip install -r .\APRIL_CAUSAL_LM_20251030\scripts\requirements.txt

# =========================
# 3) Tek prompt ile inference (CPU)
# =========================
python .\APRIL_CAUSAL_LM_20251030\scripts\run_inference.py `
  --checkpoint .\APRIL_CAUSAL_LM_20251030\checkpoint\april_step132810.pth `
  --tokenizer  .\APRIL_CAUSAL_LM_20251030\april_vocab.json `
  --prompt "Türkiye'nin başkenti neresidir?" `
  --max_new_tokens 64 --temperature 0 --top_k 0 --top_p 1

# =========================
# 4) Toplu prompt ile inference (prompts.txt)
# =========================
python .\APRIL_CAUSAL_LM_20251030\scripts\run_inference.py `
  --checkpoint .\APRIL_CAUSAL_LM_20251030\checkpoint\april_step132810.pth `
  --tokenizer  .\APRIL_CAUSAL_LM_20251030\april_vocab.json `
  --prompts_file .\APRIL_CAUSAL_LM_20251030\scripts\prompts.txt `
  --max_new_tokens 64 --temperature 0 --top_k 0 --top_p 1

# =========================
# 5) (Opsiyonel) Eval çalıştırma
# =========================
# python .\APRIL_CAUSAL_LM_20251030\scripts\run_eval.py

# =========================
# 6) (Opsiyonel) Docker ile çalışma (repo içindeki araçlara göre)
# =========================
# docker build -t april-causal-lm -f .\april_container\Dockerfile .
# docker run --rm -it april-causal-lm


## Çıktılar

* Örnek koşum çıktıları `APRIL_CAUSAL_LM_20251030/outputs/` altında tutulur.
* Inference çıktıları script’e göre ekrana basılabilir ve/veya dosyaya yazılabilir.

## Akademik Not

Bu repo “showcase/teslim” amaçlı düzenlenmiştir. Eğitim veri seti/kuralları, deney metodolojisi ve ek raporlama, ders/ödev yönergelerine göre ayrıca belgelendirilebilir.

## Lisans / Kullanım

Bu repo eğitim ve akademik gösterim amaçlıdır. Üçüncü taraf veri ve içerikler ilgili lisans/kurallara tabidir.


::contentReference[oaicite:0]{index=0}

