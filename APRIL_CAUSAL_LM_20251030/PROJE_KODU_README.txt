PROJE_KODU_README.txt
======================
Paket: APRIL_CAUSAL_LM (20251030)
Kapsam: TR-LLM deneme modeli kontrol/çalıştırma paketidir. Bu dosya; klasör yapısı, sistem gereksinimleri,
kurulum/çalıştırma adımları, doğrulama testleri, sorun giderme, BOM temizlik ve SHA‑256 üretimini içerir.

--------------------------------------------------------------------
1) KLASÖR YAPISI
--------------------------------------------------------------------
gönderilecek april/
├─ APRIL_CAUSAL_LM_20251030/          -> Model kodu + kontrol betikleri + ağırlıklar
│  ├─ checkpoint/                      -> Checkpoint (.pth)  Örn: april_step132810.pth
│  ├─ scripts/
│  │  ├─ run_inference.py             -> Basit CLI üretim betiği (tek/batch test)
│  │  ├─ prompts.txt                  -> (İsteğe bağlı) Batch test için metin istemleri
│  │  └─ requirements.txt             -> (Varsa) run_inference için ek bağımlılıklar
│  ├─ outputs/runs/                   -> Çıktıların otomatik kaydedildiği klasör
│  ├─ april_*.py                      -> Model katman kodları (nöral ağ bileşenleri)
│  ├─ april_vocab.json or tokenizer.json
│  └─ (diğer json/py dosyaları)
├─ tools/
│  ├─ docker_run_april_cli.ps1        -> Tek komutla Docker’da run_inference.py çalıştırır
│  └─ Dockerfile.cli                  -> CLI imajı (CPU) için Dockerfile
├─ april_container/                   -> (Ayrı) API/VLM kapsayıcısı için materyaller
│  ├─ Dockerfile, entrypoint.sh, cpu_api.py, validate_hf.py, run_vllm.sh, bom_guard.py
│  └─ requirements.txt, README.md
└─ payload.json                        -> Başvuruya eşlik eden meta dosya 

Not: Değerlendirici yalnızca CLI ile doğrulama yapacaksa APRIL_CAUSAL_LM_20251030 + tools yeterlidir.


--------------------------------------------------------------------
2) SİSTEM GEREKSİNİMLERİ
--------------------------------------------------------------------
- Windows 10/11 veya Linux (x86_64)
- Docker Desktop (Windows’ta WSL2 etkin olmalı) veya Docker Engine
- İnternet (ilk imaj kurulumunda bağımlılık çekmek için)
- CPU doğrulaması için ek donanım gerekmez (CLI imajı PyTorch CPU kullanır).


--------------------------------------------------------------------
3) HIZLI KURULUM ve ÇALIŞTIRMA (CLI – CPU)
--------------------------------------------------------------------
Komutlar, “gönderilecek april” klasörü içinde çalıştırılmalıdır.

A) Tek İstek (inline prompt, greedy/tekrarsız üretim örneği)
   PowerShell:
     powershell -ExecutionPolicy Bypass -File .\tools\docker_run_april_cli.ps1 `
       -Prompt "Türkiye'nin başkenti neresidir?" `
       -GenArgs @('--temperature=0','--top_p=1','--top_k=0','--max_new_tokens=24')

B) Batch Test (prompts.txt ile)
   1. Dosya oluşturun: APRIL_CAUSAL_LM_20251030\scripts\prompts.txt
      Örnek içerik:
        Türkiye'nin başkenti neresidir?
        Atatürk kaç yılında doğdu?
   2. Çalıştırın:
      powershell -ExecutionPolicy Bypass -File .\tools\docker_run_april_cli.ps1 `
        -GenArgs @('--temperature=0','--top_p=1','--top_k=0','--max_new_tokens=24')

C) (İsteğe bağlı) PDF içeriğini prompt’a ekleyerek çalıştırma
   PDF metni prompt’un sonuna eklenir.
   Örnek (tek PDF):
      powershell -ExecutionPolicy Bypass -File .\tools\docker_run_april_cli.ps1 `
        -Prompt "Metinden kısa bir özet çıkar:" `
        -GenArgs @('--temperature=0','--top_p=1','--top_k=0','--max_new_tokens=64')
   Ardından container içinde run_inference.py şu eşdeğer parametrelerle çalışır:
      --pdf_in "/work/…/belgeniz.pdf"   veya  --pdfs "/work/…/*.pdf"
   (PDF’leri /work içine bağlamak için dosyaları “gönderilecek april” altına yerleştirmeniz yeterlidir.)

Çıktıların yeri:
- Metin çıktısı: APRIL_CAUSAL_LM_20251030\outputs\runs\YYYYMMDD_HHMMSS\out.txt


--------------------------------------------------------------------
4) RUN_INFERENCE PARAMETRE ÖZETİ
--------------------------------------------------------------------
Zorunlu (betik tarafından otomatik verilir):
  --checkpoint  -> /app/model/checkpoint.pth (host’taki .pth read‑only bağlanır)
  --tokenizer   -> /app/model/tokenizer.json (april_vocab.json kullanılıyorsa da aynıdır)

İsteğe bağlı üretim parametreleri (örnek varsayılanlar):
  --max_new_tokens 128   | --temperature 1.0 | --top_k 32 | --top_p 0.9
Greedy doğrulama için: temperature=0, top_k=0, top_p=1 önerilir.

Prompt kaynakları:
  --prompt "…",  --prompt_file path,  --pdf_in path veya --pdfs çoklu girdi

Not: Windows PowerShell’de -GenArgs kullanımında her argümanı ayrı string olarak verin.
Önerilen güvenli biçim: -GenArgs @('--temperature=0','--top_p=1','--top_k=0','--max_new_tokens=24')
(“--temperature,0,…” şeklinde tek bir string vermeyin.)


--------------------------------------------------------------------
5) SORUN GİDERME
--------------------------------------------------------------------
- “Prompt boş. …” uyarısı:
   -Prompt verin veya APRIL_CAUSAL_LM_20251030\scripts\prompts.txt oluşturun.

- “unrecognized arguments: --temperature,0,…” hatası:
   -GenArgs dizisini önerildiği eşitlikli biçimde girin (bkz. 4. bölüm).

- “The property 'OutputRendering'…” (PS5.1):
   Bilgilendirme niteliğindedir; betik çalışma akışını etkilemez.

- “Resolve-Path … tokenizer.json bulunamadı”:
   Bu pakette “april_vocab.json” mevcuttur; -Tok parametresi vermeyin,
   betik otomatik bulur. Manuel verecekseniz doğru dosya adını kullanın.

- “Read-only file system … bom_guard.py”:
   Kapsayıcı içinde model dosyaları read‑only bağlanır. BOM temizliği host’ta yapılmalıdır
   (bkz. 6. bölüm).


--------------------------------------------------------------------
6) BOM TEMİZLİĞİ (HOST TARAFI)
--------------------------------------------------------------------
Amaç: .py/.sh/.txt/.json dosyalarında UTF‑8 BOM (EF BB BF) varsa temizlemek.
ÖNEMLİ: Aşağıdaki işlemler metin dosyaları içindir; .pth/.bin gibi ikililere DOKUNMAYIN.

A) Sadece hangi dosyalarda BOM var görmek için (ön izle):
  PowerShell:
    Get-ChildItem ".\APRIL_CAUSAL_LM_20251030" -Recurse -Include *.py,*.sh,*.txt,*.md,*.json |
      ForEach-Object {
        $b = [System.IO.File]::ReadAllBytes($_.FullName);
        if ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF) { $_.FullName }
      }

B) Temizlik – PowerShell 7+ (utf8NoBOM destekli):
    Get-ChildItem ".\APRIL_CAUSAL_LM_20251030" -Recurse -Include *.py,*.sh,*.txt,*.md,*.json |
      ForEach-Object {
        $raw = Get-Content -Raw -Encoding Byte $_.FullName
        if ($raw.Length -ge 3 -and $raw[0] -eq 239 -and $raw[1] -eq 187 -and $raw[2] -eq 191) {
          $txt = [System.Text.Encoding]::UTF8.GetString($raw,3,$raw.Length-3)
          Set-Content -Path $_.FullName -Value $txt -NoNewline -Encoding utf8NoBOM
        }
      }

C) Temizlik – Windows PowerShell 5.1 (utf8NoBOM yok):
    Get-ChildItem ".\APRIL_CAUSAL_LM_20251030" -Recurse -Include *.py,*.sh,*.txt,*.md,*.json |
      ForEach-Object {
        $b = [System.IO.File]::ReadAllBytes($_.FullName)
        if ($b.Length -gt 3 -and $b[0] -eq 239 -and $b[1] -eq 187 -and $b[2] -eq 191) {
          [System.IO.File]::WriteAllBytes($_.FullName, $b[3..($b.Length-1)])
        }
      }


--------------------------------------------------------------------
7) SHA‑256 ÖZET DEĞERLERİ (HASH)
--------------------------------------------------------------------
Tek tek önemli dosyalar için:
  Get-FileHash -Algorithm SHA256 ".\APRIL_CAUSAL_LM_20251030\checkpoint\april_step132810.pth"
  Get-FileHash -Algorithm SHA256 ".\APRIL_CAUSAL_LM_20251030\april_vocab.json"
  Get-FileHash -Algorithm SHA256 ".\tools\docker_run_april_cli.ps1"

Tüm klasörün imzasını tek satırda doğrulamak isterseniz önce arşivleyin:
  Compress-Archive -Path ".\APRIL_CAUSAL_LM_20251030",".\tools" -DestinationPath ".\april_cli_bundle.zip" -Force
  Get-FileHash -Algorithm SHA256 ".\april_cli_bundle.zip"

Otomatik “manifest” (tüm dosyaların SHA‑256 listesi) üretmek için:
  Get-ChildItem -Recurse | Where-Object { -not $_.PSIsContainer } |
    ForEach-Object { Get-FileHash -Algorithm SHA256 $_.FullName } |
    ForEach-Object { "$($_.Algorithm) $($_.Hash) *$($_.Path)" } |
    Set-Content -Encoding ASCII ".\HASHES.txt"


--------------------------------------------------------------------
8) DOĞRULAMA AKIŞI (ÖNERİLEN)
--------------------------------------------------------------------
1. Docker kurulu ve “hello-world” çalışıyor mu kontrol edin.
2. “Hızlı Kurulum ve Çalıştırma” bölümündeki A veya B testini uygulayın.
3. Çıktı: APRIL_CAUSAL_LM_20251030\outputs\runs\...\out.txt içinde oluşur.
4. (İsteğe bağlı) BOM temizliği yapın, ardından HASHES.txt’yi üretin ve paketle birlikte verin.



