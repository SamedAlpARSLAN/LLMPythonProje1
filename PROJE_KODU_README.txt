

PROJE: APRIL_CAUSAL_LM – ÇALIŞTIRMA KILAVUZU (SÜRÜM 2025-10-30)

1. KLASÖR YAPISI (ZIP KÖKÜ)

* APRIL_CAUSAL_LM_20251030\

  * checkpoint\ → örn: april_step132810.pth
  * scripts\ → run_inference.py, requirements.txt, prompts.txt
  * (model kaynak dosyaları: april_*.py, tokenizer vb.)
  * outputs\ → çalıştırınca otomatik oluşur (runs...\out.txt)
* tools\

  * docker_run_april_cli.ps1
  * Dockerfile.cli
* april_container\  (GELİŞMİŞ/OPSİYONEL; çalışma için şart değil)
* payload.json      (meta bilgi)

Not: Değişken/ekstra klasör adları sizde farklıysa komutlardaki yolları aynı mantıkla güncelleyin.

2. SİSTEM GEREKSİNİMLERİ

* Windows 10/11 x64 (PowerShell 5+ veya PowerShell 7+)
* İnternet gerekmez (tümü yerel). Python/Docker seçenekli.
* GPU şart değil. Varsayılan çalıştırma CPU’dadır.

3. EN HIZLI BAŞLANGIÇ (PYTHON – DOCKER GEREKTİRMEZ)
   A. PowerShell’i zip’in KÖKÜNDE açın.
   B. Sanal ortam ve kurulum:

   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r .\APRIL_CAUSAL_LM_20251030\scripts\requirements.txt
   ```

   (Gerekirse PyTorch CPU için alternatif:
   `pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html`)

C. TEK SORU testi (greedy; tekrarları azaltır):

```
python .\APRIL_CAUSAL_LM_20251030\scripts\run_inference.py `
  --checkpoint .\APRIL_CAUSAL_LM_20251030\checkpoint\april_step132810.pth `
  --tokenizer  .\APRIL_CAUSAL_LM_20251030\april_vocab.json `
  --prompt "Türkiye'nin başkenti neresidir?" `
  --max_new_tokens 24 --temperature 0 --top_k 0 --top_p 1
```

Çıktı: `APRIL_CAUSAL_LM_20251030\outputs\runs\YYYYMMDD_HHMMSS\out.txt`

D. TOPLU TEST (prompts.txt):

```
python .\APRIL_CAUSAL_LM_20251030\scripts\run_inference.py `
  --checkpoint .\APRIL_CAUSAL_LM_20251030\checkpoint\april_step132810.pth `
  --tokenizer  .\APRIL_CAUSAL_LM_20251030\april_vocab.json `
  --prompt_file .\APRIL_CAUSAL_LM_20251030\scripts\prompts.txt `
  --max_new_tokens 24 --temperature 0 --top_k 0 --top_p 1
```

Not: `prompts.txt` içinde her satır ayrı bir girdi kabul edilir.

4. ALTERNATİF: DOCKER İLE ÇALIŞTIRMA (CPU)
   A. İmajı derleyin:

   ```
   docker build -t april-cli:cpu -f .\tools\Dockerfile.cli .
   ```

B. TEK SORU (greedy) veya TOPLU:

* Tek Soru:

  ```
  powershell -ExecutionPolicy Bypass -File .\tools\docker_run_april_cli.ps1 `
    -Ckpt .\APRIL_CAUSAL_LM_20251030\checkpoint\april_step132810.pth `
    -Tok  .\APRIL_CAUSAL_LM_20251030\april_vocab.json `
    -Prompt "Türkiye'nin başkenti neresidir?" `
    -GenArgs "--temperature","0","--top_p","1","--top_k","0","--max_new_tokens","24"
  ```
* Toplu (prompts.txt):

  ```
  powershell -ExecutionPolicy Bypass -File .\tools\docker_run_april_cli.ps1 `
    -Ckpt .\APRIL_CAUSAL_LM_20251030\checkpoint\april_step132810.pth `
    -Tok  .\APRIL_CAUSAL_LM_20251030\april_vocab.json
  ```

  (Script, `scripts\prompts.txt` dosyasını otomatik algılar.)

5. ÇIKTI NEREYE YAZILIR?

* Her çalıştırmada `APRIL_CAUSAL_LM_20251030\outputs\runs\YYYYMMDD_HHMMSS\out.txt` oluşturulur.
* İsterseniz `--out_txt` ile özel bir dosya yolu belirtebilirsiniz.

6. DOĞRULAMA / HIZLI KONTROL

* Örnek girdi: “Türkiye’nin başkenti neresidir?”
* Greedy ayarlarıyla kısa ve tutarlı bir yanıt beklenir. Çıktı dosyası boş ise ya da hatalı karakter doluysa “Sorun Giderme” bölümüne bakın.

7. SORUN GİDERME
   A. “invalid non-printable character U+FEFF”

   * Neden: UTF-8 BOM başlığı bulunan dosya.
   * Çözüm (PowerShell, kökten çalıştırın):

     ```
     Get-ChildItem ".\APRIL_CAUSAL_LM_20251030" -Recurse -Include *.py,*.sh,*.txt,*.md,*.json |
       % {
         $b = [IO.File]::ReadAllBytes($_.FullName)
         if ($b.Length -ge 3 -and $b[0]-eq239 -and $b[1]-eq187 -and $b[2]-eq191) {
           [IO.File]::WriteAllBytes($_.FullName, $b[3..($b.Length-1)])
         }
       }
     ```

     Kontrol:

     ```
     Get-ChildItem ".\APRIL_CAUSAL_LM_20251030" -Recurse -Include *.py,*.sh,*.txt,*.md,*.json |
       % {
         $b = [IO.File]::ReadAllBytes($_.FullName)
         if ($b.Length -ge 3 -and $b[0]-eq239 -and $b[1]-eq187 -and $b[2]-eq191) { $_.FullName }
       }
     ```

     Çıktı boşsa BOM kalmamıştır.

B. “Tokenizer class ... does not exist / HF veya vLLM ile yüklenemedi”

* Bu dağıtım, **özel tokenizer ve model sınıfları** ile gelir; HF/vLLM otomatik yükleyici gerektirmez.
* Lütfen bu README’deki **3. veya 4. bölümdeki** komutları kullanın (resmî değerlendirme yolu).
* vLLM’e taşımak isterseniz özel bir HF dönüştürme katmanı gerekir (dağıtıma dâhil değildir).

C. “usage: run_inference.py … unrecognized arguments …”

* Parametreleri aynen örneklerdeki gibi verin.
* PowerShell’de `-GenArgs "--temperature","0","--top_p","1","--top_k","0","--max_new_tokens","24"` biçimini kullanın.

D. Yollar/Türkçe karakterler

* PowerShell’de yolları **tırnak içinde** verin; gerekirse `-LiteralPath` tercih edin.

E. Çıktı tekrarlı/bozuk görünüyor

* Greedy kullanın: `--temperature 0 --top_k 0 --top_p 1`.
* `--max_new_tokens` değerini 16–32 aralığında deneyin.

8. ZIP OLUŞTURMA VE SHA-256 DOĞRULAMASI (TESLİM İÇİN)
   A. Zip (Windows, bir üst klasörde):

   ```
   $Base = "C:\...\submission\gecici klasor"
   $FolderName = "gönderilecek april"
   $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
   $Src = Join-Path $Base $FolderName
   $Zip = Join-Path $Base ("{0}_{1}.zip" -f ($FolderName -replace ' ','_'), $Stamp)
   if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force }
   Compress-Archive -LiteralPath $Src -DestinationPath $Zip -Force
   ```

B. SHA-256:

```
(Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash
```

(Linux/macOS: `sha256sum <dosya>.zip`)

9. SIKÇA SORULANLAR

* İnternet gerekli mi? Hayır. Tüm dosyalar ziptedir.
* GPU şart mı? Hayır. Bu kılavuz CPU üzerinde çalıştırır.
* Çıktı nereye düşer? Bkz. Bölüm 5.
* promptları toplu nasıl denerim? Bkz. Bölüm 3-D veya 4-B.

10. İLETİŞİM / SÜRÜM

* Sürüm: 2025-10-30, Paket: APRIL_CAUSAL_LM_20251030
* Not: Değerlendirme kolaylığı için **3. (Python) veya 4. (Docker/CLI)** yolunu kullanmanız tavsiye edilir.


