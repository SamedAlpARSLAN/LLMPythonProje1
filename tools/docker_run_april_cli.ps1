param(
  [string]$Ckpt = "",
  [string]$Tok  = "",
  [string]$Prompt = "",
  # Varsayılan: greedy (tekrarları azaltır)
  [object[]]$GenArgs = @("--temperature","0","--top_p","1","--top_k","0","--max_new_tokens","32")
)

$ErrorActionPreference = "Stop"

# PS 7 yoksa $PSStyle olmayabilir – sessiz geç
try {
  if ($PSVersionTable.PSVersion.Major -ge 7) {
    if ($PSStyle -and ($PSStyle | Get-Member -Name OutputRendering -MemberType Properties)) {
      $PSStyle.OutputRendering = 'Ansi'
    } else { $env:NO_COLOR = '1' }
  } else { $env:NO_COLOR = '1' }
} catch { $env:NO_COLOR = '1' }

function Pick-ModelRoot {
  $candidate = Get-ChildItem -Directory -Name |
               Where-Object { $_ -like 'APRIL_CAUSAL_LM_*' } |
               Sort-Object -Descending |
               Select-Object -First 1
  if (-not $candidate) { throw "APRIL_CAUSAL_LM_* klasörü bulunamadı." }
  return (Resolve-Path $candidate).Path
}

function Pick-Checkpoint([string]$Root) {
  $dirs = @()
  $p1 = Join-Path $Root "checkpoint"
  $p2 = Join-Path $Root "checkpoints"
  if (Test-Path $p1) { $dirs += $p1 }
  if (Test-Path $p2) { $dirs += $p2 }
  foreach ($d in $dirs) {
    $f = Get-ChildItem -Path $d -Filter *.pth -File -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending |
         Select-Object -First 1
    if ($f) { return $f.FullName }
  }
  return $null
}

function Pick-Tokenizer([string]$Root) {
  foreach ($p in @(
    (Join-Path $Root "tokenizer.json"),
    (Join-Path $Root "april_vocab.json"),
    (Join-Path $Root "vocab.json")
  )) {
    if (Test-Path $p) { return (Resolve-Path $p).Path }
  }
  return $null
}

$ModelRoot = Pick-ModelRoot

if ([string]::IsNullOrWhiteSpace($Ckpt)) { $Ckpt = Pick-Checkpoint $ModelRoot }
if (-not $Ckpt) { throw "[!] Checkpoint (.pth) bulunamadı. -Ckpt ile belirtin." }
$Ckpt = (Resolve-Path $Ckpt).Path

if ([string]::IsNullOrWhiteSpace($Tok)) { $Tok = Pick-Tokenizer $ModelRoot }
if (-not $Tok) { throw "[!] Tokenizer/Vocab (.json) bulunamadı. -Tok ile belirtin." }
$Tok = (Resolve-Path $Tok).Path

Write-Host "==> Seçilen dosyalar:"
Write-Host "    CKPT(host): $Ckpt"
Write-Host "    TOK (host): $Tok"

# --- CLI imajını derle ---
$Image = "april-cli:cpu"
$DockerfileCli = (Resolve-Path ".\tools\Dockerfile.cli").Path
docker build --no-cache -t $Image -f $DockerfileCli .

# --- Çalıştır ---
$Work = (Resolve-Path ".").Path

# Container içindeki sabit path'ler
$CkptCont = "/app/model/checkpoint.pth"
$TokCont  = "/app/model/tokenizer.json"

# Komut + zorunlu argümanlar
$runArgs = @(
  "run","--rm",
  "-v","${Work}:/work",
  "-v","${Ckpt}:${CkptCont}:ro",
  "-v","${Tok}:${TokCont}:ro",
  $Image,
  "python","/work/APRIL_CAUSAL_LM_20251030/scripts/run_inference.py",
  "--checkpoint",$CkptCont,
  "--tokenizer",$TokCont
)

# --- Prompt kaynağı ---
if ($Prompt -and $Prompt.Trim()) {
  $runArgs += @("--prompt", $Prompt)
} elseif (Test-Path (Join-Path $ModelRoot "scripts\prompts.txt")) {
  $runArgs += @("--prompt_file","/work/APRIL_CAUSAL_LM_20251030/scripts/prompts.txt")
} else {
  Write-Warning "Prompt bulunamadı (-Prompt ver ya da scripts\prompts.txt oluştur)."
}

# --- GenArgs normalizasyonu (virgülle yapışan tek argümanı ayır) ---
$normGen = @()
if ($GenArgs -and $GenArgs.Count -gt 0) {
  foreach ($g in $GenArgs) {
    if ($null -eq $g) { continue }
    $s = [string]$g
    if ($s -match ",") {
      $normGen += ($s -split "\s*,\s*")
    } else {
      $normGen += $s
    }
  }
}
if ($normGen.Count -gt 0) { $runArgs += $normGen }

# Çalıştır
& docker @runArgs
