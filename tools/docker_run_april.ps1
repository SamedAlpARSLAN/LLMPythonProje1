# tools/docker_run_april.ps1
$ErrorActionPreference = "Stop"

# 1) Yol ayarları
$ContainerPath = (Resolve-Path ".\april_container").Path
$ModelHostPath = (Resolve-Path ".\APRIL_CAUSAL_LM_20251030").Path
$ImageName = "april-openai"
$ContainerName = "april-openai"

# 2) Image build
Write-Host "==> Build: $ImageName"
docker build -t $ImageName -f (Join-Path $ContainerPath "Dockerfile") $ContainerPath

# 3) Temiz çalıştır
Write-Host "==> Run"
docker rm -f $ContainerName 2>$null | Out-Null
docker run -d `
  -p 8010:8000 `
  -e CPU_DRY_RUN=1 `
  -v "$ModelHostPath:/app/models/APRIL_CAUSAL_LM:ro" `
  --name $ContainerName `
  $ImageName | Out-Null

Start-Sleep -Seconds 2

# 4) Sağlık
curl.exe -s http://localhost:8010/v1/models

# 5) Hızlı test (PowerShell güvenli JSON)
$payload = @{
  model    = "APRIL_CAUSAL_LM"
  messages = @(@{ role = "user"; content = "Merhaba, April! Tek cümlelik bir selamlasın." })
  max_tokens = 16
} | ConvertTo-Json -Depth 5

$resp = Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8010/v1/chat/completions" `
  -ContentType "application/json" `
  -Body $payload

$resp.choices[0].message.content
