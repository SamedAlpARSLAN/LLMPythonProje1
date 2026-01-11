# APRIL â€” OpenAI API (vLLM) Container
Build:
  docker build -t april-openai -f april_container/Dockerfile april_container
Run (PowerShell, tek satÄ±r):
  $ModelHostPath = (Resolve-Path ".\APRIL_CAUSAL_LM").Path
  docker run --rm -it --gpus all -p 8010:8000 -v "${ModelHostPath}:/app/models/APRIL_CAUSAL_LM:ro" --name april-openai april-openai
Test:
  curl.exe http://localhost:8010/v1/models