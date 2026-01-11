FROM python:3.12-slim

# Temel bağımlılıklar
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# PyPI paketleri
RUN pip install --no-cache-dir numpy==1.26.4 fastapi uvicorn[standard] "pydantic<3"
# Torch (CPU)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

WORKDIR /app
COPY server.py /app/server.py

# Olası BOM’u temizle (Windows’ta dosya oluşturduysan güvenlik için)
RUN python - << 'PY'\nimport pathlib; p=pathlib.Path("server.py"); b=p.read_bytes(); \n\
    p.write_bytes(b[3:] if b.startswith(b"\\xef\\xbb\\xbf") else b)\nPY

EXPOSE 8000
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]
