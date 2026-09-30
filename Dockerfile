# CAT-GraphRAG-Proctoring server (FastAPI). Build:  docker compose up --build
FROM python:3.11-slim

# OpenCV / MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /srv/backend/
COPY frontend/ /srv/frontend/

# models (MediaPipe, Whisper) and evidence live on volumes so they survive rebuilds
VOLUME ["/srv/backend/models", "/srv/backend/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
