# CPU-only inference image (proposal D4). No CUDA base, no GPU at serve time.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so source edits do not invalidate the layer.
COPY backend/requirements.txt /srv/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend /srv/backend
COPY ml/securescan_ml /srv/securescan_ml

# Baked-in checkpoint. For frequent model updates, mount it or pull from S3 at
# startup instead of rebuilding the image.
COPY artifacts /srv/artifacts

ENV PYTHONPATH=/srv/backend:/srv

# Never run as root: this process unpacks untrusted archives.
RUN useradd --create-home --uid 10001 securescan \
    && chown -R securescan:securescan /srv \
    && mkdir -p /opt/hf-cache && chown securescan /opt/hf-cache
USER securescan

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
