# Scan worker. Same image contents as the API, different entry point, so the
# two always run identical model and parsing code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /srv/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend /srv/backend
COPY ml/securescan_ml /srv/securescan_ml
COPY artifacts /srv/artifacts

ENV PYTHONPATH=/srv/backend:/srv

RUN useradd --create-home --uid 10001 securescan \
    && chown -R securescan:securescan /srv \
    && mkdir -p /opt/hf-cache && chown securescan /opt/hf-cache
USER securescan

CMD ["python", "-m", "app.workers.rq_worker"]
