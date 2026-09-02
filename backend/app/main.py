"""SecureScan API entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, findings, scans
from app.config import get_settings
from app.db import init_db
from app.workers.queue import shutdown_queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("starting SecureScan (%s), model=%s", settings.environment, settings.model_version)
    if settings.environment == "development":
        init_db()
    else:
        # Migrations are applied by a deploy step, not by the API process:
        # several replicas starting at once must not race on schema changes.
        log.info("skipping init_db; expecting `alembic upgrade head` to have run")

    # Load the model at startup, not on the first scan, so a cold worker does
    # not make the first user wait for a multi-second load.
    if not settings.use_redis:
        try:
            from app.workers.scan_worker import get_predictor

            get_predictor(settings)
        except Exception:  # noqa: BLE001 - API still serves without a checkpoint
            log.warning(
                "no model at %s -- API will serve but scans will fail. "
                "Train one: python -m securescan_ml.training.train",
                settings.model_dir,
            )

    yield
    shutdown_queue()


app = FastAPI(
    title="SecureScan API",
    description="AI-powered Python code vulnerability detection",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # From SECURESCAN_CORS_ORIGINS. Settings refuse to start if this still
    # contains localhost outside development.
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(scans.router)
app.include_router(findings.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.environment,
        "model_version": settings.model_version,
        "queue": "redis" if settings.use_redis else "in-process",
    }


@app.get("/api/taxonomy", tags=["meta"])
def taxonomy() -> dict:
    """The label set, severities, and remediation templates.

    Served from the ML package so the frontend cannot drift from the model.
    """
    from securescan_ml.labels import LABEL_ORDER, REMEDIATIONS, severity_for

    return {
        "labels": [
            {
                "name": label.value,
                "severity": severity_for(label).value,
                "has_remediation": label in REMEDIATIONS,
            }
            for label in LABEL_ORDER
        ]
    }
