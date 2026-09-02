"""The scan job: archive in, findings in PostgreSQL out (proposal §7).

Runs in a worker thread (dev) or an RQ worker process (production). Opens its
own database session because it outlives the request that enqueued it.

The model is a module-level singleton: loading UniXcoder takes seconds, and a
worker that reloads it per job would spend all its time doing that.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from threading import Lock

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import SessionLocal
from app.models import Finding, Scan, ScanStatus
from app.services.ingest import UnsafeArchiveError, extract_archive, read_source
from app.services.storage import get_storage

log = logging.getLogger(__name__)

_predictor = None
_predictor_lock = Lock()


def get_predictor(settings: Settings | None = None):
    """Load the INT8 predictor once per process."""
    global _predictor
    settings = settings or get_settings()
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                from securescan_ml.inference.predictor import VulnerabilityPredictor

                started = time.perf_counter()
                _predictor = VulnerabilityPredictor(
                    settings.model_dir, max_length=settings.max_length
                )
                warm = _predictor.warmup()
                log.info(
                    "model loaded in %.1fs (warmup %.3fs) from %s",
                    time.perf_counter() - started,
                    warm,
                    settings.model_dir,
                )
    return _predictor


def run_scan(scan_id: str, storage_key: str) -> None:
    """Entry point enqueued by POST /api/scans.

    Takes ids rather than objects so the same signature works over RQ, where
    arguments must be serializable.
    """
    settings = get_settings()
    db: Session = SessionLocal()
    started = time.perf_counter()

    scan = db.get(Scan, scan_id)
    if scan is None:
        log.error("scan %s vanished before it could run", scan_id)
        db.close()
        return

    storage = get_storage(settings)
    try:
        scan.status = ScanStatus.RUNNING
        scan.started_at = datetime.now(UTC)
        db.commit()

        archive_path = storage.local_path(storage_key)
        with extract_archive(archive_path, settings, scan_id) as extracted:
            _scan_files(db, scan, extracted, settings)

        scan.status = ScanStatus.COMPLETED

    except UnsafeArchiveError as exc:
        # User-facing: they uploaded something we refused, and should know why.
        log.warning("scan %s rejected: %s", scan_id, exc)
        scan.status = ScanStatus.FAILED
        scan.error = str(exc)
    except Exception as exc:  # noqa: BLE001
        # Internal: log the trace, but do not leak it into the API response.
        log.exception("scan %s failed", scan_id)
        scan.status = ScanStatus.FAILED
        scan.error = f"Internal error during scan ({type(exc).__name__})"
    finally:
        scan.completed_at = datetime.now(UTC)
        scan.duration_seconds = time.perf_counter() - started
        db.commit()
        log.info(
            "scan %s %s in %.1fs (%d findings)",
            scan_id,
            scan.status.value,
            scan.duration_seconds,
            scan.findings_count,
        )
        # The upload has served its purpose. Do not keep other people's source.
        try:
            storage.delete(storage_key)
        except Exception:  # noqa: BLE001
            log.exception("could not delete upload %s", storage_key)
        db.close()


def _scan_files(db: Session, scan: Scan, extracted, settings: Settings) -> None:
    from securescan_ml.chunking import active_backend, extract_analyzable_chunks
    from securescan_ml.labels import Label

    predictor = get_predictor(settings)
    scan.parser_backend = active_backend()

    files_scanned = 0
    functions_scanned = 0
    findings_count = 0
    seen: set[tuple[str, str, int]] = set()

    for path in extracted.python_files:
        source = read_source(path)
        if source is None:
            continue
        files_scanned += 1
        relative = str(path.relative_to(extracted.root))

        chunks = extract_analyzable_chunks(source, relative)
        if not chunks:
            continue
        functions_scanned += len(chunks)

        predictions = predictor.predict_chunks(chunks, batch_size=settings.inference_batch_size)

        for pred in predictions:
            if pred.label is Label.SAFE:
                continue
            # Low-confidence findings are dropped, not shown greyed out. A
            # scanner that cries wolf gets uninstalled (§10).
            if pred.confidence < settings.min_confidence:
                continue

            # The unique constraint covers this too, but catching it here
            # avoids aborting the whole transaction on a duplicate.
            key = (relative, pred.function_name, pred.start_line)
            if key in seen:
                continue
            seen.add(key)

            db.add(
                Finding(
                    scan_id=scan.id,
                    file_path=relative,
                    function_name=pred.function_name,
                    vulnerability_type=pred.label.value,
                    severity=pred.severity.value,
                    confidence=pred.confidence,
                    confidence_calibrated=pred.calibrated,
                    start_line=pred.start_line,
                    end_line=pred.end_line,
                    anchor_line=pred.anchor_line,
                    contributing_lines=[
                        {"line": a.line, "score": round(a.score, 4), "text": a.text}
                        for a in pred.contributing_lines
                    ],
                    probabilities={k: round(v, 4) for k, v in pred.probabilities.items()},
                    code_snippet=_snippet(source, pred.start_line, pred.end_line),
                )
            )
            findings_count += 1

        # Commit per file so a long scan shows progress and a late failure does
        # not discard everything found so far.
        scan.files_scanned = files_scanned
        scan.functions_scanned = functions_scanned
        scan.findings_count = findings_count
        db.commit()


def _snippet(source: str, start_line: int, end_line: int, max_lines: int = 40) -> str:
    """Keep the offending function only -- never the whole file."""
    lines = source.splitlines()
    excerpt = lines[start_line - 1 : min(end_line, start_line - 1 + max_lines)]
    if end_line - start_line + 1 > max_lines:
        excerpt.append(f"... ({end_line - start_line + 1 - max_lines} more lines)")
    return "\n".join(excerpt)
