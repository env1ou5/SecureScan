"""Scan submission and status (proposal §2, D3).

POST returns 202 with a scan id immediately; the client polls GET. This is the
contract from the first commit so that swapping the in-process worker for RQ
later touches nothing outside workers/queue.py.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.models import Finding, Scan, ScanStatus, User
from app.schemas import (
    FileCount,
    ScanAccepted,
    ScanOut,
    ScanSummary,
    SeverityCount,
    TypeCount,
)
from app.security import get_current_user
from app.services.storage import get_storage, upload_key
from app.workers.queue import get_queue
from app.workers.scan_worker import run_scan

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scans", tags=["scans"])

SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}


def _owned_scan(scan_id: str, user: User, db: Session) -> Scan:
    """Fetch a scan, scoped to its owner.

    Filtering on user_id rather than checking after the fetch means an id from
    another tenant is indistinguishable from one that does not exist.
    """
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.user_id == user.id).first()
    if scan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")
    return scan


@router.post("", response_model=ScanAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_scan(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ScanAccepted:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    scan = Scan(
        user_id=user.id,
        repository_name=file.filename[:255],
        model_version=settings.model_version,
        status=ScanStatus.QUEUED,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    key = upload_key(scan.id, file.filename)
    try:
        get_storage(settings).save(file.file, key)
    except Exception as exc:  # noqa: BLE001
        log.exception("upload failed for scan %s", scan.id)
        scan.status = ScanStatus.FAILED
        scan.error = "Upload could not be stored"
        db.commit()
        raise HTTPException(status_code=500, detail="Upload failed") from exc

    # Archive validation happens in the worker, not here: a zip bomb should not
    # be unpacked inside the request path.
    get_queue(settings).enqueue(run_scan, scan.id, key)

    return ScanAccepted(
        scan_id=scan.id,
        status=scan.status,
        status_url=f"/api/scans/{scan.id}",
    )


@router.get("", response_model=list[ScanOut])
def list_scans(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[Scan]:
    return (
        db.query(Scan)
        .filter(Scan.user_id == user.id)
        .order_by(Scan.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )


@router.get("/{scan_id}", response_model=ScanOut)
def get_scan(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Scan:
    return _owned_scan(scan_id, user, db)


@router.get("/{scan_id}/summary", response_model=ScanSummary)
def get_summary(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ScanSummary:
    """Aggregates for the dashboard, computed in the database.

    Dismissed findings are excluded everywhere so the counts match what the
    user actually sees.
    """
    scan = _owned_scan(scan_id, user, db)
    active = (Finding.scan_id == scan.id, Finding.dismissed.is_(False))

    by_severity = (
        db.query(Finding.severity, func.count(Finding.id))
        .filter(*active)
        .group_by(Finding.severity)
        .all()
    )
    by_type = (
        db.query(Finding.vulnerability_type, func.count(Finding.id))
        .filter(*active)
        .group_by(Finding.vulnerability_type)
        .all()
    )
    by_file = (
        db.query(Finding.file_path, func.count(Finding.id))
        .filter(*active)
        .group_by(Finding.file_path)
        .order_by(func.count(Finding.id).desc())
        .limit(50)
        .all()
    )
    mean_confidence = db.query(func.avg(Finding.confidence)).filter(*active).scalar()

    worst: dict[str, str] = {}
    for path, severity in db.query(Finding.file_path, Finding.severity).filter(*active).all():
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(worst.get(path, "NONE"), 0):
            worst[path] = severity

    return ScanSummary(
        scan=ScanOut.model_validate(scan),
        by_severity=sorted(
            (SeverityCount(severity=s, count=c) for s, c in by_severity),
            key=lambda x: SEVERITY_RANK.get(x.severity, 0),
            reverse=True,
        ),
        by_type=[TypeCount(vulnerability_type=t, count=c) for t, c in by_type],
        by_file=[
            FileCount(file_path=p, count=c, highest_severity=worst.get(p, "NONE"))
            for p, c in by_file
        ],
        total_findings=sum(c for _, c in by_severity),
        mean_confidence=float(mean_confidence) if mean_confidence is not None else None,
    )


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    db.delete(_owned_scan(scan_id, user, db))
    db.commit()
