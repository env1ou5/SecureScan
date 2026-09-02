"""Findings listing, remediation, and false-positive dismissal."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from securescan_ml.labels import Label, remediation_for
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Finding, Scan, User
from app.schemas import DismissRequest, FindingOut, RemediationOut
from app.security import get_current_user

router = APIRouter(prefix="/api", tags=["findings"])


def _attach_remediation(finding: Finding) -> FindingOut:
    """Look up the templated fix for this class.

    Templates live in the ML package's label module so the taxonomy, the
    severity map, and the remediation text can never disagree.
    """
    out = FindingOut.model_validate(finding)
    try:
        rem = remediation_for(Label(finding.vulnerability_type))
    except ValueError:  # a label retired since this finding was written
        rem = None
    if rem is not None:
        out.remediation = RemediationOut(
            title=rem.title,
            explanation=rem.explanation,
            unsafe_example=rem.unsafe_example,
            safe_example=rem.safe_example,
        )
    return out


def _owned_finding(finding_id: str, user: User, db: Session) -> Finding:
    finding = (
        db.query(Finding)
        .join(Scan, Finding.scan_id == Scan.id)
        .filter(Finding.id == finding_id, Scan.user_id == user.id)
        .first()
    )
    if finding is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finding not found")
    return finding


@router.get("/scans/{scan_id}/findings", response_model=list[FindingOut])
def list_findings(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    severity: str | None = Query(None),
    vulnerability_type: str | None = Query(None),
    file_path: str | None = Query(None),
    include_dismissed: bool = Query(False),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[FindingOut]:
    # Join through Scan so ownership is enforced in the same query.
    query = (
        db.query(Finding)
        .join(Scan, Finding.scan_id == Scan.id)
        .filter(Finding.scan_id == scan_id, Scan.user_id == user.id)
    )
    if not include_dismissed:
        query = query.filter(Finding.dismissed.is_(False))
    if severity:
        query = query.filter(Finding.severity == severity.upper())
    if vulnerability_type:
        query = query.filter(Finding.vulnerability_type == vulnerability_type.upper())
    if file_path:
        query = query.filter(Finding.file_path == file_path)

    findings = query.order_by(Finding.confidence.desc()).limit(limit).offset(offset).all()
    return [_attach_remediation(f) for f in findings]


@router.get("/findings/{finding_id}", response_model=FindingOut)
def get_finding(
    finding_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FindingOut:
    return _attach_remediation(_owned_finding(finding_id, user, db))


@router.post("/findings/{finding_id}/dismiss", response_model=FindingOut)
def dismiss_finding(
    finding_id: str,
    payload: DismissRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FindingOut:
    """Mark a finding as a false positive.

    Dismissals are the raw material for the feedback loop in §14: they are
    labeled examples of exactly where the model is wrong.
    """
    finding = _owned_finding(finding_id, user, db)
    finding.dismissed = True
    finding.dismissed_reason = payload.reason
    db.commit()
    db.refresh(finding)
    return _attach_remediation(finding)
