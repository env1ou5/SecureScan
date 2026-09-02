"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import ScanStatus


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ContributingLine(BaseModel):
    line: int
    score: float
    text: str = ""


class RemediationOut(BaseModel):
    title: str
    explanation: str
    unsafe_example: str
    safe_example: str


class FindingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_path: str
    function_name: str
    vulnerability_type: str
    severity: str
    confidence: float
    confidence_calibrated: bool
    start_line: int
    end_line: int
    anchor_line: int
    contributing_lines: list[ContributingLine] = []
    code_snippet: str | None = None
    dismissed: bool = False
    remediation: RemediationOut | None = None


class ScanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    repository_name: str
    status: ScanStatus
    model_version: str
    files_scanned: int
    functions_scanned: int
    findings_count: int
    duration_seconds: float | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class ScanAccepted(BaseModel):
    """202 response. The scan has not run yet -- poll `status_url`."""

    scan_id: str
    status: ScanStatus
    status_url: str


class SeverityCount(BaseModel):
    severity: str
    count: int


class TypeCount(BaseModel):
    vulnerability_type: str
    count: int


class FileCount(BaseModel):
    file_path: str
    count: int
    highest_severity: str


class ScanSummary(BaseModel):
    """Dashboard payload for one scan."""

    scan: ScanOut
    by_severity: list[SeverityCount]
    by_type: list[TypeCount]
    by_file: list[FileCount]
    total_findings: int
    mean_confidence: float | None = None


class DismissRequest(BaseModel):
    reason: str = Field(max_length=512)
