"""SQLAlchemy models.

Every scan belongs to a user, and every findings query filters on that user.
Tenant isolation is enforced at the query layer, not by the URL -- an id in a
path is not authorization.

Uploaded source is NOT stored. Findings keep the offending snippet and its line
range; the archive and its extraction directory are deleted when the job ends.
Holding other people's proprietary code is a liability with no upside here.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# JSONB on PostgreSQL, plain JSON elsewhere, so the suite can run on SQLite
# without a database server while production still gets the indexable type.
JsonType = JSON().with_variant(JSONB(), "postgresql")


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class ScanStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    scans: Mapped[list[Scan]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    repository_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ScanStatus] = mapped_column(
        Enum(ScanStatus, native_enum=False), default=ScanStatus.QUEUED, index=True
    )

    # Provenance: which model produced these findings. Without it, comparing
    # scans across model versions is meaningless.
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_backend: Mapped[str | None] = mapped_column(String(32))

    files_scanned: Mapped[int] = mapped_column(Integer, default=0)
    functions_scanned: Mapped[int] = mapped_column(Integer, default=0)
    findings_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="scans")
    findings: Mapped[list[Finding]] = relationship(
        back_populates="scan", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_scans_user_created", "user_id", "created_at"),)


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    scan_id: Mapped[str] = mapped_column(
        ForeignKey("scans.id", ondelete="CASCADE"), index=True, nullable=False
    )

    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    function_name: Mapped[str] = mapped_column(String(255), nullable=False)
    vulnerability_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), index=True, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # False when the checkpoint had no fitted temperature. The dashboard must
    # say so rather than presenting a raw softmax as a probability.
    confidence_calibrated: Mapped[bool] = mapped_column(default=True)

    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    anchor_line: Mapped[int] = mapped_column(Integer, nullable=False)

    # [{line, score, text}] from gradient attribution.
    contributing_lines: Mapped[list] = mapped_column(JsonType, default=list)
    # Full class distribution, for false-positive analysis after the fact.
    probabilities: Mapped[dict] = mapped_column(JsonType, default=dict)
    code_snippet: Mapped[str | None] = mapped_column(Text)

    # Set by a user marking a finding wrong. Feeds dataset improvement (§14).
    dismissed: Mapped[bool] = mapped_column(default=False, index=True)
    dismissed_reason: Mapped[str | None] = mapped_column(String(512))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    scan: Mapped[Scan] = relationship(back_populates="findings")

    __table_args__ = (
        Index("ix_findings_scan_severity", "scan_id", "severity"),
        UniqueConstraint(
            "scan_id",
            "file_path",
            "function_name",
            "start_line",
            name="uq_finding_location",
        ),
    )
