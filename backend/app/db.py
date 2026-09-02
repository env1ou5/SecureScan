"""Database engine, session factory, and the FastAPI session dependency."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    pool_pre_ping=True,
    echo=_settings.debug,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables directly from metadata.

    Development and tests only. Every other environment applies migrations:

        cd backend && alembic upgrade head

    `create_all` never drops or alters, so running it against a database that
    has drifted from the models leaves the drift in place silently. Production
    startup does not call this (see main.py).
    """
    from app import models  # noqa: F401 - registers mappers

    Base.metadata.create_all(bind=engine)
