"""Shared fixtures.

Environment variables are set at import time, before anything under `app` is
imported. `app.db` builds its engine at module scope, so configuring settings
later would leave the worker's `SessionLocal` bound to a stale engine -- the
tests would pass against one database while the worker wrote to another.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

# Must happen before the first `import app.*` anywhere in the suite.
_TEST_DIR = Path(tempfile.mkdtemp(prefix="securescan-tests-"))
os.environ["SECURESCAN_DATABASE_URL"] = f"sqlite+pysqlite:///{_TEST_DIR / 'test.db'}"
os.environ["SECURESCAN_ENVIRONMENT"] = "development"
os.environ["SECURESCAN_LOCAL_STORAGE_DIR"] = str(_TEST_DIR / "storage")
os.environ["SECURESCAN_JWT_SECRET"] = "test-secret-not-used-in-production-abc123"
os.environ["SECURESCAN_MODEL_DIR"] = str(_TEST_DIR / "no-such-model")


@pytest.fixture
def settings():
    from app.config import Settings

    return Settings(
        max_upload_bytes=10 * 1024 * 1024,
        max_uncompressed_bytes=5 * 1024 * 1024,
        max_archive_members=100,
        max_compression_ratio=50.0,
        max_file_bytes=512 * 1024,
    )


@pytest.fixture
def sample_source() -> str:
    return '''import subprocess


def safe_add(a, b):
    """No vulnerability here."""
    return a + b


def lookup_user(user_id):
    query = "SELECT * FROM users WHERE id=" + user_id
    return cursor.execute(query)


def ping(host):
    return subprocess.run(f"ping {host}", shell=True)
'''
