"""Object storage for uploads and model checkpoints.

Local filesystem in development, S3 in production, behind one interface so the
worker never learns which is live. Uploaded archives are transient: stored only
until the scan finishes, then deleted.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from app.config import Settings

log = logging.getLogger(__name__)


class Storage(ABC):
    @abstractmethod
    def save(self, fileobj: BinaryIO, key: str) -> str: ...

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """Materialize the object locally; the worker needs a real path."""

    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalStorage(Storage):
    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        target = (self.base / key).resolve()
        if not target.is_relative_to(self.base.resolve()):
            raise ValueError(f"storage key escapes base directory: {key!r}")
        return target

    def save(self, fileobj: BinaryIO, key: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as out:
            shutil.copyfileobj(fileobj, out)
        return str(path)

    def local_path(self, key: str) -> Path:
        return self._path(key)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Storage(Storage):  # pragma: no cover - requires AWS
    def __init__(self, bucket: str, region: str):
        import boto3

        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region)
        self._temp_dir = Path("/tmp/securescan-s3-cache")
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fileobj: BinaryIO, key: str) -> str:
        self.client.upload_fileobj(fileobj, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def local_path(self, key: str) -> Path:
        target = self._temp_dir / key.replace("/", "_")
        if not target.exists():
            self.client.download_file(self.bucket, key, str(target))
        return target

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)
        (self._temp_dir / key.replace("/", "_")).unlink(missing_ok=True)


def get_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("SECURESCAN_S3_BUCKET is required when storage_backend=s3")
        return S3Storage(settings.s3_bucket, settings.aws_region)
    return LocalStorage(settings.local_storage_dir)


def upload_key(scan_id: str, filename: str) -> str:
    """Namespaced, unguessable key. The client filename never reaches the path."""
    suffix = "".join(c for c in Path(filename).suffix if c.isalnum() or c == ".")[:16]
    return f"uploads/{scan_id}/{uuid.uuid4().hex}{suffix}"
