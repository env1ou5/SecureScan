"""Hardened archive extraction (proposal §7).

Uploaded archives are untrusted input from authenticated but unvetted users.
Every check here corresponds to a real attack:

  * absolute or `..` member paths       -> write outside the extraction root
  * symlink / hardlink members          -> escape the root after extraction
  * huge uncompressed size              -> disk exhaustion
  * extreme compression ratio           -> zip bomb
  * enormous member count               -> inode exhaustion, scan never ends

Sizes are checked against the header BEFORE extracting, and again against what
actually landed on disk, because a zip header is attacker-controlled and can lie.

Everything extracts into a per-scan temp directory that is deleted when the job
finishes. Source code is never persisted (see models.py).
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings

log = logging.getLogger(__name__)

PYTHON_SUFFIXES = frozenset({".py", ".pyi"})

SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        ".eggs",
        "build",
        "dist",
    }
)


class UnsafeArchiveError(ValueError):
    """Archive violates a safety limit. Message is surfaced to the user."""


@dataclass
class ExtractionResult:
    root: Path
    python_files: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_bytes: int = 0

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> ExtractionResult:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


def _is_safe_member_path(name: str) -> bool:
    """Reject absolute paths, drive letters, and any traversal component."""
    if not name or name.startswith(("/", "\\")):
        return False
    if len(name) > 1 and name[1] == ":":  # C:\... on a Windows-authored archive
        return False
    parts = Path(name.replace("\\", "/")).parts
    return ".." not in parts


def _should_skip(name: str) -> bool:
    parts = Path(name.replace("\\", "/")).parts
    return any(part in SKIP_DIRECTORIES for part in parts)


def _check_zip(archive: zipfile.ZipFile, settings: Settings) -> None:
    infos = archive.infolist()
    if len(infos) > settings.max_archive_members:
        raise UnsafeArchiveError(
            f"archive has {len(infos)} members, limit is {settings.max_archive_members}"
        )

    total_uncompressed = 0
    total_compressed = 0
    for info in infos:
        if not _is_safe_member_path(info.filename):
            raise UnsafeArchiveError(f"unsafe member path: {info.filename!r}")
        # Unix mode is in the top 16 bits of external_attr; 0o120000 is a symlink.
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise UnsafeArchiveError(f"archive contains a symlink: {info.filename!r}")
        total_uncompressed += info.file_size
        total_compressed += info.compress_size

    if total_uncompressed > settings.max_uncompressed_bytes:
        raise UnsafeArchiveError(
            f"uncompressed size {total_uncompressed} exceeds {settings.max_uncompressed_bytes}"
        )
    if total_compressed > 0:
        ratio = total_uncompressed / total_compressed
        if ratio > settings.max_compression_ratio:
            raise UnsafeArchiveError(
                f"compression ratio {ratio:.1f}x exceeds "
                f"{settings.max_compression_ratio}x (possible zip bomb)"
            )


def _check_tar(archive: tarfile.TarFile, settings: Settings) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if len(members) > settings.max_archive_members:
        raise UnsafeArchiveError(
            f"archive has {len(members)} members, limit is {settings.max_archive_members}"
        )
    total = 0
    for member in members:
        if not _is_safe_member_path(member.name):
            raise UnsafeArchiveError(f"unsafe member path: {member.name!r}")
        if member.issym() or member.islnk():
            raise UnsafeArchiveError(f"archive contains a link: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise UnsafeArchiveError(f"unsupported member type: {member.name!r}")
        total += member.size
    if total > settings.max_uncompressed_bytes:
        raise UnsafeArchiveError(
            f"uncompressed size {total} exceeds {settings.max_uncompressed_bytes}"
        )
    return members


def extract_archive(
    archive_path: str | Path,
    settings: Settings,
    scan_id: str = "scan",
) -> ExtractionResult:
    """Validate and extract an archive, returning the Python files inside.

    Use as a context manager so the extraction directory is always removed:

        with extract_archive(path, settings, scan_id) as extracted:
            ...
    """
    archive_path = Path(archive_path)
    size = archive_path.stat().st_size
    if size > settings.max_upload_bytes:
        raise UnsafeArchiveError(f"upload is {size} bytes, limit is {settings.max_upload_bytes}")

    root = Path(tempfile.mkdtemp(prefix=f"securescan-{scan_id}-"))
    result = ExtractionResult(root=root)

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                _check_zip(zf, settings)
                for info in zf.infolist():
                    if info.is_dir() or _should_skip(info.filename):
                        continue
                    zf.extract(info, root)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                members = _check_tar(tf, settings)
                keep = [m for m in members if not _should_skip(m.name)]
                # filter="data" refuses links and absolute paths in the stdlib
                # too; belt and braces with _check_tar above.
                tf.extractall(root, members=keep, filter="data")
        else:
            raise UnsafeArchiveError("unsupported archive format (expected zip or tar)")

        _collect_python_files(result, settings)
    except UnsafeArchiveError:
        result.cleanup()
        raise
    except Exception as exc:  # noqa: BLE001
        result.cleanup()
        raise UnsafeArchiveError(f"could not read archive: {exc}") from exc

    if not result.python_files:
        result.cleanup()
        raise UnsafeArchiveError("archive contains no Python files")

    return result


def _collect_python_files(result: ExtractionResult, settings: Settings) -> None:
    """Walk the extraction root, verifying what actually landed on disk.

    Re-checks containment via resolve(): archive headers are attacker-controlled
    and the pre-flight check trusted them.
    """
    root = result.root.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            result.skipped.append(f"{path.name}: symlink")
            path.unlink(missing_ok=True)
            continue
        if not path.is_file() or path.suffix not in PYTHON_SUFFIXES:
            continue
        if not path.resolve().is_relative_to(root):
            result.skipped.append(f"{path.name}: escapes extraction root")
            continue
        rel = str(path.relative_to(root))
        if _should_skip(rel):
            continue

        file_size = path.stat().st_size
        if file_size > settings.max_file_bytes:
            result.skipped.append(f"{rel}: {file_size} bytes exceeds per-file limit")
            continue

        result.total_bytes += file_size
        if result.total_bytes > settings.max_uncompressed_bytes:
            raise UnsafeArchiveError("extracted size exceeded limit during walk")
        result.python_files.append(path)


def read_source(path: Path) -> str | None:
    """Read a source file, returning None if it is not decodable text."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        log.warning("skipping %s: %s", path, exc)
        return None
