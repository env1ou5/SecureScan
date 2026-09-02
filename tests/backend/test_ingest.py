"""Archive extraction must reject every one of these (proposal §7)."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest
from app.services.ingest import UnsafeArchiveError, extract_archive


def make_zip(tmp_path: Path, members: list[tuple[str, str]], name: str = "a.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for member_name, data in members:
            zf.writestr(member_name, data)
    return path


def test_rejects_path_traversal(tmp_path, settings):
    archive = make_zip(tmp_path, [("../../evil.py", "x = 1")])
    with pytest.raises(UnsafeArchiveError, match="unsafe member path"):
        extract_archive(archive, settings)


def test_rejects_absolute_paths(tmp_path, settings):
    archive = make_zip(tmp_path, [("/etc/cron.d/evil.py", "x = 1")])
    with pytest.raises(UnsafeArchiveError, match="unsafe member path"):
        extract_archive(archive, settings)


def test_rejects_zip_bomb(tmp_path, settings):
    archive = make_zip(tmp_path, [("big.py", "A" * 5_000_000)])
    with pytest.raises(UnsafeArchiveError, match="compression ratio|uncompressed size"):
        extract_archive(archive, settings)


def test_rejects_too_many_members(tmp_path, settings):
    archive = make_zip(
        tmp_path, [(f"f{i}.py", "x = 1") for i in range(settings.max_archive_members + 1)]
    )
    with pytest.raises(UnsafeArchiveError, match="members"):
        extract_archive(archive, settings)


def test_rejects_zip_symlink(tmp_path, settings):
    path = tmp_path / "sym.zip"
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("link.py")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(UnsafeArchiveError, match="symlink"):
        extract_archive(path, settings)


def test_rejects_tar_symlink(tmp_path, settings):
    path = tmp_path / "sym.tar"
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo("link.py")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(UnsafeArchiveError, match="link"):
        extract_archive(path, settings)


def test_rejects_non_archive(tmp_path, settings):
    path = tmp_path / "plain.py"
    path.write_text("x = 1")
    with pytest.raises(UnsafeArchiveError, match="unsupported archive format"):
        extract_archive(path, settings)


def test_rejects_archive_without_python(tmp_path, settings):
    archive = make_zip(tmp_path, [("README.md", "docs")])
    with pytest.raises(UnsafeArchiveError, match="no Python files"):
        extract_archive(archive, settings)


def test_extracts_python_and_skips_noise(tmp_path, settings):
    archive = make_zip(
        tmp_path,
        [
            ("proj/app.py", "def f():\n    pass\n"),
            ("proj/db.py", "def g():\n    pass\n"),
            ("proj/.git/config", "[core]"),
            ("proj/node_modules/pkg.py", "junk = 1"),
            ("proj/README.md", "docs"),
        ],
    )
    with extract_archive(archive, settings) as result:
        names = sorted(p.name for p in result.python_files)
        assert names == ["app.py", "db.py"]
        assert result.total_bytes > 0


def test_extraction_directory_is_removed(tmp_path, settings):
    archive = make_zip(tmp_path, [("a.py", "def f():\n    pass\n")])
    with extract_archive(archive, settings) as result:
        root = result.root
        assert root.exists()
    assert not root.exists()
