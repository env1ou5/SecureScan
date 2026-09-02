"""Mine real Python from PyPI and label it with Semgrep (proposal §5, source 3).

Why Semgrep and not Bandit: Bandit is the *baseline* the model is benchmarked
against (§10). Labeling with the same tool you benchmark against makes the
comparison circular -- you would be measuring agreement, not correctness. Using
a different engine for labels than for the baseline keeps the benchmark
meaningful.

Findings map to our taxonomy through their CWE metadata, reusing CWE_TO_LABEL
from labels.py, so the mapping lives in exactly one place.

Honest limitation, documented rather than hidden: these are static-analysis
labels, not human-verified CVE labels. Semgrep has its own false positives and
blind spots, and a model trained on them inherits both. This is the single
biggest caveat on every number this project reports.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from securescan_ml.chunking import extract_functions
from securescan_ml.data.ast_labeler import label_source
from securescan_ml.data.schema import VulnRecord
from securescan_ml.labels import CWE_TO_LABEL, Label

log = logging.getLogger(__name__)

TOP_PACKAGES_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
PYPI_JSON = "https://pypi.org/pypi/{name}/json"

USER_AGENT = "SecureScan-dataset-builder/0.1 (research)"

# Semgrep rulesets. Registry-hosted, fetched once and cached.
SEMGREP_CONFIGS = ["r/python.lang.security", "p/security-audit"]

_CWE_RE = re.compile(r"CWE-\d+")

SKIP_PATH_PARTS = frozenset(
    {
        "test",
        "tests",
        "testing",
        "_test",
        "conftest",
        "setup",
        "docs",
        "examples",
        "example",
        "benchmark",
        "benchmarks",
        "vendor",
        "vendored",
        "_vendor",
    }
)


@dataclass
class MinedPackage:
    name: str
    version: str
    root: Path
    files: list[Path]


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def top_package_names(limit: int) -> list[str]:
    """Most-downloaded PyPI packages: a realistic sample of production Python."""
    try:
        payload = json.loads(_fetch(TOP_PACKAGES_URL))
    except Exception as exc:  # noqa: BLE001
        log.error("could not fetch top package list: %s", exc)
        return []
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    names = []
    for row in rows:
        name = row.get("project") if isinstance(row, dict) else row
        if name:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def download_package(name: str, dest: Path, timeout: int = 60) -> MinedPackage | None:
    """Download and unpack one package's sdist or wheel."""
    try:
        meta = json.loads(_fetch(PYPI_JSON.format(name=name), timeout=timeout))
    except Exception as exc:  # noqa: BLE001
        log.debug("metadata failed for %s: %s", name, exc)
        return None

    version = meta["info"]["version"]
    urls = meta.get("urls", [])
    chosen = next((u for u in urls if u["packagetype"] == "sdist"), None) or next(
        (u for u in urls if u["filename"].endswith(".whl")), None
    )
    if chosen is None:
        return None

    try:
        blob = _fetch(chosen["url"], timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        log.debug("download failed for %s: %s", name, exc)
        return None

    root = dest / name
    root.mkdir(parents=True, exist_ok=True)
    try:
        if chosen["filename"].endswith((".tar.gz", ".tgz")):
            with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
                members = [
                    m
                    for m in tf.getmembers()
                    if m.isfile()
                    and m.name.endswith(".py")
                    and not m.name.startswith("/")
                    and ".." not in Path(m.name).parts
                ]
                tf.extractall(root, members=members, filter="data")
        elif chosen["filename"].endswith((".zip", ".whl")):
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for info in zf.infolist():
                    if (
                        info.filename.endswith(".py")
                        and not info.filename.startswith("/")
                        and ".." not in Path(info.filename).parts
                    ):
                        zf.extract(info, root)
        else:
            return None
    except Exception as exc:  # noqa: BLE001
        log.debug("extract failed for %s: %s", name, exc)
        return None

    files = [
        p
        for p in root.rglob("*.py")
        if not any(part.lower() in SKIP_PATH_PARTS for part in p.parts)
    ]
    if not files:
        return None
    return MinedPackage(name=name, version=version, root=root, files=files)


def _semgrep_executable() -> str:
    """Locate semgrep next to the running interpreter.

    A bare "semgrep" resolves only when the venv is on PATH, which it is not
    when the module is invoked as `.venv/bin/python -m ...`. `python -m semgrep`
    is deprecated and prints a warning to stdout, corrupting the JSON. So look
    in the interpreter's own bin directory first.
    """
    candidate = Path(sys.executable).parent / "semgrep"
    return str(candidate) if candidate.exists() else "semgrep"


def run_semgrep(target: Path, timeout: int = 5400) -> dict[tuple[str, int], list[Label]]:
    """Run Semgrep over a directory.

    Returns {(absolute_file_path, line): [labels]} for findings whose CWE
    metadata maps onto our taxonomy. Findings outside the taxonomy are dropped
    rather than guessed at -- a wrong label is worse than a missing sample.
    """
    cmd = [
        _semgrep_executable(),
        "scan",
        "--json",
        "--quiet",
        "--no-git-ignore",
        "--metrics=off",
        "--jobs",
        str(max(os.cpu_count() or 2, 2)),
    ]
    for config in SEMGREP_CONFIGS:
        cmd += ["--config", config]
    cmd.append(str(target))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        # Fail loudly. Returning {} here would label the entire corpus SAFE and
        # the failure would only surface as a mysteriously useless model.
        raise RuntimeError(
            f"semgrep could not be run ({exc}). Refusing to continue: an "
            "unlabeled corpus would silently train on all-SAFE data."
        ) from exc

    if not proc.stdout.strip():
        raise RuntimeError(f"semgrep produced no output. stderr: {proc.stderr[-800:]}")

    try:
        results = json.loads(proc.stdout).get("results", [])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse semgrep output: {proc.stdout[:300]!r}") from exc

    hits: dict[tuple[str, int], list[Label]] = defaultdict(list)
    for finding in results:
        metadata = finding.get("extra", {}).get("metadata", {})
        cwes = metadata.get("cwe") or []
        if isinstance(cwes, str):
            cwes = [cwes]
        labels = []
        for cwe_text in cwes:
            for cwe_id in _CWE_RE.findall(str(cwe_text)):
                label = CWE_TO_LABEL.get(cwe_id.upper())
                if label is not None:
                    labels.append(label)
        if not labels:
            continue
        path = finding.get("path", "")
        line = finding.get("start", {}).get("line", 0)
        hits[(str(Path(path).resolve()), int(line))].extend(labels)
    return dict(hits)


def mine_packages(
    package_names: list[str],
    workdir: Path,
    max_safe_per_package: int = 12,
    min_function_lines: int = 3,
    max_function_lines: int = 120,
) -> list[VulnRecord]:
    """Download, scan, and convert packages into labeled function records.

    A function is labeled vulnerable when a mapped Semgrep finding falls inside
    its line range, and SAFE only when Semgrep reports nothing anywhere in it.
    Bandit is deliberately never consulted here.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    records: list[VulnRecord] = []

    packages: list[MinedPackage] = []
    for name in package_names:
        pkg = download_package(name, workdir)
        if pkg is not None:
            packages.append(pkg)
            log.info("downloaded %s==%s (%d files)", pkg.name, pkg.version, len(pkg.files))

    if not packages:
        return []

    log.info("running semgrep over %d packages", len(packages))
    hits = run_semgrep(workdir)
    log.info("semgrep produced %d mapped finding locations", len(hits))

    by_file: dict[str, list[tuple[int, Label]]] = defaultdict(list)
    for (path, line), labels in hits.items():
        for label in labels:
            by_file[path].append((line, label))

    for pkg in packages:
        safe_budget = max_safe_per_package
        for file_path in pkg.files:
            try:
                source = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            resolved = str(file_path.resolve())
            file_hits = list(by_file.get(resolved, []))
            # Semgrep covers neither HARDCODED_SECRET nor PATH_TRAVERSAL; the
            # independent AST detectors fill those in (see ast_labeler.py).
            file_hits.extend((line, label) for line, label in label_source(source).items())
            relative = str(file_path.relative_to(pkg.root))

            for chunk in extract_functions(source, relative):
                n_lines = chunk.end_line - chunk.start_line + 1
                if not (min_function_lines <= n_lines <= max_function_lines):
                    continue

                inside = [
                    (line, label)
                    for line, label in file_hits
                    if chunk.start_line <= line <= chunk.end_line
                ]

                if inside:
                    # Most specific wins when a function trips several rules:
                    # take the label of the earliest finding.
                    line, label = min(inside, key=lambda x: x[0])
                    records.append(
                        VulnRecord(
                            repository_id=f"pypi/{pkg.name}",
                            file_path=relative,
                            language="python",
                            code=chunk.code,
                            vulnerability_type=label,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            source="mined",
                            is_synthetic=False,
                            vulnerable_lines=[line],
                            metadata={
                                "package": pkg.name,
                                "version": pkg.version,
                                "labeler": "semgrep",
                                "function": chunk.name,
                            },
                        )
                    )
                elif safe_budget > 0 and not file_hits:
                    # SAFE only from files Semgrep flagged nowhere, and capped
                    # per package so one large library cannot dominate.
                    safe_budget -= 1
                    records.append(
                        VulnRecord(
                            repository_id=f"pypi/{pkg.name}",
                            file_path=relative,
                            language="python",
                            code=chunk.code,
                            vulnerability_type=Label.SAFE,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            source="mined",
                            is_synthetic=False,
                            metadata={
                                "package": pkg.name,
                                "version": pkg.version,
                                "labeler": "semgrep",
                                "function": chunk.name,
                            },
                        )
                    )
    return records
