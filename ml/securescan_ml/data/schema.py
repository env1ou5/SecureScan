"""Unified record schema for every dataset source (proposal §5).

Every source -- CVEFixes, Juliet, mined repositories -- normalizes into this one
shape before anything downstream touches it. `source` and `is_synthetic` are
carried all the way through so any metric can be recomputed on real code alone,
which is the whole defense against the synthetic-data trap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from securescan_ml.labels import Label

Source = Literal["cvefixes", "juliet", "mined"]

SYNTHETIC_SOURCES: frozenset[str] = frozenset({"juliet"})


@dataclass
class VulnRecord:
    repository_id: str  # splits are made on this -- never on the function
    file_path: str
    language: str
    code: str
    vulnerability_type: Label
    start_line: int
    end_line: int
    source: Source
    is_synthetic: bool
    # Lines changed by the security fix commit, when the source provides a diff.
    # Ground truth for the localization eval set (proposal §8). Empty otherwise.
    vulnerable_lines: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.repository_id:
            raise ValueError("repository_id is required: splits depend on it")
        if self.is_synthetic != (self.source in SYNTHETIC_SOURCES):
            raise ValueError(f"is_synthetic={self.is_synthetic} contradicts source={self.source!r}")

    @property
    def fingerprint(self) -> str:
        """Stable id for exact-duplicate detection and caching."""
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> str:
        d = asdict(self)
        d["vulnerability_type"] = self.vulnerability_type.value
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> VulnRecord:
        d = json.loads(line)
        d["vulnerability_type"] = Label(d["vulnerability_type"])
        return cls(**d)


def write_jsonl(records: list[VulnRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
    return len(records)


def read_jsonl(path: str | Path) -> Iterator[VulnRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield VulnRecord.from_json(line)
