"""Repository-disjoint splits and the real-code-only test set (proposal §5).

Two rules, both load-bearing:

  1. Split by repository, never by function. Functions from one project are
     near-duplicates of each other; scattering them across train and test
     inflates every metric.
  2. The real-code test set contains no synthetic samples, ever. It is the
     number that goes on the resume.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

from securescan_ml.data.schema import VulnRecord
from securescan_ml.labels import Label


@dataclass
class SplitResult:
    train: list[VulnRecord]
    validation: list[VulnRecord]
    test_full: list[VulnRecord]  # real + synthetic, repo-disjoint
    test_real: list[VulnRecord]  # hand-verified real code only

    def summary(self) -> str:
        rows = []
        for name in ("train", "validation", "test_full", "test_real"):
            recs = getattr(self, name)
            dist = Counter(r.vulnerability_type.value for r in recs)
            synth = sum(r.is_synthetic for r in recs)
            rows.append(
                f"{name:11} n={len(recs):6d}  synthetic={synth:6d}  "
                + " ".join(f"{k}={v}" for k, v in sorted(dist.items()))
            )
        return "\n".join(rows)


def _bucket(repository_id: str, salt: str) -> float:
    """Deterministic [0,1) hash. Same repo always lands in the same split."""
    digest = hashlib.sha256(f"{salt}:{repository_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def make_splits(
    records: list[VulnRecord],
    val_frac: float = 0.10,
    test_frac: float = 0.15,
    salt: str = "securescan-v1",
) -> SplitResult:
    """Assign whole repositories to splits, then carve out the real-code test set.

    Hashing the repository id rather than shuffling means adding new data never
    reshuffles existing repositories across splits, so metrics stay comparable
    between dataset versions.
    """
    by_repo: dict[str, list[VulnRecord]] = defaultdict(list)
    for rec in records:
        by_repo[rec.repository_id].append(rec)

    train: list[VulnRecord] = []
    validation: list[VulnRecord] = []
    test_full: list[VulnRecord] = []

    for repo, recs in by_repo.items():
        b = _bucket(repo, salt)
        if b < test_frac:
            test_full.extend(recs)
        elif b < test_frac + val_frac:
            validation.extend(recs)
        else:
            train.extend(recs)

    # The honest test set: real code only, drawn from the same held-out repos.
    test_real = [r for r in test_full if not r.is_synthetic and r.source == "mined"]

    return SplitResult(train, validation, test_full, test_real)


def class_weights(records: list[VulnRecord], label_order: tuple[Label, ...]) -> list[float]:
    """Inverse-frequency weights for the loss, normalized to mean 1.

    SAFE dominates heavily; without this the model can score well on accuracy
    while never predicting a rare class at all.
    """
    counts = Counter(r.vulnerability_type for r in records)
    total = sum(counts.values())
    n_classes = len(label_order)
    weights = []
    for lab in label_order:
        c = counts.get(lab, 0)
        weights.append(total / (n_classes * c) if c else 0.0)
    present = [w for w in weights if w > 0]
    mean = sum(present) / len(present) if present else 1.0
    return [w / mean for w in weights]
