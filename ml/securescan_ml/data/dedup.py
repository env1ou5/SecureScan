"""Near-duplicate removal via MinHash + LSH (proposal §5).

Synthetic corpora contain many near-identical variants of the same template, and
real repositories contain vendored copies. Without dedup, near-twins land on
both sides of the split and the test metric measures memorization.

Runs before splitting, never after.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from securescan_ml.data.schema import VulnRecord

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def tokenize(code: str) -> list[str]:
    """Crude lexer. Good enough for shingling; not a parser."""
    return _TOKEN_RE.findall(code)


def shingles(tokens: list[str], k: int = 5) -> set[str]:
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def minhash(code: str, num_perm: int = 128, k: int = 5) -> tuple[int, ...]:
    """MinHash signature.

    Uses seeded sha1 per permutation. Slower than the usual (a*x+b mod p)
    trick but has no parameter-choice failure modes, and dedup runs once
    offline.
    """
    shs = shingles(tokenize(code), k)
    if not shs:
        return tuple([0] * num_perm)
    sig = []
    for seed in range(num_perm):
        prefix = seed.to_bytes(4, "little")
        sig.append(
            min(
                int.from_bytes(hashlib.sha1(prefix + s.encode()).digest()[:8], "little")
                for s in shs
            )
        )
    return tuple(sig)


def jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def deduplicate(
    records: list[VulnRecord],
    threshold: float = 0.85,
    num_perm: int = 128,
    bands: int = 16,
) -> tuple[list[VulnRecord], list[tuple[int, int]]]:
    """Return (kept records, dropped (duplicate_idx, kept_idx) pairs).

    LSH bands the signature so only plausible pairs get compared. When a
    duplicate group spans multiple sources, the real-code record is kept over
    the synthetic one -- real samples are the scarce resource.
    """
    if bands <= 0 or num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands

    sigs = [minhash(r.code, num_perm) for r in records]

    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            band = sig[b * rows : (b + 1) * rows]
            key = (b, hashlib.sha1(repr(band).encode()).digest()[:8])
            buckets[key].append(idx)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                candidates.add((a, b) if a < b else (b, a))

    # Prefer keeping real code when a real/synthetic pair collides.
    def preference(idx: int) -> tuple[int, int]:
        return (0 if not records[idx].is_synthetic else 1, idx)

    dropped: dict[int, int] = {}
    for a, b in sorted(candidates):
        if a in dropped or b in dropped:
            continue
        if jaccard(sigs[a], sigs[b]) < threshold:
            continue
        keep, drop = sorted((a, b), key=preference)
        dropped[drop] = keep

    kept = [r for i, r in enumerate(records) if i not in dropped]
    return kept, sorted(dropped.items())
