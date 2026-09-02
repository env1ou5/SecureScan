"""Build the normalized corpus (proposal §5).

    python -m securescan_ml.data.build --output datasets/normalized.jsonl \
        --packages 150 --per-class 900

Order matters and is not negotiable:

    generate + mine  ->  deduplicate  ->  split by repository

Deduplicating after splitting would leave near-twins straddling train and test;
splitting on functions instead of repositories would do the same. Both inflate
every metric that follows.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from securescan_ml.data.dedup import deduplicate
from securescan_ml.data.mine import mine_packages, top_package_names
from securescan_ml.data.schema import VulnRecord, write_jsonl
from securescan_ml.data.splits import make_splits
from securescan_ml.data.synthetic import generate_corpus

log = logging.getLogger(__name__)


def build(
    output: Path,
    workdir: Path,
    n_packages: int,
    per_class: int,
    safe_ratio: float,
    seed: int,
    skip_mining: bool = False,
) -> list[VulnRecord]:
    records: list[VulnRecord] = []

    log.info("generating synthetic corpus (%d per vulnerable class)", per_class)
    synthetic = generate_corpus(per_class=per_class, safe_ratio=safe_ratio, seed=seed)
    records.extend(synthetic)
    log.info("  %d synthetic records", len(synthetic))

    if not skip_mining and n_packages > 0:
        names = top_package_names(n_packages)
        log.info("mining %d PyPI packages", len(names))
        mined = mine_packages(names, workdir)
        records.extend(mined)
        log.info("  %d mined records", len(mined))
        real_vuln = sum(
            1 for r in mined if not r.is_synthetic and r.vulnerability_type.value != "SAFE"
        )
        log.info("  %d of them real vulnerable functions", real_vuln)

    log.info("deduplicating %d records", len(records))
    kept, dropped = deduplicate(records)
    log.info("  dropped %d near-duplicates, %d remain", len(dropped), len(kept))

    splits = make_splits(kept)
    log.info("splits:\n%s", splits.summary())

    write_jsonl(kept, output)
    log.info("wrote %s (%d records)", output, len(kept))

    stats = {
        "total": len(kept),
        "synthetic": sum(r.is_synthetic for r in kept),
        "real": sum(not r.is_synthetic for r in kept),
        "duplicates_dropped": len(dropped),
        "by_label": dict(Counter(r.vulnerability_type.value for r in kept)),
        "by_source": dict(Counter(r.source for r in kept)),
        "by_label_real_only": dict(
            Counter(r.vulnerability_type.value for r in kept if not r.is_synthetic)
        ),
        "splits": {
            name: len(getattr(splits, name))
            for name in ("train", "validation", "test_full", "test_real")
        },
        "repositories": len({r.repository_id for r in kept}),
    }
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("wrote %s", stats_path)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("datasets/normalized.jsonl"))
    ap.add_argument("--workdir", type=Path, default=Path("datasets/_mined"))
    ap.add_argument("--packages", type=int, default=150, help="PyPI packages to mine")
    ap.add_argument("--per-class", type=int, default=900, help="synthetic samples per class")
    ap.add_argument("--safe-ratio", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--skip-mining", action="store_true", help="synthetic only (offline)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(
        output=args.output,
        workdir=args.workdir,
        n_packages=args.packages,
        per_class=args.per_class,
        safe_ratio=args.safe_ratio,
        seed=args.seed,
        skip_mining=args.skip_mining,
    )


if __name__ == "__main__":
    main()
