"""Measure line-level localization quality (proposal §8).

    python -m securescan_ml.evaluation.eval_localization \
        --model artifacts/unixcoder-v1 --data datasets/normalized.jsonl

Attribution is noisy, and "the heatmap looks plausible" is not a result. This
scores gradient x input attribution against ground-truth vulnerable lines and
reports top-1 / top-3 line accuracy.

Ground truth comes from `vulnerable_lines` on each record -- the Semgrep or
AST-detector finding line for mined code. Records without it are skipped.

Read the number with its ceiling in mind: the target lines are where a static
analyzer fired, so this measures agreement with the labeler's notion of the
offending line, not with a human's.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from securescan_ml.data.schema import read_jsonl
from securescan_ml.data.splits import make_splits
from securescan_ml.evaluation.metrics import localization_accuracy
from securescan_ml.labels import LABEL_TO_ID, Label

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="artifacts/unixcoder-v1")
    ap.add_argument("--data", default="datasets/normalized.jsonl")
    ap.add_argument("--output", type=Path, default=Path("artifacts/localization.json"))
    ap.add_argument("--limit", type=int, default=400, help="samples to score")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from securescan_ml.inference.attribution import localize

    splits = make_splits(list(read_jsonl(args.data)))

    # Only vulnerable records carrying ground-truth lines can be scored.
    candidates = [
        r for r in splits.test_full if r.vulnerability_type is not Label.SAFE and r.vulnerable_lines
    ]
    if not candidates:
        raise SystemExit(
            "no test records carry `vulnerable_lines`; nothing to score. "
            "Mined records get them from the labeler; synthetic ones do not."
        )

    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[: args.limit]
    log.info("scoring localization on %d records", len(candidates))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()

    predicted: list[list[int]] = []
    truth: list[list[int]] = []
    correct_class = 0

    for i, record in enumerate(candidates, 1):
        target = LABEL_TO_ID[record.vulnerability_type]
        lines, predicted_class = localize(
            model,
            tokenizer,
            record.code,
            record.start_line,
            target_class=target,
            top_k=3,
        )
        if predicted_class == target:
            correct_class += 1
        predicted.append([line.line for line in lines])
        truth.append(record.vulnerable_lines)
        if i % 100 == 0:
            log.info("  %d/%d", i, len(candidates))

    results = {
        "n": len(candidates),
        "top_1_line_accuracy": localization_accuracy(predicted, truth, k=1),
        "top_3_line_accuracy": localization_accuracy(predicted, truth, k=3),
        # Attribution is computed for the true class, so this is reported
        # separately: it says how often the classifier also got the class right.
        "class_accuracy_on_these": correct_class / len(candidates),
        "note": (
            "Ground truth is the labeling analyzer's finding line, not a human "
            "annotation. This measures agreement with Semgrep/AST detectors."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    log.info(
        "top-1 %.3f | top-3 %.3f | n=%d",
        results["top_1_line_accuracy"],
        results["top_3_line_accuracy"],
        results["n"],
    )
    log.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
