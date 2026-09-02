"""Run every approach under one harness and emit the results table (proposal §10).

    python -m securescan_ml.evaluation.run_benchmark \
        --data datasets/normalized.jsonl --model artifacts/unixcoder-v1

One harness, one set of splits, three approaches. Building the baselines and the
model against different evaluation code is how benchmark tables end up
flattering the thing the author wanted to win.

Every metric is reported twice -- once on the full held-out set and once on real
mined code only. The second number is the honest one and is expected to be
lower.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from securescan_ml.data.schema import read_jsonl
from securescan_ml.data.splits import make_splits
from securescan_ml.evaluation.metrics import (
    compute_metrics,
    confusion_matrix,
    format_confusion,
)
from securescan_ml.labels import LABEL_TO_ID, NUM_LABELS

log = logging.getLogger(__name__)


def _as_logits(preds: np.ndarray) -> np.ndarray:
    """compute_metrics consumes logits; one-hot the baselines' hard labels."""
    one_hot = np.zeros((len(preds), NUM_LABELS))
    one_hot[np.arange(len(preds)), preds] = 1.0
    return one_hot


def transformer_predictions(
    model_dir: str, records: list, batch_size: int = 32, max_length: int = 512
) -> tuple[np.ndarray, float]:
    """Predict with the fine-tuned model. Returns (logits, seconds per sample)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    outputs = []
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            encoded = tokenizer(
                [r.code for r in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            outputs.append(model(**encoded).logits.cpu().numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), elapsed / max(len(records), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("datasets/normalized.jsonl"))
    ap.add_argument("--model", type=str, default="artifacts/unixcoder-v1")
    ap.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    ap.add_argument("--skip-bandit", action="store_true")
    ap.add_argument("--max-test", type=int, default=0, help="cap test size (0 = all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    splits = make_splits(list(read_jsonl(args.data)))
    log.info("splits:\n%s", splits.summary())

    test_sets = {"test_full": splits.test_full, "test_real": splits.test_real}
    results: dict[str, dict] = {}

    for set_name, records in test_sets.items():
        if not records:
            log.warning("%s is empty -- skipping", set_name)
            continue
        if args.max_test:
            records = records[: args.max_test]

        y_true = np.array([LABEL_TO_ID[r.vulnerability_type] for r in records])
        results[set_name] = {"n": len(records)}
        log.info("=== %s (n=%d) ===", set_name, len(records))

        # Baseline 1: Bandit (rule-based). Never used for labeling -- see mine.py.
        if not args.skip_bandit:
            from securescan_ml.evaluation.baselines import bandit_predict

            started = time.perf_counter()
            preds = bandit_predict(records)
            results[set_name]["bandit"] = compute_metrics(_as_logits(preds), y_true)
            results[set_name]["bandit"]["seconds_per_sample"] = (
                time.perf_counter() - started
            ) / len(records)
            log.info("bandit macro_f1=%.4f", results[set_name]["bandit"]["macro_f1"])

        # Baseline 2: TF-IDF + Random Forest, trained on the same train split.
        from securescan_ml.evaluation.baselines import train_random_forest

        rf = train_random_forest(splits.train)
        started = time.perf_counter()
        rf_preds = np.asarray(rf.predict([r.code for r in records]))
        results[set_name]["random_forest"] = compute_metrics(_as_logits(rf_preds), y_true)
        results[set_name]["random_forest"]["seconds_per_sample"] = (
            time.perf_counter() - started
        ) / len(records)
        log.info("random_forest macro_f1=%.4f", results[set_name]["random_forest"]["macro_f1"])

        # The model.
        if Path(args.model).exists():
            logits, per_sample = transformer_predictions(args.model, records)
            results[set_name]["transformer"] = compute_metrics(logits, y_true)
            results[set_name]["transformer"]["seconds_per_sample"] = per_sample
            log.info("transformer macro_f1=%.4f", results[set_name]["transformer"]["macro_f1"])
            cm = confusion_matrix(y_true, logits.argmax(axis=-1))
            log.info("confusion matrix (transformer, %s):\n%s", set_name, format_confusion(cm))
            results[set_name]["transformer_confusion"] = cm.tolist()
        else:
            log.warning("no checkpoint at %s -- skipping transformer", args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=float))
    log.info("wrote %s", args.output)
    print("\n" + render_table(results))


def render_table(results: dict) -> str:
    """Markdown table, ready to paste into the proposal."""
    lines = []
    for set_name, block in results.items():
        lines.append(f"\n### {set_name} (n={block.get('n', 0)})\n")
        lines.append("| Approach | Precision | Recall | Macro F1 | FPR | ms/sample |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for approach, pretty in (
            ("bandit", "Bandit (rule-based)"),
            ("random_forest", "TF-IDF + Random Forest"),
            ("transformer", "UniXcoder (PyTorch)"),
        ):
            m = block.get(approach)
            if not m:
                continue
            lines.append(
                f"| {pretty} | {m['macro_precision']:.3f} | {m['macro_recall']:.3f} | "
                f"**{m['macro_f1']:.3f}** | {m['false_positive_rate']:.3f} | "
                f"{m.get('seconds_per_sample', 0) * 1000:.1f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
