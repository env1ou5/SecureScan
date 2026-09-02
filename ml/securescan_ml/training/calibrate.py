"""Temperature scaling for calibrated confidence (proposal §2).

A fine-tuned Transformer's raw softmax is badly overconfident -- it will happily
report 0.99 on a coin flip. The dashboard shows a confidence number to a
developer who will act on it, so that number has to mean something.

Temperature scaling fits a single scalar T on the validation set, dividing
logits by T before softmax. It cannot change which class is predicted (argmax is
invariant), so accuracy and F1 are untouched; only the confidence moves.

    python -m securescan_ml.training.calibrate --model artifacts/unixcoder-v1 \
        --data datasets/normalized.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration.json"


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size."""
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Optimize a single temperature against NLL with LBFGS."""
    logits_t = torch.tensor(logits, dtype=torch.float)
    labels_t = torch.tensor(labels, dtype=torch.long)

    log_t = torch.zeros(1, requires_grad=True)  # optimize log T to keep T > 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Data for the reliability diagram -- plot this, don't just cite the ECE."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        bins.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return bins


def save_calibration(model_dir: str | Path, temperature: float, stats: dict) -> Path:
    path = Path(model_dir) / CALIBRATION_FILE
    path.write_text(json.dumps({"temperature": temperature, **stats}, indent=2))
    return path


def load_temperature(model_dir: str | Path) -> float:
    """Return the fitted temperature, or 1.0 (a no-op) if none was fitted.

    Defaulting to 1.0 means an uncalibrated model still serves; the predictor
    flags `calibrated: false` so the dashboard can say so rather than quietly
    showing an inflated number.
    """
    path = Path(model_dir) / CALIBRATION_FILE
    if not path.exists():
        log.warning("no %s in %s -- serving uncalibrated confidence", CALIBRATION_FILE, model_dir)
        return 1.0
    return float(json.loads(path.read_text())["temperature"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="fine-tuned checkpoint directory")
    ap.add_argument("--data", required=True, help="normalized JSONL")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--fit-on",
        choices=("real", "all"),
        default="real",
        help=(
            "which validation samples to fit the temperature on. 'real' uses "
            "only non-synthetic samples, which is what deployment sees."
        ),
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Imported here so `load_temperature` stays importable without transformers.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from securescan_ml.data.schema import read_jsonl
    from securescan_ml.data.splits import make_splits
    from securescan_ml.labels import LABEL_TO_ID

    splits = make_splits(list(read_jsonl(args.data)))
    if not splits.validation:
        raise SystemExit("validation split is empty; cannot fit a temperature")

    # Fitting on a synthetic-dominated validation set is actively harmful. The
    # model is ~99.7% accurate there, so the optimizer learns T < 1 -- sharpen
    # further -- and then applies that to real code where accuracy is ~92% and
    # the model is already overconfident. Measured: fitting on all of
    # validation moved real-code ECE 0.0592 -> 0.0609, the wrong direction.
    # Default to fitting on the real subset, which is what deployment sees.
    fit_pool = splits.validation
    if args.fit_on == "real":
        real_only = [r for r in splits.validation if not r.is_synthetic]
        if len(real_only) < 50:
            log.warning(
                "only %d real validation samples; falling back to the full "
                "validation set. The fitted temperature will be biased toward "
                "synthetic data.",
                len(real_only),
            )
        else:
            fit_pool = real_only
            log.info(
                "fitting on %d real validation samples (of %d total)",
                len(real_only),
                len(splits.validation),
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    log.info("fitting temperature on %d validation samples (%s)", len(splits.validation), device)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(fit_pool), args.batch_size):
            batch = fit_pool[i : i + args.batch_size]
            enc = tokenizer(
                [r.code for r in batch],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(device)
            all_logits.append(model(**enc).logits.float().cpu().numpy())
            all_labels.extend(LABEL_TO_ID[r.vulnerability_type] for r in batch)

    logits = np.concatenate(all_logits)
    labels = np.asarray(all_labels)

    before = apply_temperature(logits, 1.0)
    temperature = fit_temperature(logits, labels)
    after = apply_temperature(logits, temperature)

    ece_before = expected_calibration_error(before, labels)
    ece_after = expected_calibration_error(after, labels)

    log.info("temperature = %.4f", temperature)
    log.info("ECE %.4f -> %.4f", ece_before, ece_after)

    # Fitting on validation says little when validation is synthetic and
    # saturated: confidence and accuracy are both ~0.99, so ECE is near zero
    # before any correction. The question that matters is whether the fitted
    # temperature transfers to real code, where the model is weaker. Measure it.
    transfer: dict[str, dict] = {}
    for name in ("test_full", "test_real"):
        subset = getattr(splits, name)
        if not subset:
            continue
        sub_logits, sub_labels = [], []
        with torch.no_grad():
            for i in range(0, len(subset), args.batch_size):
                batch = subset[i : i + args.batch_size]
                enc = tokenizer(
                    [r.code for r in batch],
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                sub_logits.append(model(**enc).logits.float().cpu().numpy())
                sub_labels.extend(LABEL_TO_ID[r.vulnerability_type] for r in batch)
        sl = np.concatenate(sub_logits)
        sy = np.asarray(sub_labels)
        transfer[name] = {
            "n": int(len(sy)),
            "ece_before": expected_calibration_error(apply_temperature(sl, 1.0), sy),
            "ece_after": expected_calibration_error(apply_temperature(sl, temperature), sy),
            "accuracy": float((sl.argmax(axis=1) == sy).mean()),
            "mean_confidence": float(apply_temperature(sl, temperature).max(axis=1).mean()),
        }
        t = transfer[name]
        log.info(
            "%s (n=%d): acc %.4f | mean conf %.4f | ECE %.4f -> %.4f",
            name,
            t["n"],
            t["accuracy"],
            t["mean_confidence"],
            t["ece_before"],
            t["ece_after"],
        )
        gap = t["mean_confidence"] - t["accuracy"]
        if gap > 0.1:
            log.warning(
                "  %s is overconfident by %.3f even after scaling -- a temperature "
                "fit on synthetic validation does not transfer to real code.",
                name,
                gap,
            )

    path = save_calibration(
        args.model,
        temperature,
        {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "n_validation": int(len(labels)),
            "fit_on": args.fit_on,
            "validation_synthetic_share": float(
                sum(r.is_synthetic for r in splits.validation) / len(splits.validation)
            ),
            "transfer": transfer,
            "reliability_bins": reliability_bins(after, labels),
        },
    )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
