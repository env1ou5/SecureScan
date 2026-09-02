"""Evaluation metrics (proposal §10).

Macro F1 is the headline. Accuracy is deliberately reported but never used for
model selection: with SAFE dominating, a constant predictor wins on accuracy
while being useless.
"""

from __future__ import annotations

import numpy as np

from securescan_ml.labels import ID_TO_LABEL, LABEL_ORDER, NUM_LABELS


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = NUM_LABELS) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_scores(cm: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[ID_TO_LABEL[i].value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }
    return out


def false_positive_rate(cm: np.ndarray, safe_id: int = 0) -> float:
    """Fraction of genuinely-safe functions flagged as vulnerable.

    The metric that decides whether developers keep the tool switched on. A
    scanner that cries wolf gets uninstalled regardless of its recall.
    """
    safe_total = cm[safe_id, :].sum()
    if not safe_total:
        return 0.0
    return float((safe_total - cm[safe_id, safe_id]) / safe_total)


def vulnerable_detection_rate(cm: np.ndarray, safe_id: int = 0) -> float:
    """Recall on the binary vulnerable/not question, ignoring class confusion.

    Calling SQL injection 'command injection' is a far smaller failure than
    calling it safe; this separates the two.
    """
    vuln_total = cm.sum() - cm[safe_id, :].sum()
    if not vuln_total:
        return 0.0
    caught = vuln_total - cm[np.arange(cm.shape[0]) != safe_id, safe_id].sum()
    return float(caught / vuln_total)


def compute_metrics(
    logits: np.ndarray, labels: np.ndarray, min_support: int = 20
) -> dict[str, float]:
    """Metric dict for the HF Trainer. `macro_f1` selects the best checkpoint."""
    preds = logits.argmax(axis=-1)
    cm = confusion_matrix(labels, preds)
    per_class = per_class_scores(cm)

    present = [s for s in per_class.values() if s["support"] > 0]
    macro_f1 = float(np.mean([s["f1"] for s in present])) if present else 0.0
    macro_recall = float(np.mean([s["recall"] for s in present])) if present else 0.0
    macro_precision = float(np.mean([s["precision"] for s in present])) if present else 0.0

    # test_real has fewer than ten examples for several classes, where a single
    # sample swings that class's F1 by 0.5+. This reports the same macro F1
    # restricted to classes with enough support to mean anything. It is
    # published ALONGSIDE macro_f1, never instead of it -- quoting only the
    # flattering subset would be exactly the kind of metric-shopping the
    # two-test-set design exists to prevent.
    supported = [s for s in per_class.values() if s["support"] >= min_support]
    macro_f1_supported = float(np.mean([s["f1"] for s in supported])) if supported else 0.0

    metrics = {
        "macro_f1": macro_f1,
        "macro_f1_supported": macro_f1_supported,
        "n_classes_supported": len(supported),
        "min_support": min_support,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "accuracy": float((preds == labels).mean()),
        "false_positive_rate": false_positive_rate(cm),
        "vulnerable_detection_rate": vulnerable_detection_rate(cm),
    }
    for name, scores in per_class.items():
        metrics[f"recall_{name}"] = scores["recall"]
        metrics[f"f1_{name}"] = scores["f1"]
        metrics[f"support_{name}"] = float(scores["support"])
    return metrics


def format_confusion(cm: np.ndarray) -> str:
    labels = [lab.value[:9] for lab in LABEL_ORDER]
    header = "true\\pred".ljust(24) + "".join(name.rjust(10) for name in labels)
    rows = [header]
    for i, lab in enumerate(LABEL_ORDER):
        rows.append(lab.value.ljust(24) + "".join(str(v).rjust(10) for v in cm[i]))
    return "\n".join(rows)


def localization_accuracy(
    predicted_lines: list[list[int]], true_lines: list[list[int]], k: int = 1
) -> float:
    """Top-k line accuracy for localization (proposal §8).

    A sample counts as a hit when any of the k highest-attributed lines is in
    the ground-truth set from the security fix diff. This is what turns "the
    heatmap looks plausible" into a number.
    """
    if not predicted_lines:
        return 0.0
    hits = sum(
        1
        for pred, truth in zip(predicted_lines, true_lines)
        if truth and set(pred[:k]) & set(truth)
    )
    return hits / len(predicted_lines)
