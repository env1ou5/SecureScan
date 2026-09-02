"""Baselines the Transformer must beat (proposal §10).

Both run under the same harness and the same splits as the model. Build these
BEFORE the Transformer: a baseline written after the fact tends to be a weak
one, and the comparison stops meaning anything.

Note the circularity risk (proposal §5): if training labels were mined using
Bandit, then beating Bandit on those labels proves nothing. That is precisely
why the real-code test set is hand-verified.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from securescan_ml.data.schema import VulnRecord
from securescan_ml.labels import LABEL_TO_ID, Label

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Baseline 1: Bandit (rule-based)
# --------------------------------------------------------------------------

# Bandit test ids -> our taxonomy. Bandit does not cover XSS, so that class is
# an automatic miss for this baseline -- report it rather than hiding it.
BANDIT_TEST_TO_LABEL: dict[str, Label] = {
    "B608": Label.SQL_INJECTION,
    "B610": Label.SQL_INJECTION,
    "B611": Label.SQL_INJECTION,
    "B602": Label.COMMAND_INJECTION,
    "B603": Label.COMMAND_INJECTION,
    "B604": Label.COMMAND_INJECTION,
    "B605": Label.COMMAND_INJECTION,
    "B606": Label.COMMAND_INJECTION,
    "B607": Label.COMMAND_INJECTION,
    "B301": Label.UNSAFE_DESERIALIZATION,
    "B302": Label.UNSAFE_DESERIALIZATION,
    "B403": Label.UNSAFE_DESERIALIZATION,
    "B506": Label.UNSAFE_DESERIALIZATION,
    "B105": Label.HARDCODED_SECRET,
    "B106": Label.HARDCODED_SECRET,
    "B107": Label.HARDCODED_SECRET,
    "B108": Label.PATH_TRAVERSAL,
}


def _bandit_executable() -> str:
    """Locate bandit next to the running interpreter.

    Same trap as semgrep (see data/mine.py): a bare name only resolves when the
    venv is on PATH, and silently predicting all-SAFE would make the baseline
    look artificially terrible.
    """
    candidate = Path(sys.executable).parent / "bandit"
    return str(candidate) if candidate.exists() else "bandit"


def bandit_predict(records: list[VulnRecord], timeout: int = 1800) -> np.ndarray:
    """Run Bandit over every record and map its findings onto our labels.

    All samples are written to one directory and scanned in a single Bandit
    invocation. Spawning one process per sample costs ~0.4s of interpreter
    startup each, which on a few thousand records dominates everything else in
    the benchmark.

    Anything Bandit does not flag, or that maps to no label of ours, is SAFE.
    """
    preds = np.zeros(len(records), dtype=int)  # 0 == SAFE
    if not records:
        return preds

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i, rec in enumerate(records):
            (root / f"sample_{i:06d}.py").write_text(rec.code, encoding="utf-8")

        cmd = [_bandit_executable(), "-f", "json", "-q", "-r", str(root)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"bandit could not be run ({exc}). Refusing to report an "
                "all-SAFE baseline, which would flatter the model."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"bandit timed out after {timeout}s") from exc

        # Bandit exits non-zero when it finds issues, which is the normal case.
        if not proc.stdout.strip():
            raise RuntimeError(f"bandit produced no output. stderr: {proc.stderr[-500:]}")

        try:
            results = json.loads(proc.stdout).get("results", [])
        except json.JSONDecodeError as exc:
            raise RuntimeError("could not parse bandit output") from exc

        index_re = re.compile(r"sample_(\d{6})\.py$")
        for finding in results:
            match = index_re.search(finding.get("filename", ""))
            if not match:
                continue
            idx = int(match.group(1))
            label = BANDIT_TEST_TO_LABEL.get(finding.get("test_id", ""))
            # First mapped finding wins, matching how the miner resolves
            # multiple hits in one function.
            if label is not None and preds[idx] == 0:
                preds[idx] = LABEL_TO_ID[label]

    return preds


# --------------------------------------------------------------------------
# Baseline 2: TF-IDF + Random Forest (classical ML)
# --------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]")


def code_tokenizer(code: str) -> list[str]:
    return _IDENT_RE.findall(code)


def train_random_forest(train: list[VulnRecord], seed: int = 42):
    """TF-IDF over code tokens into a Random Forest.

    Deliberately naive: it has no notion of data flow, only of which tokens
    co-occur. If the Transformer cannot beat this, the Transformer is not
    earning its complexity -- which is exactly what the benchmark is for.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=code_tokenizer,
                    lowercase=False,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=50_000,
                    token_pattern=None,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(
        [r.code for r in train],
        [LABEL_TO_ID[r.vulnerability_type] for r in train],
    )
    return pipeline


def evaluate_baselines(train: list[VulnRecord], test: list[VulnRecord]) -> dict[str, dict]:
    """Score both baselines on the same test set the Transformer uses."""
    from securescan_ml.evaluation.metrics import compute_metrics
    from securescan_ml.labels import NUM_LABELS

    y_true = np.array([LABEL_TO_ID[r.vulnerability_type] for r in test])
    results: dict[str, dict] = {}

    def as_logits(preds: np.ndarray) -> np.ndarray:
        """compute_metrics takes logits; one-hot the hard predictions."""
        one_hot = np.zeros((len(preds), NUM_LABELS))
        one_hot[np.arange(len(preds)), preds] = 1.0
        return one_hot

    results["bandit"] = compute_metrics(as_logits(bandit_predict(test)), y_true)

    rf = train_random_forest(train)
    rf_preds = np.asarray(rf.predict([r.code for r in test]))
    results["random_forest"] = compute_metrics(as_logits(rf_preds), y_true)

    return results
