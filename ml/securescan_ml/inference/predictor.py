"""Serving-side inference: float32 CPU predictor (proposal §2, D4/D6).

**Quantization is off by default, and that is a measured decision, not an
oversight.** Decision D4 originally specified INT8 dynamic quantization for CPU
serving. Measured on 400 real held-out functions
(`artifacts/quantization_study.json`):

    float32            macro F1 0.703  supported 0.861  acc 0.918  260 ms/fn
    INT8 all Linear    macro F1 0.143  supported 0.286  acc 0.752  133 ms/fn
    INT8 encoder only  macro F1 0.143  supported 0.286  acc 0.752  130 ms/fn

INT8 halves latency and destroys the model: 0.752 accuracy is roughly the SAFE
base rate, i.e. it predicts SAFE for almost everything. Restricting
quantization to the encoder changes nothing, so the damage is not in the
classifier head.

This was caught by `scripts/demo_scan.py`, which runs the real checkpoint
through the real API and returned **zero findings on code with five planted
vulnerabilities**. The unit tests did not catch it because they stub the
predictor. Halving latency is not worth serving a scanner that finds nothing.

Attribution always uses the float model regardless, since quantized layers do
not support the backward pass gradient x input needs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from securescan_ml.chunking import FunctionChunk, window_oversized
from securescan_ml.inference.attribution import LineAttribution, localize
from securescan_ml.labels import ID_TO_LABEL, Label, Severity, severity_for
from securescan_ml.training.calibrate import load_temperature

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: Label
    severity: Severity
    confidence: float  # calibrated unless `calibrated` is False
    calibrated: bool
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    contributing_lines: list[LineAttribution] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def is_vulnerable(self) -> bool:
        return self.label is not Label.SAFE

    @property
    def anchor_line(self) -> int:
        """The line the finding points at: highest-attributed, else the header."""
        return self.contributing_lines[0].line if self.contributing_lines else self.start_line


class VulnerabilityPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 512,
        quantize: bool = False,
        attribution_top_k: int = 3,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.attribution_top_k = attribution_top_k

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.float_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()

        if quantize:
            quantize = getattr(
                torch.ao.quantization, "quantize_dynamic", torch.quantization.quantize_dynamic
            )
            self.model = quantize(self.float_model, {torch.nn.Linear}, dtype=torch.qint8).eval()
            log.info("loaded INT8-quantized model from %s", model_dir)
        else:
            self.model = self.float_model
            log.info("loaded float model from %s", model_dir)

        self.temperature = load_temperature(model_dir)
        self.calibrated = self.temperature != 1.0
        if not self.calibrated:
            log.warning("serving uncalibrated confidence -- run securescan_ml.training.calibrate")

    def _probabilities(self, codes: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            codes,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
        return torch.softmax(logits / self.temperature, dim=-1)

    def predict_chunks(
        self,
        chunks: list[FunctionChunk],
        batch_size: int = 16,
        with_attribution: bool = True,
    ) -> list[Prediction]:
        """Classify chunks, then attribute only the ones that were flagged."""
        if not chunks:
            return []

        # Split anything over the window, keeping a link back to the parent.
        expanded: list[FunctionChunk] = []
        parents: list[int] = []
        for i, chunk in enumerate(chunks):
            windows = window_oversized(chunk, self.tokenizer, self.max_length)
            expanded.extend(windows)
            parents.extend([i] * len(windows))

        all_probs: list[torch.Tensor] = []
        for i in range(0, len(expanded), batch_size):
            all_probs.append(self._probabilities([c.code for c in expanded[i : i + batch_size]]))
        probs = torch.cat(all_probs)

        # Merge windows back: take the window with the highest non-SAFE mass,
        # since one vulnerable window makes the whole function vulnerable.
        best_for_parent: dict[int, tuple[int, float]] = {}
        for row, parent in enumerate(parents):
            vuln_mass = float(probs[row][1:].sum())
            if parent not in best_for_parent or vuln_mass > best_for_parent[parent][1]:
                best_for_parent[parent] = (row, vuln_mass)

        predictions: list[Prediction] = []
        for parent_idx, (row, _mass) in sorted(best_for_parent.items()):
            chunk = expanded[row]
            row_probs = probs[row]
            class_id = int(row_probs.argmax())
            label = ID_TO_LABEL[class_id]

            contributing: list[LineAttribution] = []
            if with_attribution and label is not Label.SAFE:
                try:
                    contributing, _ = localize(
                        self.float_model,
                        self.tokenizer,
                        chunk.code,
                        chunk.start_line,
                        target_class=class_id,
                        top_k=self.attribution_top_k,
                    )
                except Exception:  # noqa: BLE001 - a finding without lines beats no finding
                    log.exception("attribution failed for %s:%s", chunk.file_path, chunk.name)

            parent = chunks[parent_idx]
            predictions.append(
                Prediction(
                    label=label,
                    severity=severity_for(label),
                    confidence=float(row_probs[class_id]),
                    calibrated=self.calibrated,
                    file_path=parent.file_path,
                    function_name=parent.name,
                    start_line=parent.start_line,
                    end_line=parent.end_line,
                    contributing_lines=contributing,
                    probabilities={ID_TO_LABEL[i].value: float(p) for i, p in enumerate(row_probs)},
                )
            )
        return predictions

    def predict_source(self, source: str, file_path: str, **kwargs) -> list[Prediction]:
        from securescan_ml.chunking import extract_analyzable_chunks

        return self.predict_chunks(extract_analyzable_chunks(source, file_path), **kwargs)

    def warmup(self) -> float:
        """Run one throwaway inference so the first real scan is not the slow one.

        Returns elapsed seconds; the worker logs it at startup.
        """
        started = time.perf_counter()
        self._probabilities(["def _warmup():\n    return 0\n"])
        return time.perf_counter() - started
