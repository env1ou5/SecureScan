"""Line-level localization via gradient x input attribution (proposal §8, D5).

The gradient of the predicted class logit with respect to the input embeddings,
dotted with those embeddings, gives a per-token relevance score. Token scores
are mapped to source lines through tokenizer offsets and max-pooled per line.

Chosen over supervised token tagging because it needs no line-level labels,
which Python vulnerability data largely lacks. The cost is that attribution is
noisy and correlational: it shows what drove the prediction, not a proof of
exploitability. Evaluate it with top-k line accuracy against CVEFixes diffs
(see evaluation/metrics.py::localization_accuracy) -- a plausible-looking
heatmap is not a result.

Requires a real forward/backward pass, so it runs on the un-quantized model.
INT8 dynamic quantization does not support the gradients this needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)


@dataclass
class LineAttribution:
    line: int  # absolute line number in the original file
    score: float  # normalized 0..1 within this function
    text: str = ""


def _input_embeddings(model) -> torch.nn.Module:
    return model.get_input_embeddings()


def token_attributions(
    model,
    tokenizer,
    code: str,
    target_class: int | None = None,
    max_length: int = 512,
) -> tuple[list[tuple[int, int, float]], int]:
    """Return ([(start_char, end_char, score)], predicted_class).

    Offsets are character offsets into `code`, from the fast tokenizer's
    offset mapping. Special tokens map to (0, 0) and are dropped.
    """
    model.eval()
    encoded = tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()

    embedding_layer = _input_embeddings(model)
    embeddings = embedding_layer(encoded["input_ids"]).detach().requires_grad_(True)

    inputs = {k: v for k, v in encoded.items() if k != "input_ids"}
    outputs = model(inputs_embeds=embeddings, **inputs)
    logits = outputs.logits[0]

    predicted = int(logits.argmax().item()) if target_class is None else int(target_class)

    model.zero_grad(set_to_none=True)
    logits[predicted].backward()

    if embeddings.grad is None:  # pragma: no cover - defensive
        log.warning("no gradient reached the embeddings; attribution unavailable")
        return [], predicted

    # grad x input, L2-normed over the hidden dimension -> one score per token.
    scores = (embeddings.grad[0] * embeddings[0]).norm(dim=-1)

    spans: list[tuple[int, int, float]] = []
    for (start, end), score in zip(offsets, scores.tolist()):
        if end > start:  # drop special tokens, which carry (0, 0)
            spans.append((int(start), int(end), float(score)))
    return spans, predicted


def attributions_to_lines(
    spans: list[tuple[int, int, float]],
    code: str,
    start_line: int,
    top_k: int | None = None,
) -> list[LineAttribution]:
    """Max-pool token scores per line and normalize to 0..1.

    Max rather than mean: one decisive token (a `+` concatenating into SQL) on
    an otherwise ordinary line is exactly the signal worth surfacing, and
    averaging would bury it.
    """
    if not spans:
        return []

    lines = code.splitlines()
    # Character offset at which each 0-indexed line begins.
    line_starts: list[int] = []
    cursor = 0
    for line in code.splitlines(keepends=True):
        line_starts.append(cursor)
        cursor += len(line)

    def line_index(char_offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= char_offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    best: dict[int, float] = {}
    for start, _end, score in spans:
        idx = line_index(start)
        if score > best.get(idx, float("-inf")):
            best[idx] = score

    max_score = max(best.values())
    min_score = min(best.values())
    spread = max_score - min_score

    results = [
        LineAttribution(
            line=start_line + idx,
            score=(score - min_score) / spread if spread > 0 else 1.0,
            text=lines[idx].rstrip() if idx < len(lines) else "",
        )
        for idx, score in best.items()
    ]
    results.sort(key=lambda a: (-a.score, a.line))
    return results[:top_k] if top_k else results


def localize(
    model,
    tokenizer,
    code: str,
    start_line: int,
    target_class: int | None = None,
    top_k: int = 3,
) -> tuple[list[LineAttribution], int]:
    """Convenience wrapper: code in, ranked absolute line numbers out."""
    spans, predicted = token_attributions(model, tokenizer, code, target_class)
    return attributions_to_lines(spans, code, start_line, top_k), predicted
