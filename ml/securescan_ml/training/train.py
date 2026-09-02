"""Fine-tune UniXcoder for 7-way vulnerability classification (proposal §4).

    python -m securescan_ml.training.train \
        --data datasets/normalized.jsonl --output artifacts/unixcoder-v1

An explicit PyTorch loop rather than the HuggingFace Trainer. The Trainer's
argument surface churns between major versions (`warmup_ratio` and
`evaluation_strategy` both disappeared in transformers 5.x), and the loop here
is ~100 lines with no hidden behavior -- which matters for a project whose
point is the machine learning engineering, not the convenience wrapper.

Runs on a rented GPU by the hour (decision D4). A 125M encoder over this corpus
is a sub-hour job.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from securescan_ml.data.schema import read_jsonl
from securescan_ml.data.splits import class_weights, make_splits
from securescan_ml.evaluation.metrics import (
    compute_metrics,
    confusion_matrix,
    format_confusion,
)
from securescan_ml.labels import LABEL_ORDER, LABEL_TO_ID, NUM_LABELS

log = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/unixcoder-base"


class VulnDataset(Dataset):
    """Holds raw code; tokenization happens in the collate function.

    Padding to the longest sequence *in the batch* rather than to max_length is
    a large win here: the median function is far shorter than 512 tokens, so
    fixed-length padding would spend most of the compute on padding.
    """

    def __init__(self, records):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        return rec.code, LABEL_TO_ID[rec.vulnerability_type]


class Collator:
    """Tokenize a batch, padding only to the batch's longest sequence."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        codes, labels = zip(*batch)
        enc = self.tokenizer(
            list(codes),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            logits = model(**batch).logits
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return compute_metrics(logits, labels), logits, labels


def lr_lambda_factory(total_steps: int, warmup_ratio: float):
    """Linear warmup then linear decay -- the schedule Trainer used to provide."""
    warmup = max(int(total_steps * warmup_ratio), 1)

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        remaining = max(total_steps - warmup, 1)
        return max(0.0, (total_steps - step) / remaining)

    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="artifacts/unixcoder-v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--patience",
        type=int,
        default=2,
        help="stop after this many epochs without validation improvement (0 = never)",
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="improvement below this counts as no improvement",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 on Ampere and newer; fp16 elsewhere; nothing on CPU.
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None
    log.info("device=%s amp=%s", device, amp_dtype)

    records = list(read_jsonl(args.data))
    splits = make_splits(records)
    log.info("loaded %d records\n%s", len(records), splits.summary())
    if not splits.train or not splits.validation:
        raise SystemExit("train or validation split is empty")

    # Checkpoint selection is only as meaningful as the validation set. When
    # validation is overwhelmingly synthetic, its macro F1 saturates early and
    # stops discriminating between checkpoints -- the score says the model is
    # excellent at reproducing the generator, which is not the claim we want to
    # make. Surface that rather than let a 0.99 go unexamined.
    synthetic_share = sum(r.is_synthetic for r in splits.validation) / len(splits.validation)
    if synthetic_share > 0.8:
        log.warning(
            "validation is %.0f%% synthetic -- its macro F1 will saturate and is a "
            "weak checkpoint-selection signal. Treat test_real as the real result.",
            synthetic_share * 100,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label={i: lab.value for i, lab in enumerate(LABEL_ORDER)},
        label2id={lab.value: i for i, lab in enumerate(LABEL_ORDER)},
    ).to(device)

    collate = Collator(tokenizer, args.max_length)

    def loader(subset, batch_size, shuffle):
        return DataLoader(
            VulnDataset(subset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate,
        )

    train_loader = loader(splits.train, args.batch_size, True)
    val_loader = loader(splits.validation, args.eval_batch_size, False)

    # SAFE dominates; unweighted, the model can score well on accuracy while
    # never predicting a rare class (proposal §4).
    weights = torch.tensor(
        class_weights(splits.train, LABEL_ORDER), dtype=torch.float, device=device
    )
    log.info("class weights: %s", [round(w, 3) for w in weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=weights)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if any(k in name for k in ("bias", "LayerNorm.weight")) else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(total_steps, args.warmup_ratio)
    )
    scaler = torch.amp.GradScaler(enabled=amp_dtype is torch.float16)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, started = 0.0, 0, time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                logits = model(**batch).logits
                loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % 50 == 0:
                log.info(
                    "epoch %d step %d/%d loss %.4f lr %.2e",
                    epoch,
                    step,
                    len(train_loader),
                    running / seen,
                    scheduler.get_last_lr()[0],
                )

        metrics, _, _ = evaluate(model, val_loader, device, amp_dtype)
        elapsed = time.perf_counter() - started
        log.info(
            "epoch %d done in %.1fs | train_loss %.4f | val macro_f1 %.4f acc %.4f fpr %.4f",
            epoch,
            elapsed,
            running / seen,
            metrics["macro_f1"],
            metrics["accuracy"],
            metrics["false_positive_rate"],
        )
        history.append({"epoch": epoch, "train_loss": running / seen, **metrics})

        if metrics["macro_f1"] > best_f1 + args.min_delta:
            best_f1 = metrics["macro_f1"]
            epochs_without_improvement = 0
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            log.info("  new best macro_f1=%.4f -- checkpoint saved", best_f1)
        else:
            epochs_without_improvement += 1
            log.info(
                "  no improvement over %.4f (%d/%d epochs)",
                best_f1,
                epochs_without_improvement,
                args.patience,
            )
            if args.patience and epochs_without_improvement >= args.patience:
                log.info("early stopping: validation stopped improving")
                break

    # Reload the best checkpoint; the last epoch is not necessarily the best.
    model = AutoModelForSequenceClassification.from_pretrained(out).to(device)

    results = {"history": history, "best_val_macro_f1": best_f1}
    for name in ("test_full", "test_real"):
        subset = getattr(splits, name)
        if not subset:
            log.warning("%s is empty -- skipping", name)
            continue
        metrics, logits, labels = evaluate(
            model, loader(subset, args.eval_batch_size, False), device, amp_dtype
        )
        results[name] = {"n": len(subset), **metrics}
        log.info("%s (n=%d) macro_f1=%.4f", name, len(subset), metrics["macro_f1"])
        log.info("\n%s", format_confusion(confusion_matrix(labels, logits.argmax(-1))))

    (out / "test_results.json").write_text(json.dumps(results, indent=2, default=float))
    log.info("wrote %s", out / "test_results.json")
    log.info(
        "NEXT: python -m securescan_ml.training.calibrate --model %s --data %s", out, args.data
    )


if __name__ == "__main__":
    main()
