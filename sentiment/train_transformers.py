"""Train one transformer/seed and export its selected checkpoint and predictions."""

import argparse
import csv
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, set_seed

from .evaluation import compute_metrics
from .transformer_models import (
    ID2LABEL, LABEL2ID, count_parameters, load_method_a, load_method_b,
    tokenize_batch,
)

ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 5
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def select_device(name):
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if cuda else "mps" if mps else "cpu"
    if (name == "cuda" and not cuda) or (name == "mps" and not mps):
        raise ValueError(f"Requested device {name!r} is unavailable")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def read_split(path):
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"text", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError(f"{path} must contain nonempty 'text' and 'label' columns")
    if not frame["label"].isin(LABEL2ID).all():
        raise ValueError(f"{path}: labels must be one of {list(LABEL2ID)}")
    if not frame["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError(f"{path}: texts must be nonempty strings")
    return frame


def make_loader(frame, tokenizer, shuffle=False, seed=42):
    def collate(rows):
        inputs = tokenize_batch([row["text"] for row in rows], tokenizer)
        labels = torch.tensor([LABEL2ID[row["label"]] for row in rows], dtype=torch.long)
        return inputs, labels

    return DataLoader(
        frame.to_dict("records"), batch_size=BATCH_SIZE, shuffle=shuffle,
        collate_fn=collate, generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Accumulate summed losses, normalizing by the actual group weight.

    This also handles a short last group and class-weighted cross-entropy
    without giving small microbatches disproportionate influence.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = total_weight = group_weight = 0.0
    truth, predictions = [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, (inputs, labels) in enumerate(loader, start=1):
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs).logits
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced nonfinite logits")
            loss = F.cross_entropy(logits, labels, weight=loss_fn.weight, reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            weight = (
                loss_fn.weight[labels].sum().item()
                if loss_fn.weight is not None else len(labels)
            )
            total_loss += loss.item()
            total_weight += weight
            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            if training:
                loss.backward()
                group_weight += weight
                if step % ACCUMULATION_STEPS == 0 or step == len(loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_weight)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    group_weight = 0.0
    return {"loss": total_loss / total_weight, **compute_metrics(truth, predictions)}, predictions


def train(method, seed, data_dir, output_dir, device_name="auto"):
    device = select_device(device_name)
    paths = {split: data_dir/ f"{split}_split.csv" for split in ("train", "val", "test")}
    frames = {split: read_split(path) for split, path in paths.items()}
    run_dir = output_dir / f"method_{method}" / f"seed_{seed}"
    # Claim a fresh directory before loading weights; never overwrite a past run.
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Method {method.upper()} | seed {seed} | device {device} | output {run_dir}", flush=True)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if method == "a":
        model, tokenizer, loss_fn = load_method_a()
    else:
        model, tokenizer, loss_fn = load_method_b(frames["train"]["label"])
    parameters = count_parameters(model)
    model.to(device)
    loss_fn.to(device)
    loaders = {
        split: make_loader(frame, tokenizer, shuffle=split == "train", seed=seed)
        for split, frame in frames.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    checkpoint = run_dir / "best_checkpoint"
    best_score, best_epoch, runtime = -1.0, 0, 0.0
    with (run_dir / "history.csv").open("w", newline="") as handle:
        keys = ["loss", "accuracy", "macro_f1", "weighted_f1"]
        fields = ["epoch"] + [
            f"{split}_{key}" for split in ("train", "val") for key in keys
        ] + ["epoch_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for epoch in range(1, EPOCHS + 1):
            synchronize(device)
            start = perf_counter()
            train_metrics, _ = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
            val_metrics, _ = run_epoch(model, loaders["val"], loss_fn, device)
            synchronize(device)
            elapsed = perf_counter() - start
            runtime += elapsed
            row = {"epoch": epoch, "epoch_seconds": elapsed}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["macro_f1"] > best_score:
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                best_score, best_epoch = val_metrics["macro_f1"], epoch
            writer.writerow(row)
            handle.flush()
            print(f"Epoch {epoch}/{EPOCHS} | train loss {train_metrics['loss']:.4f} | "
                  f"val Macro-F1 {val_metrics['macro_f1']:.4f} | {elapsed:.1f}s", flush=True)

    # Release optimizer state and the final epoch's weights before reloading.
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    final_metrics = {}
    for split in ("val", "test"):
        metrics, predictions = run_epoch(model, loaders[split], loss_fn, device)
        final_metrics[split] = metrics
        output = frames[split][["text", "label"]].copy()
        output["prediction"] = [ID2LABEL[index] for index in predictions]
        output.to_csv(run_dir / f"{split}_predictions.csv", index=False)

    summary = {
        "method": f"method_{method}",
        "model": "xlm-roberta-base" if method == "a" else "indobenchmark/indobert-base-p1",
        "seed": seed, "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda"
            else "Apple MPS" if device.type == "mps" else None
        ),
        "parameters": parameters, "selected_epoch": best_epoch,
        "training_seconds": runtime,
        "timing": (
            "Sum of training and validation epoch durations; excludes downloads, "
            "checkpoint writes, and final inference"
        ),
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "max_grad_norm": MAX_GRAD_NORM,
            "max_length": 256, "label2id": LABEL2ID,
            "class_weights": (
                loss_fn.weight.detach().cpu().tolist()
                if loss_fn.weight is not None else None
            ),
        },
        "splits": {
            split: {
                "rows": len(frames[split]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for split, path in paths.items()
        },
        "versions": {
            package: version(package)
            for package in ("torch", "transformers", "numpy", "pandas", "scikit-learn")
        },
        "metrics": final_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Complete: {run_dir} | selected epoch {best_epoch}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 999), required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    try:
        train(args.method, args.seed, args.data_dir, args.output_dir, args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
