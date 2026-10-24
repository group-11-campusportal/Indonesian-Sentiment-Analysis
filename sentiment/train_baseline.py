"""Train one Baseline BiLSTM seed and export its selected checkpoint and predictions."""

import argparse
import csv
import gc
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .baseline_model import BiLSTMSentiment, load_baseline
from .baseline_tokenizer import WordVocabulary
from .evaluation import compute_metrics
from .transformer_models import ID2LABEL, LABEL2ID, count_parameters

ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 128


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if cuda else "mps" if mps else "cpu"
    if (name == "cuda" and not cuda) or (name == "mps" and not mps):
        raise ValueError(f"Requested device {name!r} is unavailable")
    return torch.device(name)


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"text", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError(f"{path} must contain nonempty 'text' and 'label' columns")
    if not frame["label"].isin(LABEL2ID).all():
        raise ValueError(f"{path}: labels must be one of {list(LABEL2ID)}")
    if not frame["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError(f"{path}: texts must be nonempty strings")
    return frame


def make_loader(frame: pd.DataFrame, vocab: WordVocabulary, shuffle: bool = False, seed: int = 42) -> DataLoader:
    def collate(rows):
        input_ids = torch.tensor(
            [vocab.encode(row["text"], max_length=MAX_LENGTH) for row in rows],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [LABEL2ID[row["label"]] for row in rows],
            dtype=torch.long,
        )
        return input_ids, labels

    return DataLoader(
        frame.to_dict("records"),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    truth, predictions = [], []

    if training:
        optimizer.zero_grad()

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced non-finite logits")

            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered")

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics = {"loss": avg_loss, **compute_metrics(truth, predictions)}
    return metrics, predictions


def train(seed: int, data_dir: Path, output_dir: Path, device_name: str = "auto") -> Path:
    device = select_device(device_name)
    paths = {split: data_dir / f"{split}_split.csv" for split in ("train", "val", "test")}
    frames = {split: read_split(path) for split, path in paths.items()}

    run_dir = output_dir / "baseline" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model BASELINE | seed {seed} | device {device} | output {run_dir}", flush=True)

    set_all_seeds(seed)

    # 1. Build vocabulary strictly from training texts
    vocab = WordVocabulary(min_freq=2)
    vocab.build_vocab(frames["train"]["text"])
    print(f"Vocabulary size: {len(vocab):,} tokens", flush=True)

    # 2. Instantiate model and unweighted cross-entropy loss
    model, loss_fn = load_baseline(vocab_size=len(vocab))
    parameters = count_parameters(model)
    model.to(device)
    loss_fn.to(device)

    loaders = {
        split: make_loader(frame, vocab, shuffle=(split == "train"), seed=seed)
        for split, frame in frames.items()
    }

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    checkpoint_dir = run_dir / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_model_path = checkpoint_dir / "model.pt"

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
                torch.save(model.state_dict(), checkpoint_model_path)
                best_score, best_epoch = val_metrics["macro_f1"], epoch

            writer.writerow(row)
            handle.flush()
            print(
                f"Epoch {epoch}/{EPOCHS} | train loss {train_metrics['loss']:.4f} | "
                f"val Macro-F1 {val_metrics['macro_f1']:.4f} | {elapsed:.1f}s",
                flush=True,
            )

    # Save vocabulary for checkpoint reproducibility
    vocab_info = {
        "pad_token": vocab.pad_token,
        "unk_token": vocab.unk_token,
        "min_freq": vocab.min_freq,
        "w2i": vocab.w2i,
    }
    with (checkpoint_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab_info, f, indent=2)

    # Reload best model weights for final test/val inference
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, _ = load_baseline(vocab_size=len(vocab))
    model.load_state_dict(torch.load(checkpoint_model_path, map_location=device))
    model.to(device)

    final_metrics = {}
    for split in ("val", "test"):
        metrics, predictions = run_epoch(model, loaders[split], loss_fn, device)
        final_metrics[split] = metrics
        output = frames[split][["text", "label"]].copy()
        output["prediction"] = [ID2LABEL[idx] for idx in predictions]
        output.to_csv(run_dir / f"{split}_predictions.csv", index=False)

    summary = {
        "method": "baseline",
        "model": "bilstm_custom",
        "seed": seed,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda"
            else "Apple MPS" if device.type == "mps" else None
        ),
        "parameters": parameters,
        "selected_epoch": best_epoch,
        "training_seconds": runtime,
        "timing": (
            "Sum of training and validation epoch durations; excludes checkpoint "
            "writes and final inference"
        ),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "max_length": MAX_LENGTH,
            "vocab_size": len(vocab),
            "label2id": LABEL2ID,
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
            for package in ("torch", "numpy", "pandas", "scikit-learn")
        },
        "metrics": final_metrics,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Complete: {run_dir} | selected epoch {best_epoch} | val Macro-F1: {final_metrics['val']['macro_f1']:.4f} | test Macro-F1: {final_metrics['test']['macro_f1']:.4f}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 123, 999), required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    try:
        train(args.seed, args.data_dir, args.output_dir, args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
