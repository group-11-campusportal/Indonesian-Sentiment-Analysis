"""Shared evaluation metrics for the sentiment models."""

from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .transformer_models import ID2LABEL, LABEL2ID, NUM_LABELS

METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1")
LABEL_ORDER = list(LABEL2ID)          # ["negative", "neutral", "positive"]
LABEL_IDS = list(range(NUM_LABELS))   # [0, 1, 2]


def _to_ids(labels: Iterable) -> np.ndarray:
    """Normalize class names or integer ids to an int id array."""
    values = list(labels)
    if not values:
        raise ValueError("labels must not be empty")

    ids = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Label must be a class name or integer id, got {value!r}")
        if isinstance(value, str):
            key = value.strip().lower()
            if key not in LABEL2ID:
                raise ValueError(f"Unknown label name: {value!r}; expected one of {LABEL_ORDER}")
            ids.append(LABEL2ID[key])
        elif isinstance(value, (int, np.integer)):
            as_int = int(value)
            if as_int not in ID2LABEL:
                raise ValueError(f"Label id out of range: {value!r}; expected 0..{NUM_LABELS - 1}")
            ids.append(as_int)
        else:
            raise ValueError(f"Label must be a class name or integer id, got {value!r}")
    return np.asarray(ids, dtype=int)


def _aligned_ids(y_true: Iterable, y_pred: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both label sequences and check equal length."""
    true_ids = _to_ids(y_true)
    pred_ids = _to_ids(y_pred)
    if true_ids.shape != pred_ids.shape:
        raise ValueError(f"y_true and y_pred length mismatch: {true_ids.size} vs {pred_ids.size}")
    return true_ids, pred_ids


def predicted_ids(scores) -> np.ndarray:
    """Argmax logits/probabilities of shape (n, NUM_LABELS) to class ids."""
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != NUM_LABELS:
        raise ValueError(f"scores must be shaped (n_samples, {NUM_LABELS}), got {arr.shape}")
    return arr.argmax(axis=1).astype(int)


def compute_metrics(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    """Accuracy, macro-F1, and weighted-F1 for one set of predictions."""
    true_ids, pred_ids = _aligned_ids(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(true_ids, pred_ids)),
        "macro_f1": float(
            f1_score(true_ids, pred_ids, labels=LABEL_IDS, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(true_ids, pred_ids, labels=LABEL_IDS, average="weighted", zero_division=0)
        ),
    }


def per_class_metrics(y_true: Iterable, y_pred: Iterable) -> dict[str, dict[str, float]]:
    """Precision, recall, F1, and support per class, keyed by label name."""
    true_ids, pred_ids = _aligned_ids(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids, pred_ids, labels=LABEL_IDS, zero_division=0
    )
    return {
        ID2LABEL[class_id]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_id in enumerate(LABEL_IDS)
    }


def confusion(y_true: Iterable, y_pred: Iterable) -> np.ndarray:
    """Confusion matrix in shared label order (rows=true, cols=pred)."""
    true_ids, pred_ids = _aligned_ids(y_true, y_pred)
    return confusion_matrix(true_ids, pred_ids, labels=LABEL_IDS)


def aggregate_runs(runs: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Per-seed metric dicts to mean and sample std per metric."""
    runs = list(runs)
    if not runs:
        raise ValueError("aggregate_runs needs at least one run")
    keys = set(runs[0])
    if not keys:
        raise ValueError("run metrics must not be empty")
    for run in runs:
        if set(run) != keys:
            raise ValueError("every run must report the same metric keys")

    summary = {}
    for key in runs[0]:
        values = [float(run[key]) for run in runs]
        summary[key] = {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}
    return summary


def format_aggregate(summary: Mapping[str, Mapping[str, float]], digits: int = 4) -> dict[str, str]:
    """Render an aggregate summary as 'mean +/- std' strings."""
    return {
        key: f"{stats['mean']:.{digits}f} +/- {stats['std']:.{digits}f}"
        for key, stats in summary.items()
    }


def _load_predictions(path, true_col: str, pred_col: str):
    """Read gold and predicted labels from one prediction CSV."""
    import pandas as pd

    frame = pd.read_csv(path)
    missing = {true_col, pred_col} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    return frame[true_col].tolist(), frame[pred_col].tolist()


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate sentiment predictions across seeds.")
    parser.add_argument("predictions", type=Path, nargs="+", help="Prediction CSV files (one per seed).")
    parser.add_argument("--true-col", default="label", help="Gold-label column (default: label).")
    parser.add_argument("--pred-col", default="prediction", help="Predicted-label column (default: prediction).")
    parser.add_argument("--digits", type=int, default=4, help="Decimal places (default: 4).")
    args = parser.parse_args()

    runs = []
    try:
        for path in args.predictions:
            y_true, y_pred = _load_predictions(path, args.true_col, args.pred_col)
            metrics = compute_metrics(y_true, y_pred)
            runs.append(metrics)

            print(f"\n== {path} ==")
            for key in METRIC_KEYS:
                print(f"  {key:<12} {metrics[key]:.{args.digits}f}")
            print("  per-class F1:")
            for label, scores in per_class_metrics(y_true, y_pred).items():
                print(f"    {label:<9} f1={scores['f1']:.{args.digits}f}  support={scores['support']}")
            print(f"  confusion (rows=true, cols=pred; order {LABEL_ORDER}):")
            for label, row in zip(LABEL_ORDER, confusion(y_true, y_pred)):
                print(f"    {label:<9} {row.tolist()}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")

    if len(runs) > 1:
        summary = format_aggregate(aggregate_runs(runs), digits=args.digits)
        print(f"\n== Aggregate over {len(runs)} runs (mean +/- std) ==")
        for key in METRIC_KEYS:
            print(f"  {key:<12} {summary[key]}")


if __name__ == "__main__":
    main()
