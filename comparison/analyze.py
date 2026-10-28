"""Compare saved Method A/B predictions without loading or running models.

Usage from the repository root:
    python comparison/analyze.py --output-dir outputs
"""

import argparse
import csv
import json
from math import isclose
from pathlib import Path
from statistics import mean, stdev

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "method_a": "xlm-roberta-base",
    "method_b": "indobenchmark/indobert-base-p1",
}
SEEDS = (42, 123, 999)
LABELS = ("negative", "neutral", "positive")
# Each split must contain the original examples, including its neutral support.
SPLITS = {"test": (500, 88), "val": (1260, 131)}
OVERALL_METRICS = ("accuracy", "macro_f1", "weighted_f1")
TABLE_METRICS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "neutral_precision": "Neutral precision",
    "neutral_recall": "Neutral recall",
    "neutral_f1": "Neutral F1",
}


def read_predictions(path: Path, split: str) -> list[dict]:
    """Reject incomplete splits, invalid labels, and missing text columns."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"text", "label", "prediction"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: requires text, label, prediction columns")
        rows = list(reader)
    expected_rows, expected_neutral = SPLITS[split]
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    for line, row in enumerate(rows, start=2):
        if not row["text"] or not row["text"].strip():
            raise ValueError(f"{path}:{line}: text must be nonempty")
        if row["label"] not in LABELS or row["prediction"] not in LABELS:
            raise ValueError(f"{path}:{line}: labels must be one of {LABELS}")
    neutral = sum(row["label"] == "neutral" for row in rows)
    if neutral != expected_neutral:
        raise ValueError(f"{path}: expected {expected_neutral} neutral rows, found {neutral}")
    return rows


def calculate_metrics(rows: list[dict]) -> dict[str, float]:
    """Evaluate all classes; extract neutral precision, recall, and F1."""
    truth = [row["label"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, zero_division=0
    )
    neutral = LABELS.index("neutral")
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(
            truth, predictions, labels=LABELS, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            truth, predictions, labels=LABELS, average="weighted", zero_division=0
        )),
        "neutral_precision": float(precision[neutral]),
        "neutral_recall": float(recall[neutral]),
        "neutral_f1": float(f1[neutral]),
        "neutral_missed": sum(
            row["label"] == "neutral" and row["prediction"] != "neutral" for row in rows
        ),
    }


def load_runs(output_dir: Path) -> tuple[dict, dict]:
    """Require all six runs and verify their metrics and evaluation examples."""
    runs, summaries, reference = {}, {}, {}
    reference_splits = None
    for method, model in MODELS.items():
        for seed in SEEDS:
            run_dir = output_dir / method / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (summary["method"], summary["model"], summary["seed"]) != (method, model, seed):
                raise ValueError(f"{summary_path}: method, model, or seed does not match its folder")
            splits = summary["splits"]
            if splits["train"]["rows"] != 10984:
                raise ValueError(f"{summary_path}: expected 10984 training rows")
            if reference_splits is None:
                reference_splits = splits
            if splits != reference_splits:
                raise ValueError(f"{summary_path}: split counts or hashes differ between runs")
            summaries[method, seed] = summary
            for split, (expected_rows, _) in SPLITS.items():
                if splits[split]["rows"] != expected_rows:
                    raise ValueError(f"{summary_path}: incorrect {split} row count")
                path = run_dir / f"{split}_predictions.csv"
                rows = read_predictions(path, split)
                examples = [(row["text"], row["label"]) for row in rows]
                if split not in reference:
                    reference[split] = examples
                if examples != reference[split]:
                    raise ValueError(f"{path}: text/label rows differ between runs (including order)")
                metrics = calculate_metrics(rows)
                for key in OVERALL_METRICS:
                    saved = summary["metrics"][split][key]
                    if not isclose(metrics[key], saved, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError(
                            f"{path}: recalculated {key}={metrics[key]:.10f} "
                            f"does not match summary={saved:.10f}"
                        )
                runs[method, seed, split] = metrics
    return runs, summaries


def score(values: list[float]) -> str:
    """Display percentages with sample SD across seeds, rather than pooled rows."""
    return f"{100 * mean(values):.2f} ± {100 * stdev(values):.2f}"


def print_comparison(runs: dict, summaries: dict) -> None:
    print("Verified all 6 runs; prediction metrics match their summaries.")
    print("Each run: 500 test rows (88 neutral), 1260 validation rows (131 neutral).")
    print("Scores are percentages, mean ± sample SD over seeds 42, 123, 999.")
    print("Differences are B − A in percentage points (pp); positive favors B.")
    for split in SPLITS:
        print(f"\n### {'Test' if split == 'test' else 'Validation'} comparison\n")
        print("| Metric | Method A | Method B | B − A (pp) |")
        print("| :--- | ---: | ---: | ---: |")
        for key, title in TABLE_METRICS.items():
            a = [runs["method_a", seed, split][key] for seed in SEEDS]
            b = [runs["method_b", seed, split][key] for seed in SEEDS]
            print(f"| {title} | {score(a)} | {score(b)} | {100 * (mean(b) - mean(a)):+.2f} |")
        print("\nNeutral F1 per seed (%):\n")
        print("| Seed | Method A | Method B | B − A (pp) |")
        print("| ---: | ---: | ---: | ---: |")
        for seed in SEEDS:
            a = runs["method_a", seed, split]["neutral_f1"]
            b = runs["method_b", seed, split]["neutral_f1"]
            print(f"| {seed} | {100 * a:.2f} | {100 * b:.2f} | {100 * (b - a):+.2f} |")
        misses = {
            method: "/".join(str(runs[method, seed, split]["neutral_missed"]) for seed in SEEDS)
            for method in MODELS
        }
        print(f"\nNeutral examples missed (42/123/999): A={misses['method_a']}; B={misses['method_b']}.")

    print("\n### Recorded training cost\n")
    print("| Method | Parameters | Training + validation seconds (mean ± SD) |")
    print("| :--- | ---: | ---: |")
    for method in MODELS:
        params = {summaries[method, seed]["parameters"]["total"] for seed in SEEDS}
        if len(params) != 1:
            raise ValueError(f"{method}: parameter counts differ between seeds")
        seconds = [summaries[method, seed]["training_seconds"] for seed in SEEDS]
        print(f"| {method} | {next(iter(params)):,} | {mean(seconds):.1f} ± {stdev(seconds):.1f} |")
    print("\nTiming excludes downloads, checkpoint writes, and final inference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs",
                        help="Existing outputs/<method>/seed_<seed>/ folders")
    args = parser.parse_args()
    try:
        runs, summaries = load_runs(args.output_dir)
        print_comparison(runs, summaries)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
