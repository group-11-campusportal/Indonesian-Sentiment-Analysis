"""Gather every run's summary.json and report mean +/- std per model.

Expected layout (written by train_transformers.py):
    outputs/<model>/seed_<seed>/summary.json
for example outputs/method_a/seed_42/summary.json. Each summary.json holds
the final Val and Test scores of the checkpoint picked by validation Macro-F1.

Usage:
    python -m sentiment.aggregate_results
    python -m sentiment.aggregate_results --output-dir outputs
"""

import argparse                          # reads options typed on the command line
import csv                               # writes the result tables as .csv files
import json                              # reads each run's summary.json file
from pathlib import Path                 # handles folder and file paths safely
from statistics import mean, stdev       # average and sample standard deviation

ROOT = Path(__file__).resolve().parent.parent   # the repository's top folder

# The plan: 3 models x 3 seeds = 9 runs. The baseline folder name assumes the
# baseline trainer will follow the same outputs/<model>/seed_<seed>/ layout.
EXPECTED_MODELS = ("baseline", "method_a", "method_b")
EXPECTED_SEEDS = (42, 123, 999)
SPLITS = ("val", "test")                         # the two sets we report on
METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1")   # the paper's metrics


def find_runs(output_dir: Path) -> dict[str, dict[int, dict]]:
    """Load every outputs/<model>/seed_<seed>/summary.json into a nested dict.

    The result looks like {"method_a": {42: {...summary...}, 123: {...}}, ...}.
    """
    runs: dict[str, dict[int, dict]] = {}                    # start empty
    # glob finds every file matching the pattern, one folder level per "*"
    for path in sorted(output_dir.glob("*/seed_*/summary.json")):
        model = path.parent.parent.name                       # e.g. "method_a"
        seed_text = path.parent.name.removeprefix("seed_")    # "seed_42" -> "42"
        if not seed_text.isdigit():                           # skip odd folders
            print(f"Skipping {path}: folder name is not seed_<number>")
            continue
        summary = json.loads(path.read_text())                # read the JSON file
        if "metrics" not in summary:                          # must hold scores
            raise ValueError(f"{path} has no 'metrics' section")
        runs.setdefault(model, {})[int(seed_text)] = summary  # store under model+seed
    return runs


def report_missing(runs: dict[str, dict[int, dict]]) -> list[str]:
    """List every expected (model, seed) pair that has no summary.json yet."""
    missing = []
    for model in EXPECTED_MODELS:                    # check each planned model
        for seed in EXPECTED_SEEDS:                  # ...and each planned seed
            if seed not in runs.get(model, {}):      # no file found for this pair
                missing.append(f"{model}/seed_{seed}")
    return missing


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample std). One value has no spread, so std is 0."""
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def per_run_rows(runs: dict[str, dict[int, dict]]) -> list[dict]:
    """One flat row per run: model, seed, then every split/metric score."""
    rows = []
    for model, by_seed in sorted(runs.items()):          # models in A-Z order
        for seed, summary in sorted(by_seed.items()):     # seeds low to high
            row = {"model": model, "seed": seed}
            for split in SPLITS:                          # val, then test
                for key in METRIC_KEYS:                   # accuracy, macro_f1, ...
                    row[f"{split}_{key}"] = summary["metrics"][split][key]
            row["training_seconds"] = summary.get("training_seconds")       # time cost
            row["parameters"] = summary.get("parameters", {}).get("total")  # model size
            rows.append(row)
    return rows


def aggregate_rows(rows: list[dict], digits: int) -> list[dict]:
    """Collapse the per-run rows into one mean +/- std row per model."""
    columns = [f"{split}_{key}" for split in SPLITS for key in METRIC_KEYS]
    columns.append("training_seconds")
    table = []
    models = sorted({row["model"] for row in rows})               # unique model names
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]   # this model's seeds
        summary = {"model": model, "n_seeds": len(model_rows)}
        for column in columns:
            values = [row[column] for row in model_rows if row[column] is not None]
            if not values:                                        # nothing recorded
                summary[column] = ""
                continue
            avg, spread = mean_std([float(value) for value in values])
            summary[f"{column}_mean"] = avg                       # raw numbers for plots
            summary[f"{column}_std"] = spread
            # the human-readable form used in the paper's results table;
            # seconds only need 1 decimal place, scores use --digits
            places = 1 if column == "training_seconds" else digits
            summary[column] = f"{avg:.{places}f} +/- {spread:.{places}f}"
        # parameter count is the same for every seed, so report it once
        summary["parameters"] = model_rows[0]["parameters"]
        table.append(summary)
    return table


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save a list of dicts as a CSV file, one dict per line."""
    fields = list(dict.fromkeys(key for row in rows for key in row))   # all column names, in order
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()          # first line: the column names
        writer.writerows(rows)        # then one line per row


def print_table(table: list[dict]) -> None:
    """Print the mean +/- std table as Markdown, ready to paste or read."""
    for split in SPLITS:
        print(f"\n### {split.capitalize()} set (mean +/- std over seeds)\n")
        print("| Model | Seeds | Accuracy | Macro-F1 | Weighted-F1 |")
        print("| :--- | :---: | :---: | :---: | :---: |")
        for row in table:
            scores = " | ".join(row[f"{split}_{key}"] for key in METRIC_KEYS)
            print(f"| {row['model']} | {row['n_seeds']} | {scores} |")
    print("\n### Cost\n")
    print("| Model | Parameters | Training seconds |")
    print("| :--- | ---: | :---: |")
    for row in table:
        params = f"{row['parameters']:,}" if row["parameters"] else "?"   # 1,234,567 style
        print(f"| {row['model']} | {params} | {row['training_seconds'] or '?'} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs",
                        help="Folder that holds <model>/seed_<seed>/summary.json")
    parser.add_argument("--digits", type=int, default=4, help="Decimal places (default: 4)")
    args = parser.parse_args()

    try:
        runs = find_runs(args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Aggregation failed: {exc}\n")
    if not runs:
        parser.exit(1, f"No summary.json files found under {args.output_dir}\n")

    found = sum(len(by_seed) for by_seed in runs.values())    # how many runs we have
    print(f"Found {found} run(s) under {args.output_dir}")
    missing = report_missing(runs)
    if missing:   # warn, but still aggregate whatever exists
        print(f"WARNING: {len(missing)} expected run(s) missing: {', '.join(missing)}")

    try:
        rows = per_run_rows(runs)
    except KeyError as exc:   # a summary.json without val/test scores
        parser.exit(1, f"Aggregation failed: a summary.json is missing {exc}\n")
    table = aggregate_rows(rows, args.digits)

    write_csv(args.output_dir / "all_runs.csv", rows)          # every seed, raw scores
    write_csv(args.output_dir / "results_table.csv", table)    # mean +/- std per model
    print_table(table)
    print(f"\nSaved {args.output_dir / 'all_runs.csv'} and {args.output_dir / 'results_table.csv'}")


if __name__ == "__main__":
    main()
