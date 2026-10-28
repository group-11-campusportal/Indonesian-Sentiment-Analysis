"""Milestone 4 figures: loss curves, confusion matrix, and misclassified examples.

Reads the training outputs written by train_baseline.py and train_transformers.py:
    outputs/<model>/seed_<seed>/history.csv          (loss per epoch)
    outputs/<model>/seed_<seed>/summary.json         (final scores)
    outputs/<model>/seed_<seed>/test_predictions.csv (test predictions)
    outputs/<model>/seed_<seed>/best_checkpoint/     (saved model weights)

The "best model" is picked on the VALIDATION set (highest mean val Macro-F1),
then its seed with the highest val Macro-F1. The test set is never used to choose.

Usage:
    python -m sentiment.visualize
    python -m sentiment.visualize --output-dir outputs --skip-confidence
"""

import argparse                          # reads options typed on the command line
import json                              # reads each run's summary.json
from pathlib import Path                 # handles folder and file paths safely

import matplotlib
matplotlib.use("Agg")                    # draw to files only, no pop-up window needed
import matplotlib.pyplot as plt          # the plotting library
import pandas as pd                      # reads CSV files as tables
import seaborn as sns                    # nicer heatmaps on top of matplotlib
from sklearn.metrics import confusion_matrix   # counts true-vs-predicted pairs

ROOT = Path(__file__).resolve().parent.parent   # the repository's top folder

# Same class order as LABEL2ID in transformer_models.py (kept here so the plots
# can be drawn on a computer without PyTorch installed).
LABEL_ORDER = ["negative", "neutral", "positive"]
# Display names and plotting order for the three models.
MODEL_NAMES = {
    "baseline": "Baseline: BiLSTM",
    "method_a": "Method A: XLM-RoBERTa",
    "method_b": "Method B: IndoBERT + class weights",
}


def find_runs(output_dir: Path) -> dict[str, list[Path]]:
    """Return {"method_a": [run folder per seed], ...} for every finished run."""
    runs: dict[str, list[Path]] = {}
    for summary in sorted(output_dir.glob("*/seed_*/summary.json")):   # one per finished run
        run_dir = summary.parent                                     # e.g. outputs/method_a/seed_42
        runs.setdefault(run_dir.parent.name, []).append(run_dir)    # group by model folder
    return runs


def plot_loss_curves(runs: dict[str, list[Path]], figures_dir: Path) -> Path:
    """Draw train and validation loss per epoch: one panel per model.

    The line is the mean over seeds; the shaded band spans the lowest to the
    highest seed. Each panel has its own y axis because Method B's loss is
    class-weighted, so its numbers are on a different scale.
    """
    models = [model for model in MODEL_NAMES if model in runs]       # keep the fixed order
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        # stack every seed's history.csv into one table, with a "seed" column
        history = pd.concat(
            pd.read_csv(run / "history.csv").assign(seed=run.name) for run in runs[model]
        )
        for column, label, color in (
            ("train_loss", "Train loss", "#1f77b4"),       # blue line
            ("val_loss", "Validation loss", "#d62728"),    # red line
        ):
            per_epoch = history.groupby("epoch")[column]   # all seeds' values for each epoch
            mean = per_epoch.mean()                        # the line
            ax.plot(mean.index, mean.values, marker="o", color=color, label=label)
            # shaded band from the lowest to the highest seed at each epoch
            ax.fill_between(mean.index, per_epoch.min(), per_epoch.max(), color=color, alpha=0.2)
        ax.set_title(f"{MODEL_NAMES[model]}\n({len(runs[model])} seeds)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_xticks(sorted(history["epoch"].unique()))   # whole epochs only: 1, 2, 3, ...
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = figures_dir / "loss_curves.png"
    fig.savefig(path, dpi=200)                     # 200 dpi is sharp enough for the paper
    plt.close(fig)
    return path


def pick_best_run(runs: dict[str, list[Path]]) -> tuple[str, Path]:
    """Best model = highest mean val Macro-F1; then its best seed on val Macro-F1."""
    def val_f1(run: Path) -> float:
        summary = json.loads((run / "summary.json").read_text())
        return summary["metrics"]["val"]["macro_f1"]

    # average val Macro-F1 over seeds, for each model
    model_scores = {model: sum(map(val_f1, dirs)) / len(dirs) for model, dirs in runs.items()}
    best_model = max(model_scores, key=model_scores.get)      # model with the top average
    best_run = max(runs[best_model], key=val_f1)              # its seed with the top score
    return best_model, best_run


def plot_confusion_matrix(model: str, run: Path, figures_dir: Path) -> Path:
    """Heatmap of true class (rows) vs predicted class (columns) on the test set."""
    predictions = pd.read_csv(run / "test_predictions.csv", keep_default_na=False)
    counts = confusion_matrix(predictions["label"], predictions["prediction"], labels=LABEL_ORDER)
    # each row as a percentage of that true class, so classes of different sizes compare fairly
    row_percent = counts / counts.sum(axis=1, keepdims=True) * 100
    # each cell shows the count and the row percentage, e.g. "190\n91.3%"
    cell_text = [
        [f"{counts[i, j]}\n{row_percent[i, j]:.1f}%" for j in range(len(LABEL_ORDER))]
        for i in range(len(LABEL_ORDER))
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        row_percent, annot=cell_text, fmt="", cmap="Blues", vmin=0, vmax=100,
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
        cbar_kws={"label": "% of true class"}, ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{MODEL_NAMES.get(model, model)}, {run.name}\nTest set (n={counts.sum()})")
    fig.tight_layout()
    path = figures_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def predict_probabilities(model: str, run: Path, texts: list[str], device_name: str):
    """Reload the best checkpoint and return softmax probabilities, one row per text.

    PyTorch is imported here, not at the top, so the plots above still work on
    a computer without it.
    """
    import torch                                          # deep learning library
    from torch.nn import functional as F                  # softmax lives here
    from .transformer_models import tokenize_batch        # same tokenization as training

    if device_name == "auto":   # use the GPU when there is one
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = run / "best_checkpoint"

    if model == "baseline":
        from .baseline_model import load_baseline         # the BiLSTM class
        from .baseline_tokenizer import WordVocabulary    # word -> number lookup
        from .train_baseline import MAX_LENGTH            # same text length as training
        saved = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))
        vocab = WordVocabulary(saved["pad_token"], saved["unk_token"], saved["min_freq"])
        vocab.w2i = saved["w2i"]                          # restore the training vocabulary
        net, _ = load_baseline(vocab_size=len(vocab))
        net.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device))

        def encode(batch):
            ids = [vocab.encode(text, max_length=MAX_LENGTH) for text in batch]
            return {"input_ids": torch.tensor(ids, dtype=torch.long)}
        def logits_of(inputs):
            return net(inputs["input_ids"])
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)    # saved next to the weights
        net = AutoModelForSequenceClassification.from_pretrained(checkpoint)

        def encode(batch):
            return tokenize_batch(batch, tokenizer)
        def logits_of(inputs):
            return net(**inputs).logits

    net.to(device).eval()                                  # eval = no dropout, fixed outputs
    probabilities = []
    with torch.no_grad():                                  # no training, so no gradients
        for start in range(0, len(texts), 16):             # 16 texts at a time
            inputs = encode(texts[start:start + 16])
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            probabilities.extend(F.softmax(logits_of(inputs), dim=1).cpu().tolist())
    return probabilities


def misclassified_examples(model: str, run: Path, figures_dir: Path, device_name: str) -> Path:
    """Save every wrong test prediction with its confidence, most confident first."""
    predictions = pd.read_csv(run / "test_predictions.csv", keep_default_na=False)
    probs = predict_probabilities(model, run, predictions["text"].tolist(), device_name)
    for index, label in enumerate(LABEL_ORDER):            # one column per class probability
        predictions[f"p_{label}"] = [row[index] for row in probs]
    recomputed = [LABEL_ORDER[row.index(max(row))] for row in probs]
    # safety check: the reloaded model must agree with the saved predictions
    agreement = (pd.Series(recomputed) == predictions["prediction"]).mean()
    if agreement < 0.99:
        raise RuntimeError(f"Reloaded checkpoint matches only {agreement:.1%} of saved predictions")

    # confidence = the probability the model gave to the class it predicted
    predictions["confidence"] = [max(row) for row in probs]
    wrong = predictions[predictions["label"] != predictions["prediction"]]
    wrong = wrong.sort_values("confidence", ascending=False)       # most confidently wrong first
    wrong = wrong.rename(columns={"label": "ground_truth", "prediction": "predicted"})
    path = figures_dir / "test_misclassified.csv"
    wrong.to_csv(path, index=False)

    print(f"\n{len(wrong)} of {len(predictions)} test sentences are misclassified "
          f"(saved to {path}).")
    print("Most confident mistake for each kind of error (true -> predicted):")
    for (truth, guess), group in wrong.groupby(["ground_truth", "predicted"]):
        top = group.iloc[0]                                # already sorted by confidence
        print(f"\n  [{truth} -> {guess}] confidence {top['confidence']:.3f} "
              f"({len(group)} errors of this kind)\n  {top['text']}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs",
                        help="Folder that holds <model>/seed_<seed>/ run folders")
    parser.add_argument("--figures-dir", type=Path, default=None,
                        help="Where to save figures (default: <output-dir>/figures)")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--skip-confidence", action="store_true",
                        help="Only draw the plots; don't reload the checkpoint")
    args = parser.parse_args()

    runs = find_runs(args.output_dir)
    if not runs:
        parser.exit(1, f"No finished runs found under {args.output_dir}\n")
    figures_dir = args.figures_dir or args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Saved", plot_loss_curves(runs, figures_dir))
    model, run = pick_best_run(runs)
    print(f"Best model on validation: {MODEL_NAMES.get(model, model)}, {run.name}")
    print("Saved", plot_confusion_matrix(model, run, figures_dir))
    if not args.skip_confidence:
        misclassified_examples(model, run, figures_dir, args.device)


if __name__ == "__main__":
    main()
