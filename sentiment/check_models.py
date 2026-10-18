"""Check one training batch for Method A or B; no checkpoints are saved."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import set_seed

from .transformer_models import (
    LABEL2ID,
    NUM_LABELS,
    count_parameters,
    load_method_a,
    load_method_b,
    tokenize_batch,
)


def run_check(method: str, data_dir: Path):
    train_df = pd.read_csv(data_dir / "train_split.csv")
    if not {"text", "label"}.issubset(train_df.columns):
        raise ValueError("train_split.csv must contain 'text' and 'label' columns")
    if not train_df["label"].isin(LABEL2ID).all():
        raise ValueError(f"Training labels must be one of {list(LABEL2ID)}")

    # Select one text per class in the shared label order.
    texts = []
    for label in LABEL2ID:
        examples = train_df.loc[train_df["label"] == label, "text"]
        if examples.empty:
            raise ValueError(f"Training data is missing class: {label}")
        texts.append(examples.iloc[0])
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Selected training texts must be nonempty strings")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == "a":
        model, tokenizer, loss_fn = load_method_a()
    else:
        # Weights use the full training split, not just the selected batch.
        model, tokenizer, loss_fn = load_method_b(train_df["label"])
    model = model.to(device).train()
    loss_fn = loss_fn.to(device)
    batch = tokenize_batch(texts, tokenizer).to(device)
    labels = torch.tensor(list(LABEL2ID.values()), dtype=torch.long, device=device)

    counts = count_parameters(model)
    if counts["total"] != counts["trainable"]:
        raise RuntimeError("Every model parameter must have requires_grad=True")
    print(f"Method {method.upper()} | device: {device} | seed: 42")
    print(f"Training rows: {len(train_df)} | batch label order: {list(LABEL2ID)}")
    print(f"Parameters: total={counts['total']:,}, trainable={counts['trainable']:,}")
    for name, value in batch.items():
        print(f"{name}: {tuple(value.shape)}")
    print(f"labels: {tuple(labels.shape)}")
    if loss_fn.weight is not None:
        print(f"Class weights: {loss_fn.weight.detach().cpu().tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad()
    logits = model(**batch).logits
    if logits.shape != (len(texts), NUM_LABELS):
        raise RuntimeError(f"Unexpected class-score shape: {tuple(logits.shape)}")
    loss = loss_fn(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite")
    print(f"logits: {tuple(logits.shape)} | loss: {loss.item():.6f}")

    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Gradients are missing or not finite")

    # Both architectures expose their classification head as model.classifier.
    head_weights = {
        name: parameter for name, parameter in model.classifier.named_parameters()
        if name.endswith("weight")
    }
    if not any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in head_weights.values()
    ):
        raise RuntimeError("Classifier weights did not receive nonzero gradients")
    before = {name: parameter.detach().clone() for name, parameter in head_weights.items()}
    optimizer.step()
    if not any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in head_weights.items()
    ):
        raise RuntimeError("Classifier weights did not change after the optimizer step")
    print("PASS: finite loss and gradients; classifier weights changed after one AdamW update.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing the cleaned train_split.csv.",
    )
    args = parser.parse_args()
    try:
        run_check(args.method, args.data_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Check failed: {exc}\n")


if __name__ == "__main__":
    main()
