"""Check one training batch for the Baseline BiLSTM; no checkpoints are saved."""

import argparse
from pathlib import Path
import pandas as pd
import torch

from .baseline_model import load_baseline
from .baseline_tokenizer import WordVocabulary
from .transformer_models import LABEL2ID, NUM_LABELS, count_parameters


def run_check(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train_split.csv")
    if not {"text", "label"}.issubset(train_df.columns):
        raise ValueError("train_split.csv must contain 'text' and 'label' columns")

    # Build vocab on training set
    vocab = WordVocabulary(min_freq=2)
    vocab.build_vocab(train_df["text"])
    print(f"Vocabulary size: {len(vocab):,} tokens")

    # Grab one sample per class
    texts, labels_list = [], []
    for label, label_id in LABEL2ID.items():
        sample = train_df.loc[train_df["label"] == label, "text"].iloc[0]
        texts.append(sample)
        labels_list.append(label_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loss_fn = load_baseline(vocab_size=len(vocab))
    model = model.to(device).train()
    loss_fn = loss_fn.to(device)

    # Encode batch
    batch_ids = torch.tensor([vocab.encode(t, max_length=128) for t in texts], dtype=torch.long, device=device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)

    counts = count_parameters(model)
    print(f"Baseline BiLSTM | device: {device}")
    print(f"Parameters: total={counts['total']:,}, trainable={counts['trainable']:,}")
    print(f"input_ids shape: {tuple(batch_ids.shape)}, labels shape: {tuple(labels.shape)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    logits = model(batch_ids)
    if logits.shape != (len(texts), NUM_LABELS):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    loss = loss_fn(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite")
    print(f"logits shape: {tuple(logits.shape)} | loss: {loss.item():.6f}")

    loss.backward()
    fc_weight = model.fc.weight
    if fc_weight.grad is None or not torch.any(fc_weight.grad != 0):
        raise RuntimeError("FC weights did not receive non-zero gradients")

    before = fc_weight.detach().clone()
    optimizer.step()
    if torch.equal(before, fc_weight.detach()):
        raise RuntimeError("FC weights did not change after optimizer step")

    print("PASS: Baseline BiLSTM forward, backward, and parameter updates verified!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()
    run_check(args.data_dir)


if __name__ == "__main__":
    main()