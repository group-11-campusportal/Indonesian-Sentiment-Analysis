"""Shared setup helpers for the pretrained sentiment analysis models."""

from collections import Counter
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Use the same class order for model outputs, labels, and future loss weights.
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {label_id: label for label, label_id in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def load_transformer(model_name: str):
    """Return a pretrained model and matching tokenizer with shared labels."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID.copy(),
        id2label=ID2LABEL.copy(),
    )
    return model, tokenizer


def tokenize_batch(texts: list[str], tokenizer: AutoTokenizer):
    """Tokenize original texts into PyTorch inputs for the matching model.

    Limit sequences to 256 tokens including special tokens, then pad only to
    the longest sequence in this batch.
    """
    return tokenizer(
        texts,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )


def create_loss(class_weights: Optional[Tensor] = None) -> nn.CrossEntropyLoss:
    """Create cross-entropy loss, optionally weighted in LABEL2ID order."""
    return nn.CrossEntropyLoss(weight=class_weights)


def load_method_a():
    """Return XLM-R, its tokenizer, and unweighted cross-entropy loss."""
    model, tokenizer = load_transformer("xlm-roberta-base")
    return model, tokenizer, create_loss()


def compute_class_weights(train_labels: Iterable[str]) -> Tensor:
    """Compute balanced weights from all training labels in LABEL2ID order."""
    counts = Counter(train_labels)
    unknown = set(counts) - LABEL2ID.keys()
    if unknown:
        raise ValueError(f"Unknown training labels: {sorted(unknown, key=repr)}")
    missing = [label for label in LABEL2ID if counts[label] == 0]
    if missing:
        raise ValueError(f"Training labels must include every class; missing: {missing}")

    total = sum(counts.values())
    return torch.tensor(
        [total / (NUM_LABELS * counts[label]) for label in LABEL2ID],
        dtype=torch.float32,
    )


def load_method_b(train_labels: Iterable[str]):
    """Return IndoBERT, its tokenizer, and training-class-weighted loss.

    Use training labels, never validation or test labels.
    """
    class_weights = compute_class_weights(train_labels)
    model, tokenizer = load_transformer("indobenchmark/indobert-base-p1")
    return model, tokenizer, create_loss(class_weights)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return all parameter elements and those eligible for training."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        # Frozen parameters still count toward the total, but not trainable.
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
