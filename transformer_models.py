"""Shared setup helpers for the pretrained sentiment analysis models."""

from typing import Optional

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


def create_loss(class_weights: Optional[Tensor] = None) -> nn.CrossEntropyLoss:
    """Create cross-entropy loss, optionally weighted in LABEL2ID order."""
    return nn.CrossEntropyLoss(weight=class_weights)


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
