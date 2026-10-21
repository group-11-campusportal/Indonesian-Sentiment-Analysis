"""Custom 2-layer Bidirectional LSTM baseline for sentiment analysis."""

import torch
import torch.nn as nn
from .transformer_models import NUM_LABELS, create_loss


class BiLSTMSentiment(nn.Module):
    """
    Tensor Dimensions at Each Step (as required by IEEE paper Section 4.A):
      1. Input Token IDs:     [Batch_Size, Seq_Len]
      2. Embedding Layer:     [Batch_Size, Seq_Len, Embed_Dim]
      3. 2-layer BiLSTM:      [Batch_Size, Seq_Len, 2 * Hidden_Dim]
      4. Max Pooling:         [Batch_Size, 2 * Hidden_Dim]
      5. Dropout:             [Batch_Size, 2 * Hidden_Dim]
      6. Fully Connected Head: [Batch_Size, NUM_LABELS]  (Logits for 3 classes)
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = NUM_LABELS,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        # Global max-pooling across sequence dimension
        pooled, _ = torch.max(lstm_out, dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits


def load_baseline(vocab_size: int):
    """Return the Baseline BiLSTM model and unweighted cross-entropy loss."""
    model = BiLSTMSentiment(vocab_size=vocab_size)
    return model, create_loss()