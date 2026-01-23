"""Attention output dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ablms.core.sequence import AntibodySequence


@dataclass
class AttentionOutput:
    """
    Output container for attention weights.

    Attributes:
        attention_weights: Attention weights from all layers and heads.
            Shape: [batch, num_layers, num_heads, seq_len, seq_len].
        attention_mask: Boolean mask indicating valid (non-padding) positions.
        token_offsets: List of dictionaries mapping chain names to (start, end)
            positions for each sequence in the batch.
        sequences: Original input sequences.
    """

    attention_weights: torch.Tensor
    attention_mask: torch.Tensor | None = None
    token_offsets: list[dict[str, tuple[int, int]]] | None = None
    sequences: list[AntibodySequence] | None = field(default=None, repr=False)

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the attention weights tensor."""
        return self.attention_weights.shape

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self.attention_weights.shape[0]

    @property
    def num_layers(self) -> int:
        """Return the number of attention layers."""
        return self.attention_weights.shape[1]

    @property
    def num_heads(self) -> int:
        """Return the number of attention heads."""
        return self.attention_weights.shape[2]

    @property
    def seq_len(self) -> int:
        """Return the sequence length."""
        return self.attention_weights.shape[3]

    def get_layer(self, layer: int) -> torch.Tensor:
        """
        Get attention weights for a specific layer.

        Args:
            layer: Layer index (supports negative indexing).

        Returns:
            Attention weights with shape [batch, num_heads, seq_len, seq_len].
        """
        return self.attention_weights[:, layer, :, :, :]

    def get_head(self, layer: int, head: int) -> torch.Tensor:
        """
        Get attention weights for a specific layer and head.

        Args:
            layer: Layer index (supports negative indexing).
            head: Head index (supports negative indexing).

        Returns:
            Attention weights with shape [batch, seq_len, seq_len].
        """
        return self.attention_weights[:, layer, head, :, :]

    def get_mean_attention(self) -> torch.Tensor:
        """
        Get attention weights averaged across all layers and heads.

        Returns:
            Mean attention weights with shape [batch, seq_len, seq_len].
        """
        return self.attention_weights.mean(dim=(1, 2))

    def get_chain_attention(
        self, idx: int, query_chain: str, key_chain: str
    ) -> torch.Tensor | None:
        """
        Extract attention between specific chains.

        Args:
            idx: Batch index.
            query_chain: Query chain name ("heavy" or "light").
            key_chain: Key chain name ("heavy" or "light").

        Returns:
            Attention weights between the specified chains with shape
            [num_layers, num_heads, query_len, key_len], or None if
            chains are not present.
        """
        if self.token_offsets is None:
            return None

        if idx >= len(self.token_offsets):
            raise IndexError(f"Batch index {idx} out of range")

        offsets = self.token_offsets[idx]
        if query_chain not in offsets or key_chain not in offsets:
            return None

        q_start, q_end = offsets[query_chain]
        k_start, k_end = offsets[key_chain]

        return self.attention_weights[idx, :, :, q_start:q_end, k_start:k_end]

    def to(self, device: torch.device) -> AttentionOutput:
        """Move attention weights to a specific device."""
        return AttentionOutput(
            attention_weights=self.attention_weights.to(device),
            attention_mask=(
                self.attention_mask.to(device)
                if self.attention_mask is not None
                else None
            ),
            token_offsets=self.token_offsets,
            sequences=self.sequences,
        )

    def cpu(self) -> AttentionOutput:
        """Move attention weights to CPU."""
        return self.to(torch.device("cpu"))

    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"AttentionOutput("
            f"layers={self.num_layers}, "
            f"heads={self.num_heads}, "
            f"seq_len={self.seq_len})"
        )
