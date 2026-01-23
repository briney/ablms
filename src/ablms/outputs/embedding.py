"""Embedding output dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ablms.core.sequence import AntibodySequence


@dataclass
class EmbeddingOutput:
    """
    Output container for embedding results.

    Attributes:
        embeddings: Token-level embeddings with shape [batch, seq_len, hidden_dim]
            or sequence-level embeddings with shape [batch, hidden_dim].
        attention_mask: Boolean mask indicating valid (non-padding) positions.
            Shape: [batch, seq_len].
        token_offsets: List of dictionaries mapping chain names to (start, end)
            positions for each sequence in the batch.
        pooled: Optional pooled sequence-level embeddings. Shape: [batch, hidden_dim].
        sequences: Original input sequences.
        layer: The layer index from which embeddings were extracted.
    """

    embeddings: torch.Tensor
    attention_mask: torch.Tensor | None = None
    token_offsets: list[dict[str, tuple[int, int]]] | None = None
    pooled: torch.Tensor | None = None
    sequences: list[AntibodySequence] | None = field(default=None, repr=False)
    layer: int = -1

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the embeddings tensor."""
        return self.embeddings.shape

    @property
    def hidden_dim(self) -> int:
        """Return the hidden dimension size."""
        return self.embeddings.shape[-1]

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self.embeddings.shape[0]

    @property
    def is_pooled(self) -> bool:
        """Check if embeddings are pooled (sequence-level)."""
        return len(self.embeddings.shape) == 2

    def get_chain_embeddings(
        self, idx: int, chain: str
    ) -> torch.Tensor | None:
        """
        Extract embeddings for a specific chain from a batch element.

        Args:
            idx: Batch index.
            chain: Chain name ("heavy" or "light").

        Returns:
            Embeddings for the specified chain with shape [chain_len, hidden_dim],
            or None if the chain is not present or offsets are unavailable.
        """
        if self.is_pooled:
            raise ValueError(
                "Cannot extract chain embeddings from pooled output. "
                "Use get_embeddings() instead of get_sequence_embeddings()."
            )

        if self.token_offsets is None:
            return None

        if idx >= len(self.token_offsets):
            raise IndexError(f"Batch index {idx} out of range")

        offsets = self.token_offsets[idx]
        if chain not in offsets:
            return None

        start, end = offsets[chain]
        return self.embeddings[idx, start:end, :]

    def to(self, device: torch.device) -> EmbeddingOutput:
        """
        Move embeddings to a specific device.

        Args:
            device: Target device.

        Returns:
            New EmbeddingOutput with tensors on the specified device.
        """
        return EmbeddingOutput(
            embeddings=self.embeddings.to(device),
            attention_mask=(
                self.attention_mask.to(device)
                if self.attention_mask is not None
                else None
            ),
            token_offsets=self.token_offsets,
            pooled=self.pooled.to(device) if self.pooled is not None else None,
            sequences=self.sequences,
            layer=self.layer,
        )

    def cpu(self) -> EmbeddingOutput:
        """Move embeddings to CPU."""
        return self.to(torch.device("cpu"))

    def numpy(self) -> "EmbeddingOutput":
        """
        Convert embeddings to numpy arrays.

        Returns:
            New EmbeddingOutput with numpy arrays instead of tensors.
        """
        import numpy as np

        return EmbeddingOutput(
            embeddings=self.embeddings.cpu().numpy(),
            attention_mask=(
                self.attention_mask.cpu().numpy()
                if self.attention_mask is not None
                else None
            ),
            token_offsets=self.token_offsets,
            pooled=self.pooled.cpu().numpy() if self.pooled is not None else None,
            sequences=self.sequences,
            layer=self.layer,
        )

    def get_sequence_tokens(self, idx: int) -> torch.Tensor:
        """
        Get unpadded token embeddings for a single sequence.

        Args:
            idx: Batch index.

        Returns:
            Tensor of shape [actual_seq_len, hidden_dim] without padding.
        """
        if self.is_pooled:
            raise ValueError("Cannot get token embeddings from pooled output.")
        if self.attention_mask is None:
            return self.embeddings[idx]
        mask = self.attention_mask[idx].bool()
        return self.embeddings[idx][mask]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Index access returns unpadded embeddings for sequence idx."""
        return self.get_sequence_tokens(idx)

    def __iter__(self):
        """Iterate over unpadded embeddings for each sequence."""
        for i in range(self.batch_size):
            yield self.get_sequence_tokens(i)

    def __len__(self) -> int:
        """Return number of sequences."""
        return self.batch_size

    def __repr__(self) -> str:
        """Return a string representation."""
        shape_str = "x".join(str(d) for d in self.embeddings.shape)
        pooled_str = f", pooled={self.pooled.shape}" if self.pooled is not None else ""
        return f"EmbeddingOutput(shape=[{shape_str}], layer={self.layer}{pooled_str})"
