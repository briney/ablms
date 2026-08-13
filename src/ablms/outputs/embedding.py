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
        pooled: Optional pooled sequence-level embeddings. Shape:
            [batch, hidden_dim], or [batch, layers, hidden_dim] when multiple
            layers were selected.
        sequences: Original input sequences.
        layer: The layer index from which embeddings were extracted, or None
            when multiple layers were selected (see `layers`).
        layers: Resolved indices of every selected layer when the output carries
            a layer axis, or None for a single-layer output.
    """

    embeddings: torch.Tensor
    attention_mask: torch.Tensor | None = None
    token_offsets: list[dict[str, tuple[int, int]]] | None = None
    pooled: torch.Tensor | None = None
    sequences: list[AntibodySequence] | None = field(default=None, repr=False)
    layer: int | None = -1
    layers: list[int] | None = None

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
    def is_multi_layer(self) -> bool:
        """Whether the embeddings carry a layer axis at dimension 1."""
        return self.layers is not None

    @property
    def num_layers(self) -> int:
        """Number of layers represented in this output."""
        return len(self.layers) if self.layers is not None else 1

    @property
    def is_pooled(self) -> bool:
        """Check if embeddings are pooled (sequence-level).

        Pooling removes the sequence axis, so a pooled output has one fewer
        dimension than a token-level one at the same layer arity: [batch,
        hidden] against [batch, seq, hidden], and [batch, layers, hidden]
        against [batch, layers, seq, hidden].
        """
        return self.embeddings.ndim == (3 if self.is_multi_layer else 2)

    def get_layer(self, layer: int) -> torch.Tensor:
        """
        Extract a single layer from a multi-layer output.

        Args:
            layer: The model's layer index, as reported in `layers` - not a
                position within the stack.

        Returns:
            The embeddings for that layer, with the layer axis removed:
            [batch, hidden_dim] if pooled, else [batch, seq_len, hidden_dim].

        Raises:
            ValueError: If this output holds a single layer, or if the
                requested layer was not among those selected.
        """
        if self.layers is None:
            raise ValueError(
                "This output holds a single layer; use .embeddings directly."
            )
        if layer not in self.layers:
            raise ValueError(
                f"Layer {layer} was not selected. Available layers: {self.layers}"
            )
        return self.embeddings[:, self.layers.index(layer)]

    def concat_layers(self) -> torch.Tensor:
        """
        Fold the layer axis into the hidden dimension.

        This is the form used for dimensionality reduction over every layer at
        once, where each sequence becomes one long feature vector.

        Returns:
            [batch, num_layers * hidden_dim] if pooled, else
            [batch, seq_len, num_layers * hidden_dim]. Layers appear in the
            order given by `layers`.

        Raises:
            ValueError: If this output holds a single layer.

        Note:
            Requires torch tensors: this uses `permute`/`flatten`, which
            `numpy.ndarray` does not support. Call this before `.numpy()`,
            not after - `get_layer()`, `get_chain_embeddings()`, and
            `get_sequence_tokens()` all still work post-`.numpy()` because
            they only slice, but `concat_layers()` will raise a plain
            `AttributeError` if called on a numpy-backed output.
        """
        if self.layers is None:
            raise ValueError(
                "This output holds a single layer; there is nothing to concatenate."
            )
        if self.is_pooled:
            # [batch, layers, hidden] -> [batch, layers * hidden]
            return self.embeddings.flatten(start_dim=1)
        # [batch, layers, seq, hidden] -> [batch, seq, layers * hidden]
        return self.embeddings.permute(0, 2, 1, 3).flatten(start_dim=2)

    def get_chain_embeddings(self, idx: int, chain: str) -> torch.Tensor | None:
        """
        Extract embeddings for a specific chain from a batch element.

        Args:
            idx: Batch index.
            chain: Chain name ("heavy" or "light").

        Returns:
            Embeddings for the specified chain with shape [chain_len, hidden_dim],
            or None if the chain is not present or offsets are unavailable.
            For multi-layer output the layer axis leads: [num_layers,
            chain_len, hidden_dim].
        """
        if self.is_pooled:
            raise ValueError(
                "Cannot extract chain embeddings from pooled output. "
                "Use get_embeddings() without pooling to get token-level embeddings."
            )

        if self.token_offsets is None:
            return None

        if idx >= len(self.token_offsets):
            raise IndexError(f"Batch index {idx} out of range")

        offsets = self.token_offsets[idx]
        if chain not in offsets:
            return None

        start, end = offsets[chain]
        if self.is_multi_layer:
            return self.embeddings[idx, :, start:end, :]
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
            layers=self.layers,
        )

    def cpu(self) -> EmbeddingOutput:
        """Move embeddings to CPU."""
        return self.to(torch.device("cpu"))

    def numpy(self) -> EmbeddingOutput:
        """
        Convert embeddings to numpy arrays.

        Returns:
            New EmbeddingOutput with numpy arrays instead of tensors.
        """

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
            layers=self.layers,
        )

    def get_sequence_tokens(self, idx: int) -> torch.Tensor:
        """
        Get unpadded token embeddings for a single sequence.

        Args:
            idx: Batch index.

        Returns:
            Tensor of shape [actual_seq_len, hidden_dim] without padding.
            For multi-layer output: [num_layers, actual_seq_len, hidden_dim].
        """
        if self.is_pooled:
            raise ValueError("Cannot get token embeddings from pooled output.")
        if self.attention_mask is None:
            return self.embeddings[idx]
        mask = self.attention_mask[idx].bool()
        if self.is_multi_layer:
            return self.embeddings[idx][:, mask]
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
        layer_str = (
            f", layers={self.layers}"
            if self.is_multi_layer
            else f", layer={self.layer}"
        )
        pooled_str = f", pooled={self.pooled.shape}" if self.pooled is not None else ""
        return f"EmbeddingOutput(shape=[{shape_str}]{layer_str}{pooled_str})"
