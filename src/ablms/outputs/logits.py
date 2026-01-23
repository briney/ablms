"""Logits output dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ablms.core.sequence import AntibodySequence


@dataclass
class LogitsOutput:
    """
    Output container for logits/probability results.

    Attributes:
        logits: Raw logits from the model. Shape: [batch, seq_len, vocab_size].
        attention_mask: Boolean mask indicating valid (non-padding) positions.
        token_offsets: List of dictionaries mapping chain names to (start, end)
            positions for each sequence in the batch.
        vocab: Dictionary mapping token strings to indices.
        sequences: Original input sequences.
    """

    logits: torch.Tensor
    attention_mask: torch.Tensor | None = None
    token_offsets: List[Dict[str, Tuple[int, int]]] | None = None
    vocab: Dict[str, int] | None = field(default=None, repr=False)
    sequences: List[AntibodySequence] | None = field(default=None, repr=False)

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the logits tensor."""
        return self.logits.shape

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self.logits.shape[0]

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.logits.shape[-1]

    @property
    def probabilities(self) -> torch.Tensor:
        """
        Compute softmax probabilities from logits.

        Returns:
            Probability tensor with shape [batch, seq_len, vocab_size].
        """
        return F.softmax(self.logits, dim=-1)

    @property
    def log_probabilities(self) -> torch.Tensor:
        """
        Compute log probabilities from logits.

        Returns:
            Log probability tensor with shape [batch, seq_len, vocab_size].
        """
        return F.log_softmax(self.logits, dim=-1)

    @property
    def predictions(self) -> torch.Tensor:
        """
        Get predicted token indices (argmax).

        Returns:
            Tensor of predicted indices with shape [batch, seq_len].
        """
        return self.logits.argmax(dim=-1)

    @property
    def predicted_tokens(self) -> List[List[str]] | None:
        """
        Get predicted tokens as strings.

        Returns:
            List of lists of predicted token strings, or None if vocab is unavailable.
        """
        if self.vocab is None:
            return None

        # Build reverse vocab mapping
        idx_to_token = {v: k for k, v in self.vocab.items()}
        predictions = self.predictions.cpu().tolist()

        result = []
        for batch_preds in predictions:
            tokens = [idx_to_token.get(idx, "<UNK>") for idx in batch_preds]
            result.append(tokens)

        return result

    def top_k_predictions(
        self, k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k predictions for each position.

        Args:
            k: Number of top predictions to return.

        Returns:
            Tuple of (values, indices) tensors, each with shape [batch, seq_len, k].
        """
        return torch.topk(self.logits, k=k, dim=-1)

    def get_chain_logits(
        self, idx: int, chain: str
    ) -> torch.Tensor | None:
        """
        Extract logits for a specific chain from a batch element.

        Args:
            idx: Batch index.
            chain: Chain name ("heavy" or "light").

        Returns:
            Logits for the specified chain with shape [chain_len, vocab_size],
            or None if the chain is not present.
        """
        if self.token_offsets is None:
            return None

        if idx >= len(self.token_offsets):
            raise IndexError(f"Batch index {idx} out of range")

        offsets = self.token_offsets[idx]
        if chain not in offsets:
            return None

        start, end = offsets[chain]
        return self.logits[idx, start:end, :]

    def to(self, device: torch.device) -> LogitsOutput:
        """Move logits to a specific device."""
        return LogitsOutput(
            logits=self.logits.to(device),
            attention_mask=(
                self.attention_mask.to(device)
                if self.attention_mask is not None
                else None
            ),
            token_offsets=self.token_offsets,
            vocab=self.vocab,
            sequences=self.sequences,
        )

    def cpu(self) -> LogitsOutput:
        """Move logits to CPU."""
        return self.to(torch.device("cpu"))

    def __repr__(self) -> str:
        """Return a string representation."""
        shape_str = "x".join(str(d) for d in self.logits.shape)
        return f"LogitsOutput(shape=[{shape_str}])"
