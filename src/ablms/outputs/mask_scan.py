"""Mask scan output dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ablms.core.sequence import AntibodySequence


@dataclass
class MaskScanOutput:
    """
    Output container for mask scan results.

    Contains per-position predictions when each position was masked,
    along with helper methods for computing accuracy, perplexity, and entropy.

    Attributes:
        logits: Raw logits when each position was masked. Shape: [seq_len, vocab_size].
        original_token_ids: Original token indices at each position. Shape: [seq_len].
        attention_mask: Boolean mask indicating valid positions (excludes special tokens).
            Shape: [seq_len].
        vocab: Dictionary mapping token strings to indices.
        sequence: Original input sequence.
        token_offsets: Dictionary mapping chain names to (start, end) positions.
    """

    logits: torch.Tensor
    original_token_ids: torch.Tensor
    attention_mask: torch.Tensor
    vocab: dict[str, int] | None = field(default=None, repr=False)
    sequence: AntibodySequence | None = field(default=None, repr=False)
    token_offsets: dict[str, tuple[int, int]] | None = None

    @property
    def seq_len(self) -> int:
        """Return the sequence length."""
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
            Probability tensor with shape [seq_len, vocab_size].
        """
        return F.softmax(self.logits, dim=-1)

    @property
    def log_probabilities(self) -> torch.Tensor:
        """
        Compute log probabilities from logits.

        Returns:
            Log probability tensor with shape [seq_len, vocab_size].
        """
        return F.log_softmax(self.logits, dim=-1)

    @property
    def predictions(self) -> torch.Tensor:
        """
        Get predicted token indices (argmax).

        Returns:
            Tensor of predicted indices with shape [seq_len].
        """
        return self.logits.argmax(dim=-1)

    @property
    def predicted_tokens(self) -> list[str] | None:
        """
        Get predicted tokens as strings.

        Returns:
            List of predicted token strings, or None if vocab is unavailable.
        """
        if self.vocab is None:
            return None

        # Build reverse vocab mapping
        idx_to_token = {v: k for k, v in self.vocab.items()}
        predictions = self.predictions.cpu().tolist()

        return [idx_to_token.get(idx, "<UNK>") for idx in predictions]

    @property
    def accuracy(self) -> torch.Tensor:
        """
        Compute per-position accuracy.

        Returns:
            Tensor of shape [seq_len] with 1.0 where prediction matches original,
            0.0 otherwise. Invalid positions (where attention_mask is False) are 0.0.
        """
        correct = (self.predictions == self.original_token_ids).float()
        return correct * self.attention_mask.float()

    @property
    def perplexity(self) -> torch.Tensor:
        """
        Compute per-position perplexity.

        Perplexity is exp(-log_prob[original_token]) for each position.

        Returns:
            Tensor of shape [seq_len] with perplexity values.
            Invalid positions have perplexity of 0.0.
        """
        log_probs = self.log_probabilities
        # Gather log probability of the original token at each position
        original_log_probs = log_probs.gather(
            dim=-1, index=self.original_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        # Perplexity = exp(-log_prob)
        ppl = torch.exp(-original_log_probs)
        return ppl * self.attention_mask.float()

    @property
    def entropy(self) -> torch.Tensor:
        """
        Compute per-position entropy.

        Entropy = -sum(p * log(p)) over vocab for each position.

        Returns:
            Tensor of shape [seq_len] with entropy values.
            Invalid positions have entropy of 0.0.
        """
        probs = self.probabilities
        log_probs = self.log_probabilities
        # Entropy = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        ent = -torch.sum(probs * log_probs, dim=-1)
        return ent * self.attention_mask.float()

    @property
    def mean_accuracy(self) -> float:
        """
        Compute mean accuracy over valid positions.

        Returns:
            Mean accuracy as a float.
        """
        valid_count = self.attention_mask.sum().item()
        if valid_count == 0:
            return 0.0
        return (self.accuracy.sum() / valid_count).item()

    @property
    def mean_perplexity(self) -> float:
        """
        Compute mean perplexity over valid positions.

        Returns:
            Mean perplexity as a float.
        """
        valid_count = self.attention_mask.sum().item()
        if valid_count == 0:
            return 0.0
        return (self.perplexity.sum() / valid_count).item()

    @property
    def mean_entropy(self) -> float:
        """
        Compute mean entropy over valid positions.

        Returns:
            Mean entropy as a float.
        """
        valid_count = self.attention_mask.sum().item()
        if valid_count == 0:
            return 0.0
        return (self.entropy.sum() / valid_count).item()

    def get_chain_accuracy(self, chain: str) -> torch.Tensor | None:
        """
        Get accuracy for a specific chain.

        Args:
            chain: Chain name ("heavy" or "light").

        Returns:
            Accuracy tensor for the specified chain, or None if chain not present.
        """
        if self.token_offsets is None or chain not in self.token_offsets:
            return None
        start, end = self.token_offsets[chain]
        return self.accuracy[start:end]

    def get_chain_perplexity(self, chain: str) -> torch.Tensor | None:
        """
        Get perplexity for a specific chain.

        Args:
            chain: Chain name ("heavy" or "light").

        Returns:
            Perplexity tensor for the specified chain, or None if chain not present.
        """
        if self.token_offsets is None or chain not in self.token_offsets:
            return None
        start, end = self.token_offsets[chain]
        return self.perplexity[start:end]

    def get_chain_entropy(self, chain: str) -> torch.Tensor | None:
        """
        Get entropy for a specific chain.

        Args:
            chain: Chain name ("heavy" or "light").

        Returns:
            Entropy tensor for the specified chain, or None if chain not present.
        """
        if self.token_offsets is None or chain not in self.token_offsets:
            return None
        start, end = self.token_offsets[chain]
        return self.entropy[start:end]

    def top_k_predictions(
        self, k: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k predictions for each position.

        Args:
            k: Number of top predictions to return.

        Returns:
            Tuple of (values, indices) tensors, each with shape [seq_len, k].
        """
        return torch.topk(self.logits, k=k, dim=-1)

    def to(self, device: torch.device) -> MaskScanOutput:
        """
        Move tensors to a specific device.

        Args:
            device: Target device.

        Returns:
            New MaskScanOutput with tensors on the specified device.
        """
        return MaskScanOutput(
            logits=self.logits.to(device),
            original_token_ids=self.original_token_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            vocab=self.vocab,
            sequence=self.sequence,
            token_offsets=self.token_offsets,
        )

    def cpu(self) -> MaskScanOutput:
        """Move tensors to CPU."""
        return self.to(torch.device("cpu"))

    def __repr__(self) -> str:
        """Return a string representation."""
        valid_count = self.attention_mask.sum().item()
        return (
            f"MaskScanOutput(seq_len={self.seq_len}, "
            f"valid_positions={int(valid_count)}, "
            f"mean_accuracy={self.mean_accuracy:.3f})"
        )
