"""Mask scan output dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ablms.core.sequence import AntibodySequence


@dataclass
class MaskScanOutput:
    """
    Output container for mask scan results for a single sequence.

    Contains per-position predictions when each position was masked,
    along with helper methods for computing accuracy, perplexity, and entropy.

    The `mask_scan()` method returns a list of `MaskScanOutput` objects, one per
    input sequence. Each `MaskScanOutput` contains the results for scanning all
    positions of that single sequence.

    Attributes:
        logits: Raw logits when each position was masked.
            Shape: ``[seq_len, vocab_size]`` where ``seq_len`` includes special tokens.
        original_token_ids: Original token indices at each position.
            Shape: ``[seq_len]``.
        attention_mask: Boolean mask indicating valid (scannable) positions.
            True for amino acid positions, False for special tokens (CLS, SEP, etc.).
            Shape: ``[seq_len]``.
        vocab: Dictionary mapping token strings to indices.
        sequence: Original input sequence (AntibodySequence object).
        token_offsets: Dictionary mapping chain names ("heavy", "light") to
            ``(start, end)`` token positions for extracting chain-specific metrics.
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

    def _aggregate(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        agg: str | Callable | None,
    ) -> torch.Tensor | float:
        """
        Apply aggregation to masked values.

        Args:
            values: Tensor of values to aggregate.
            mask: Boolean mask indicating valid positions.
            agg: Aggregation method. None returns raw values.
                String options: "mean", "sum", "min", "max", "median".
                Or pass a callable that takes a 1D tensor.

        Returns:
            Raw tensor if agg=None, else aggregated scalar.
        """
        if agg is None:
            return values

        # Get only valid positions
        valid_values = values[mask.bool()]
        if valid_values.numel() == 0:
            return 0.0

        if isinstance(agg, str):
            agg_funcs = {
                "mean": torch.mean,
                "sum": torch.sum,
                "min": torch.min,
                "max": torch.max,
                "median": torch.median,
            }
            if agg not in agg_funcs:
                raise ValueError(
                    f"Unknown aggregation: {agg}. Use one of {list(agg_funcs.keys())}"
                )
            result = agg_funcs[agg](valid_values)
        else:
            # Callable
            result = agg(valid_values)

        # Convert to float if scalar tensor
        if isinstance(result, torch.Tensor) and result.ndim == 0:
            return result.item()
        return result

    def accuracy(
        self, agg: str | Callable | None = None
    ) -> torch.Tensor | float:
        """
        Compute per-position accuracy.

        Args:
            agg: Aggregation method. None returns per-position tensor.
                String options: "mean", "sum", "min", "max", "median".
                Or pass a callable that takes a 1D tensor.

        Returns:
            Per-position tensor if agg=None (shape [seq_len] with 1.0 where
            prediction matches original, 0.0 otherwise), else aggregated scalar.
            Invalid positions (where attention_mask is False) are 0.0.
        """
        correct = (self.predictions == self.original_token_ids).float()
        values = correct * self.attention_mask.float()
        return self._aggregate(values, self.attention_mask, agg)

    def perplexity(
        self, agg: str | Callable | None = None
    ) -> torch.Tensor | float:
        """
        Compute per-position perplexity.

        Perplexity is exp(-log_prob[original_token]) for each position.

        Args:
            agg: Aggregation method. None returns per-position tensor.
                String options: "mean", "sum", "min", "max", "median".
                Or pass a callable that takes a 1D tensor.

        Returns:
            Per-position tensor if agg=None (shape [seq_len]), else aggregated scalar.
            Invalid positions have perplexity of 0.0.
        """
        log_probs = self.log_probabilities
        # Gather log probability of the original token at each position
        original_log_probs = log_probs.gather(
            dim=-1, index=self.original_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        # Perplexity = exp(-log_prob)
        ppl = torch.exp(-original_log_probs)
        values = ppl * self.attention_mask.float()
        return self._aggregate(values, self.attention_mask, agg)

    def entropy(
        self, agg: str | Callable | None = None
    ) -> torch.Tensor | float:
        """
        Compute per-position entropy.

        Entropy = -sum(p * log(p)) over vocab for each position.

        Args:
            agg: Aggregation method. None returns per-position tensor.
                String options: "mean", "sum", "min", "max", "median".
                Or pass a callable that takes a 1D tensor.

        Returns:
            Per-position tensor if agg=None (shape [seq_len]), else aggregated scalar.
            Invalid positions have entropy of 0.0.
        """
        probs = self.probabilities
        log_probs = self.log_probabilities
        # Entropy = -sum(p * log(p))
        ent = -torch.sum(probs * log_probs, dim=-1)
        values = ent * self.attention_mask.float()
        return self._aggregate(values, self.attention_mask, agg)

    def get_chain_accuracy(
        self, chain: str, agg: str | Callable | None = None
    ) -> torch.Tensor | float | None:
        """
        Get accuracy for a specific chain.

        Args:
            chain: Chain name ("heavy" or "light").
            agg: Aggregation method. None returns per-position tensor.
                String options: "mean", "sum", "min", "max", "median".
                Or pass a callable that takes a 1D tensor.

        Returns:
            Accuracy tensor or aggregated scalar for the specified chain,
            or None if chain not present.
        """
        if self.token_offsets is None or chain not in self.token_offsets:
            return None
        start, end = self.token_offsets[chain]
        chain_accuracy = self.accuracy()[start:end]
        chain_mask = self.attention_mask[start:end]
        return self._aggregate(chain_accuracy, chain_mask, agg)

    def get_chain_perplexity(
        self, chain: str, agg: str | Callable | None = None
    ) -> torch.Tensor | float | None:
        """
        Get perplexity for a specific chain.

        Args:
            chain: Chain name ("heavy" or "light").
            agg: Aggregation method. None returns per-position tensor.
                String options: "mean", "sum", "min", "max", "median".
                Or pass a callable that takes a 1D tensor.

        Returns:
            Perplexity tensor or aggregated scalar for the specified chain,
            or None if chain not present.
        """
        if self.token_offsets is None or chain not in self.token_offsets:
            return None
        start, end = self.token_offsets[chain]
        chain_perplexity = self.perplexity()[start:end]
        chain_mask = self.attention_mask[start:end]
        return self._aggregate(chain_perplexity, chain_mask, agg)

    def get_chain_entropy(
        self, chain: str, agg: str | Callable | None = None
    ) -> torch.Tensor | float | None:
        """
        Get entropy for a specific chain.

        Args:
            chain: Chain name ("heavy" or "light").
            agg: Aggregation method. None returns per-position tensor.
                String options: "mean", "sum", "min", "max", "median".
                Or pass a callable that takes a 1D tensor.

        Returns:
            Entropy tensor or aggregated scalar for the specified chain,
            or None if chain not present.
        """
        if self.token_offsets is None or chain not in self.token_offsets:
            return None
        start, end = self.token_offsets[chain]
        chain_entropy = self.entropy()[start:end]
        chain_mask = self.attention_mask[start:end]
        return self._aggregate(chain_entropy, chain_mask, agg)

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
            f"mean_accuracy={self.accuracy(agg='mean'):.3f})"
        )
