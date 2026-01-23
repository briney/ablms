"""Pooling strategies for sequence embeddings."""

from enum import Enum

import torch


class PoolingStrategy(Enum):
    """Available pooling strategies."""

    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    FIRST = "first"  # Alias for CLS
    LAST = "last"


def mean_pooling(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply mean pooling to token embeddings.

    Args:
        embeddings: Token embeddings with shape [batch, seq_len, hidden_dim].
        attention_mask: Boolean mask indicating valid positions.
            Shape: [batch, seq_len].

    Returns:
        Pooled embeddings with shape [batch, hidden_dim].
    """
    if attention_mask is None:
        return embeddings.mean(dim=1)

    # Expand mask to match embedding dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    # Sum embeddings for valid positions
    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)

    # Count valid positions
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


def max_pooling(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply max pooling to token embeddings.

    Args:
        embeddings: Token embeddings with shape [batch, seq_len, hidden_dim].
        attention_mask: Boolean mask indicating valid positions.

    Returns:
        Pooled embeddings with shape [batch, hidden_dim].
    """
    if attention_mask is None:
        return embeddings.max(dim=1).values

    # Set masked positions to very negative value
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    embeddings_masked = embeddings.masked_fill(~mask_expanded.bool(), float("-inf"))

    return embeddings_masked.max(dim=1).values


def cls_pooling(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Extract CLS token embedding (first token).

    Args:
        embeddings: Token embeddings with shape [batch, seq_len, hidden_dim].
        attention_mask: Not used, included for API consistency.

    Returns:
        CLS embeddings with shape [batch, hidden_dim].
    """
    return embeddings[:, 0, :]


def last_pooling(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Extract last valid token embedding.

    Args:
        embeddings: Token embeddings with shape [batch, seq_len, hidden_dim].
        attention_mask: Boolean mask indicating valid positions.

    Returns:
        Last token embeddings with shape [batch, hidden_dim].
    """
    if attention_mask is None:
        return embeddings[:, -1, :]

    # Find the last valid position for each sequence
    # Sum along seq_len to get lengths, subtract 1 for index
    lengths = attention_mask.sum(dim=1).long() - 1
    batch_size = embeddings.size(0)

    # Gather the last valid embedding for each sequence
    indices = lengths.unsqueeze(-1).expand(-1, embeddings.size(-1))
    return embeddings.gather(1, indices.unsqueeze(1).long()).squeeze(1)


def apply_pooling(
    embeddings: torch.Tensor,
    strategy: str,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a pooling strategy to embeddings.

    Args:
        embeddings: Token embeddings with shape [batch, seq_len, hidden_dim].
        strategy: Pooling strategy name ("mean", "max", "cls", "first", "last").
        attention_mask: Boolean mask indicating valid positions.

    Returns:
        Pooled embeddings with shape [batch, hidden_dim].

    Raises:
        ValueError: If an unknown pooling strategy is specified.
    """
    strategy = strategy.lower()

    if strategy == "mean":
        return mean_pooling(embeddings, attention_mask)
    elif strategy == "max":
        return max_pooling(embeddings, attention_mask)
    elif strategy in ("cls", "first"):
        return cls_pooling(embeddings, attention_mask)
    elif strategy == "last":
        return last_pooling(embeddings, attention_mask)
    else:
        raise ValueError(
            f"Unknown pooling strategy: {strategy}. "
            f"Choose from: mean, max, cls, first, last"
        )
