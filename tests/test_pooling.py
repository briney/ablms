"""Tests for pooling strategies."""

import pytest
import torch

from ablms.utils.pooling import (
    apply_pooling,
    cls_pooling,
    last_pooling,
    max_pooling,
    mean_pooling,
)


class TestMeanPooling:
    """Test mean pooling function."""

    def test_mean_pooling_no_mask(self):
        """Test mean pooling without attention mask."""
        embeddings = torch.ones(2, 10, 768)
        pooled = mean_pooling(embeddings)

        assert pooled.shape == torch.Size([2, 768])
        assert torch.allclose(pooled, torch.ones(2, 768))

    def test_mean_pooling_with_mask(self):
        """Test mean pooling with attention mask."""
        embeddings = torch.ones(2, 10, 768)
        # Mask out last 5 tokens
        attention_mask = torch.ones(2, 10)
        attention_mask[:, 5:] = 0

        pooled = mean_pooling(embeddings, attention_mask)

        assert pooled.shape == torch.Size([2, 768])
        # Should still be ones since all valid positions are ones
        assert torch.allclose(pooled, torch.ones(2, 768))


class TestMaxPooling:
    """Test max pooling function."""

    def test_max_pooling_no_mask(self):
        """Test max pooling without attention mask."""
        embeddings = torch.arange(20).float().view(2, 10, 1).expand(2, 10, 768)
        pooled = max_pooling(embeddings)

        assert pooled.shape == torch.Size([2, 768])
        # Max should be 9 for first batch, 19 for second
        assert pooled[0, 0].item() == 9.0

    def test_max_pooling_with_mask(self):
        """Test max pooling with attention mask."""
        embeddings = torch.arange(10).float().view(1, 10, 1).expand(1, 10, 4)
        attention_mask = torch.ones(1, 10)
        attention_mask[:, 5:] = 0  # Mask out positions 5-9

        pooled = max_pooling(embeddings, attention_mask)

        assert pooled.shape == torch.Size([1, 4])
        # Max should be 4 (from position 4, since 5-9 are masked)
        assert pooled[0, 0].item() == 4.0


class TestClsPooling:
    """Test CLS token pooling function."""

    def test_cls_pooling(self):
        """Test CLS pooling extracts first token."""
        embeddings = torch.randn(2, 10, 768)
        pooled = cls_pooling(embeddings)

        assert pooled.shape == torch.Size([2, 768])
        assert torch.allclose(pooled, embeddings[:, 0, :])


class TestLastPooling:
    """Test last token pooling function."""

    def test_last_pooling_no_mask(self):
        """Test last pooling without attention mask."""
        embeddings = torch.randn(2, 10, 768)
        pooled = last_pooling(embeddings)

        assert pooled.shape == torch.Size([2, 768])
        assert torch.allclose(pooled, embeddings[:, -1, :])

    def test_last_pooling_with_mask(self):
        """Test last pooling with attention mask."""
        embeddings = torch.arange(10).float().view(1, 10, 1).expand(1, 10, 4)
        attention_mask = torch.ones(1, 10)
        attention_mask[:, 5:] = 0  # Valid positions are 0-4

        pooled = last_pooling(embeddings, attention_mask)

        assert pooled.shape == torch.Size([1, 4])
        # Last valid position is 4
        assert pooled[0, 0].item() == 4.0


class TestApplyPooling:
    """Test apply_pooling function."""

    def test_apply_mean(self):
        """Test applying mean pooling."""
        embeddings = torch.ones(2, 10, 768)
        pooled = apply_pooling(embeddings, "mean")
        assert pooled.shape == torch.Size([2, 768])

    def test_apply_max(self):
        """Test applying max pooling."""
        embeddings = torch.ones(2, 10, 768)
        pooled = apply_pooling(embeddings, "max")
        assert pooled.shape == torch.Size([2, 768])

    def test_apply_cls(self):
        """Test applying CLS pooling."""
        embeddings = torch.randn(2, 10, 768)
        pooled = apply_pooling(embeddings, "cls")
        assert pooled.shape == torch.Size([2, 768])

    def test_apply_first(self):
        """Test applying first (alias for CLS) pooling."""
        embeddings = torch.randn(2, 10, 768)
        pooled = apply_pooling(embeddings, "first")
        assert pooled.shape == torch.Size([2, 768])

    def test_apply_last(self):
        """Test applying last pooling."""
        embeddings = torch.randn(2, 10, 768)
        pooled = apply_pooling(embeddings, "last")
        assert pooled.shape == torch.Size([2, 768])

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        embeddings = torch.randn(2, 10, 768)
        with pytest.raises(ValueError):
            apply_pooling(embeddings, "invalid")

    def test_case_insensitive(self):
        """Test that strategy names are case insensitive."""
        embeddings = torch.ones(2, 10, 768)
        pooled = apply_pooling(embeddings, "MEAN")
        assert pooled.shape == torch.Size([2, 768])


class TestPoolingIsPaddingInvariant:
    """Pooling a batch alone must equal pooling it inside a longer padded stack.

    This is what makes it safe to pool per batch rather than after all batches
    have been concatenated and padded to a global maximum length.
    """

    @pytest.mark.parametrize("strategy", ["mean", "max", "cls", "first", "last"])
    def test_extra_right_padding_does_not_change_result(self, strategy):
        torch.manual_seed(0)
        batch, real_len, hidden, extra = 3, 7, 16, 11

        embeddings = torch.randn(batch, real_len, hidden)
        mask = torch.ones(batch, real_len)
        mask[1, 5:] = 0  # a shorter sequence in the batch

        padded = torch.cat([embeddings, torch.zeros(batch, extra, hidden)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(batch, extra)], dim=1)

        tight = apply_pooling(embeddings, strategy, mask)
        loose = apply_pooling(padded, strategy, padded_mask)

        assert torch.allclose(tight, loose, atol=1e-6)
