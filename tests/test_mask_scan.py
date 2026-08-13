"""Tests for mask_scan method and MaskScanOutput class."""

import pytest
import torch

from ablms.outputs import MaskScanOutput


class TestMaskScanOutput:
    """Test MaskScanOutput dataclass."""

    def test_basic_properties(self):
        """Test basic properties of MaskScanOutput."""
        seq_len = 20
        vocab_size = 100
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[0] = False  # CLS
        mask[-1] = False  # SEP

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        assert output.seq_len == seq_len
        assert output.vocab_size == vocab_size

    def test_probabilities(self):
        """Test probability computation."""
        seq_len = 10
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        probs = output.probabilities
        assert probs.shape == logits.shape
        # Probabilities should sum to 1 along vocab dimension
        assert torch.allclose(probs.sum(dim=-1), torch.ones(seq_len), atol=1e-5)

    def test_log_probabilities(self):
        """Test log probability computation."""
        seq_len = 10
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        log_probs = output.log_probabilities
        assert log_probs.shape == logits.shape
        # Log probabilities should be <= 0
        assert (log_probs <= 0).all()

    def test_predictions(self):
        """Test prediction computation."""
        seq_len = 10
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        predictions = output.predictions
        assert predictions.shape == torch.Size([seq_len])
        assert predictions.dtype == torch.int64

    def test_accuracy_computation(self):
        """Test accuracy is computed correctly."""
        seq_len = 10
        vocab_size = 50
        # Create logits where position 2, 5, 7 will predict the original token
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Make logits predict the original token at positions 2, 5, 7
        for pos in [2, 5, 7]:
            logits[pos, original_ids[pos]] = 10.0

        # Make logits predict wrong token at other positions
        for pos in [0, 1, 3, 4, 6, 8, 9]:
            wrong_token = (original_ids[pos].item() + 1) % vocab_size
            logits[pos, wrong_token] = 10.0

        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        accuracy = output.accuracy()
        assert accuracy.shape == torch.Size([seq_len])
        # Check specific positions
        assert accuracy[2].item() == 1.0
        assert accuracy[5].item() == 1.0
        assert accuracy[7].item() == 1.0
        assert accuracy[0].item() == 0.0
        assert accuracy[1].item() == 0.0

    def test_accuracy_with_invalid_positions(self):
        """Test accuracy excludes invalid positions."""
        seq_len = 10
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[0] = False
        mask[-1] = False

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        accuracy = output.accuracy()
        # Invalid positions should have accuracy 0
        assert accuracy[0].item() == 0.0
        assert accuracy[-1].item() == 0.0

    def test_accuracy_aggregation(self):
        """Test accuracy with different aggregation methods."""
        seq_len = 10
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Make all positions predict correctly
        for pos in range(seq_len):
            logits[pos, original_ids[pos]] = 10.0

        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[0] = False
        mask[-1] = False

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        # 8 valid positions, all correct
        assert output.accuracy(agg="mean") == 1.0
        assert output.accuracy(agg="sum") == 8.0
        assert output.accuracy(agg="min") == 1.0
        assert output.accuracy(agg="max") == 1.0
        assert output.accuracy(agg="median") == 1.0

    def test_perplexity_computation(self):
        """Test perplexity is computed correctly."""
        seq_len = 5
        vocab_size = 10
        # Create deterministic logits
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4])

        # Set high logit for original token (low perplexity)
        for pos in range(seq_len):
            logits[pos, original_ids[pos]] = 10.0

        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        perplexity = output.perplexity()
        assert perplexity.shape == torch.Size([seq_len])
        # High confidence predictions should have low perplexity (close to 1)
        assert (perplexity < 2.0).all()

    def test_perplexity_aggregation(self):
        """Test perplexity with aggregation."""
        seq_len = 5
        vocab_size = 10
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4])

        for pos in range(seq_len):
            logits[pos, original_ids[pos]] = 10.0

        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        mean_ppl = output.perplexity(agg="mean")
        assert isinstance(mean_ppl, float)
        assert mean_ppl > 0

        # Test other aggregations
        sum_ppl = output.perplexity(agg="sum")
        assert isinstance(sum_ppl, float)
        assert sum_ppl > mean_ppl  # Sum should be larger than mean

    def test_entropy_computation(self):
        """Test entropy is computed correctly."""
        seq_len = 5
        vocab_size = 10

        # Uniform distribution has high entropy
        uniform_logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4])
        mask = torch.ones(seq_len, dtype=torch.bool)

        uniform_output = MaskScanOutput(
            logits=uniform_logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        # Peaked distribution has low entropy
        peaked_logits = torch.zeros(seq_len, vocab_size)
        for pos in range(seq_len):
            peaked_logits[pos, original_ids[pos]] = 100.0

        peaked_output = MaskScanOutput(
            logits=peaked_logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        # Test per-position entropy
        uniform_entropy = uniform_output.entropy()
        assert uniform_entropy.shape == torch.Size([seq_len])

        peaked_entropy = peaked_output.entropy()
        assert peaked_entropy.shape == torch.Size([seq_len])

        # Uniform distribution should have higher entropy
        assert uniform_output.entropy(agg="mean") > peaked_output.entropy(agg="mean")

    def test_entropy_aggregation(self):
        """Test entropy with aggregation."""
        seq_len = 5
        vocab_size = 10
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        mean_ent = output.entropy(agg="mean")
        assert isinstance(mean_ent, float)
        assert mean_ent >= 0

        # Test other aggregations
        min_ent = output.entropy(agg="min")
        max_ent = output.entropy(agg="max")
        assert min_ent <= mean_ent <= max_ent

    def test_chain_specific_accuracy(self):
        """Test chain-specific accuracy extraction."""
        seq_len = 20
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
            token_offsets=token_offsets,
        )

        heavy_acc = output.get_chain_accuracy("heavy")
        assert heavy_acc is not None
        assert heavy_acc.shape == torch.Size([9])

        light_acc = output.get_chain_accuracy("light")
        assert light_acc is not None
        assert light_acc.shape == torch.Size([8])

    def test_chain_specific_perplexity(self):
        """Test chain-specific perplexity extraction."""
        seq_len = 20
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
            token_offsets=token_offsets,
        )

        heavy_ppl = output.get_chain_perplexity("heavy")
        assert heavy_ppl is not None
        assert heavy_ppl.shape == torch.Size([9])

        light_ppl = output.get_chain_perplexity("light")
        assert light_ppl is not None
        assert light_ppl.shape == torch.Size([8])

    def test_chain_specific_entropy(self):
        """Test chain-specific entropy extraction."""
        seq_len = 20
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
            token_offsets=token_offsets,
        )

        heavy_ent = output.get_chain_entropy("heavy")
        assert heavy_ent is not None
        assert heavy_ent.shape == torch.Size([9])

        light_ent = output.get_chain_entropy("light")
        assert light_ent is not None
        assert light_ent.shape == torch.Size([8])

    def test_chain_metrics_missing_chain(self):
        """Test chain metrics return None for missing chain."""
        seq_len = 20
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10)}  # Only heavy chain

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
            token_offsets=token_offsets,
        )

        assert output.get_chain_accuracy("light") is None
        assert output.get_chain_perplexity("light") is None
        assert output.get_chain_entropy("light") is None

    def test_top_k_predictions(self):
        """Test top-k predictions."""
        seq_len = 10
        vocab_size = 100
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        values, indices = output.top_k_predictions(k=5)
        assert values.shape == torch.Size([seq_len, 5])
        assert indices.shape == torch.Size([seq_len, 5])

    def test_predicted_tokens_with_vocab(self):
        """Test predicted tokens with vocab."""
        seq_len = 5
        vocab_size = 10
        logits = torch.zeros(seq_len, vocab_size)
        # Set predictions to specific tokens
        for pos in range(seq_len):
            logits[pos, pos] = 10.0

        original_ids = torch.zeros(seq_len, dtype=torch.long)
        mask = torch.ones(seq_len, dtype=torch.bool)
        vocab = {f"token_{i}": i for i in range(vocab_size)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
            vocab=vocab,
        )

        predicted = output.predicted_tokens
        assert predicted is not None
        assert len(predicted) == seq_len
        assert predicted[0] == "token_0"
        assert predicted[4] == "token_4"

    def test_predicted_tokens_without_vocab(self):
        """Test predicted tokens returns None without vocab."""
        seq_len = 5
        vocab_size = 10
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
            vocab=None,
        )

        assert output.predicted_tokens is None

    def test_to_device(self):
        """Test moving to device."""
        seq_len = 10
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        cpu_output = output.to(torch.device("cpu"))
        assert cpu_output.logits.device.type == "cpu"
        assert cpu_output.original_token_ids.device.type == "cpu"
        assert cpu_output.attention_mask.device.type == "cpu"

    def test_cpu_method(self):
        """Test cpu() method."""
        seq_len = 10
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        cpu_output = output.cpu()
        assert cpu_output.logits.device.type == "cpu"

    def test_repr(self):
        """Test string representation."""
        seq_len = 10
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.zeros(seq_len, dtype=torch.long)

        # Make all predictions correct
        for pos in range(seq_len):
            logits[pos, 0] = 10.0

        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[0] = False
        mask[-1] = False

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        repr_str = repr(output)
        assert "MaskScanOutput" in repr_str
        assert "seq_len=10" in repr_str
        assert "valid_positions=8" in repr_str

    def test_empty_mask(self):
        """Test behavior with no valid positions."""
        seq_len = 5
        vocab_size = 10
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.zeros(seq_len, dtype=torch.bool)  # All invalid

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        assert output.accuracy(agg="mean") == 0.0
        assert output.perplexity(agg="mean") == 0.0
        assert output.entropy(agg="mean") == 0.0

    def test_callable_aggregation(self):
        """Test custom callable aggregation."""
        seq_len = 10
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Make half positions predict correctly
        for pos in range(0, seq_len, 2):
            logits[pos, original_ids[pos]] = 10.0
        for pos in range(1, seq_len, 2):
            wrong_token = (original_ids[pos].item() + 1) % vocab_size
            logits[pos, wrong_token] = 10.0

        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        # Test with lambda
        std_acc = output.accuracy(agg=lambda x: x.std())
        assert isinstance(std_acc, float)
        assert std_acc > 0  # std of [1,0,1,0,...] should be non-zero

        # Test with torch function
        var_acc = output.accuracy(agg=torch.var)
        assert isinstance(var_acc, float)

    def test_invalid_aggregation_string(self):
        """Test that invalid aggregation string raises error."""
        seq_len = 5
        vocab_size = 10
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
        )

        with pytest.raises(ValueError, match="Unknown aggregation"):
            output.accuracy(agg="invalid")

    def test_chain_aggregation(self):
        """Test chain-specific methods with aggregation."""
        seq_len = 20
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.arange(seq_len) % vocab_size

        # Make all positions predict correctly
        for pos in range(seq_len):
            logits[pos, original_ids[pos]] = 10.0

        mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=mask,
            token_offsets=token_offsets,
        )

        # Test chain accuracy with aggregation
        heavy_acc_tensor = output.get_chain_accuracy("heavy")
        assert heavy_acc_tensor is not None
        assert heavy_acc_tensor.shape == torch.Size([9])

        heavy_acc_mean = output.get_chain_accuracy("heavy", agg="mean")
        assert isinstance(heavy_acc_mean, float)
        assert heavy_acc_mean == 1.0

        # Test chain perplexity with aggregation
        light_ppl_mean = output.get_chain_perplexity("light", agg="mean")
        assert isinstance(light_ppl_mean, float)
        assert light_ppl_mean > 0

        # Test chain entropy with aggregation
        heavy_ent_max = output.get_chain_entropy("heavy", agg="max")
        assert isinstance(heavy_ent_max, float)


class TestBuildMask:
    """Tests for build_mask() method."""

    def test_build_mask_heavy_only(self):
        """Test mask built with only heavy chain (light chain all True)."""
        seq_len = 25
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 12), "light": (13, 24)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        # Mask heavy chain positions 3-6 only
        heavy_mask = torch.zeros(11, dtype=torch.bool)
        heavy_mask[3:7] = True

        full_mask = output.build_mask(heavy=heavy_mask)

        assert full_mask.shape == torch.Size([seq_len])
        # Special token positions (0, 24) should be True
        assert full_mask[0].item() is True
        assert full_mask[24].item() is True
        # Heavy chain: positions 1-11, mask applies
        assert full_mask[1].item() is False  # heavy[0]
        assert full_mask[4].item() is True  # heavy[3]
        assert full_mask[7].item() is True  # heavy[6]
        assert full_mask[8].item() is False  # heavy[7]
        # Light chain should all be True (not masked)
        for i in range(13, 24):
            assert full_mask[i].item() is True

    def test_build_mask_light_only(self):
        """Test mask built with only light chain (heavy chain all True)."""
        seq_len = 25
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 12), "light": (13, 24)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        # Mask light chain positions 0-2 only
        light_mask = torch.zeros(11, dtype=torch.bool)
        light_mask[0:3] = True

        full_mask = output.build_mask(light=light_mask)

        assert full_mask.shape == torch.Size([seq_len])
        # Heavy chain should all be True (not masked)
        for i in range(1, 12):
            assert full_mask[i].item() is True
        # Light chain: positions 13-23
        assert full_mask[13].item() is True  # light[0]
        assert full_mask[15].item() is True  # light[2]
        assert full_mask[16].item() is False  # light[3]

    def test_build_mask_both_chains(self):
        """Test mask built with both chains."""
        seq_len = 25
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 12), "light": (13, 24)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        heavy_mask = torch.tensor([True] * 5 + [False] * 6)  # First 5 positions
        light_mask = torch.tensor([False] * 5 + [True] * 6)  # Last 6 positions

        full_mask = output.build_mask(heavy=heavy_mask, light=light_mask)

        # Heavy: positions 1-5 True, 6-11 False
        for i in range(1, 6):
            assert full_mask[i].item() is True
        for i in range(6, 12):
            assert full_mask[i].item() is False
        # Light: positions 13-17 False, 18-23 True
        for i in range(13, 18):
            assert full_mask[i].item() is False
        for i in range(18, 24):
            assert full_mask[i].item() is True

    def test_build_mask_no_token_offsets(self):
        """Test error raised when token_offsets is None."""
        seq_len = 20
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        attn_mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=None,
        )

        with pytest.raises(ValueError, match="token_offsets is required"):
            output.build_mask(heavy=torch.ones(10, dtype=torch.bool))

    def test_build_mask_length_mismatch(self):
        """Test error for wrong mask length."""
        seq_len = 25
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 12), "light": (13, 24)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        # Wrong length for heavy (should be 11, not 5)
        with pytest.raises(ValueError, match="heavy mask length 5 doesn't match"):
            output.build_mask(heavy=torch.ones(5, dtype=torch.bool))

        # Wrong length for light (should be 11, not 8)
        with pytest.raises(ValueError, match="light mask length 8 doesn't match"):
            output.build_mask(light=torch.ones(8, dtype=torch.bool))

    def test_build_mask_missing_chain(self):
        """Test error when mask provided for absent chain."""
        seq_len = 15
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 14)}  # Only heavy chain

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        with pytest.raises(ValueError, match="light mask provided but no light chain"):
            output.build_mask(light=torch.ones(10, dtype=torch.bool))


class TestMetricsWithMask:
    """Tests for metric methods with mask parameter."""

    def test_accuracy_with_mask(self):
        """Test accuracy computation with user mask."""
        seq_len = 10
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Make positions 2, 5, 7 predict correctly
        for pos in [2, 5, 7]:
            logits[pos, original_ids[pos]] = 10.0
        # Make other positions predict wrong
        for pos in [0, 1, 3, 4, 6, 8, 9]:
            wrong_token = (original_ids[pos].item() + 1) % vocab_size
            logits[pos, wrong_token] = 10.0

        attn_mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
        )

        # Without mask: 3/10 correct = 0.3
        assert output.accuracy(agg="mean") == pytest.approx(0.3)

        # With mask including only positions 2, 5, 7 (all correct): 3/3 = 1.0
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[[2, 5, 7]] = True
        assert output.accuracy(mask=mask, agg="mean") == 1.0

        # With mask including only positions 0, 1 (all wrong): 0/2 = 0.0
        mask2 = torch.zeros(seq_len, dtype=torch.bool)
        mask2[[0, 1]] = True
        assert output.accuracy(mask=mask2, agg="mean") == 0.0

    def test_perplexity_with_mask(self):
        """Test perplexity computation with user mask."""
        seq_len = 5
        vocab_size = 10
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.tensor([0, 1, 2, 3, 4])

        # High confidence for first 3 positions
        for pos in range(3):
            logits[pos, original_ids[pos]] = 10.0
        # Low confidence for last 2 positions (uniform-ish)
        for pos in range(3, 5):
            logits[pos, :] = 1.0

        attn_mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
        )

        # Mask for only first 3 positions (low perplexity)
        mask_low = torch.tensor([True, True, True, False, False])
        ppl_low = output.perplexity(mask=mask_low, agg="mean")

        # Mask for only last 2 positions (high perplexity)
        mask_high = torch.tensor([False, False, False, True, True])
        ppl_high = output.perplexity(mask=mask_high, agg="mean")

        assert ppl_low < ppl_high

    def test_entropy_with_mask(self):
        """Test entropy computation with user mask."""
        seq_len = 5
        vocab_size = 10

        # First 3 positions: peaked distribution (low entropy)
        # Last 2 positions: uniform distribution (high entropy)
        logits = torch.zeros(seq_len, vocab_size)
        for pos in range(3):
            logits[pos, pos] = 100.0
        for pos in range(3, 5):
            logits[pos, :] = 0.0  # Uniform

        original_ids = torch.zeros(seq_len, dtype=torch.long)
        attn_mask = torch.ones(seq_len, dtype=torch.bool)

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
        )

        # Mask for only first 3 positions (low entropy)
        mask_low = torch.tensor([True, True, True, False, False])
        ent_low = output.entropy(mask=mask_low, agg="mean")

        # Mask for only last 2 positions (high entropy)
        mask_high = torch.tensor([False, False, False, True, True])
        ent_high = output.entropy(mask=mask_high, agg="mean")

        assert ent_low < ent_high

    def test_mask_combined_with_attention_mask(self):
        """Test that special tokens are still excluded when using mask."""
        seq_len = 10
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.arange(seq_len) % vocab_size

        # Make all positions predict correctly
        for pos in range(seq_len):
            logits[pos, original_ids[pos]] = 10.0

        # Attention mask excludes first and last positions (special tokens)
        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        attn_mask[0] = False
        attn_mask[-1] = False

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
        )

        # User mask includes all positions
        user_mask = torch.ones(seq_len, dtype=torch.bool)

        # Even with user_mask=all True, special tokens should still be excluded
        # 8 valid positions, all correct = 1.0 mean
        assert output.accuracy(mask=user_mask, agg="mean") == 1.0
        assert output.accuracy(mask=user_mask, agg="sum") == 8.0

        # User mask also includes special tokens, but they're still excluded
        user_mask_with_special = torch.ones(seq_len, dtype=torch.bool)
        result = output.accuracy(mask=user_mask_with_special, agg="sum")
        assert result == 8.0  # Not 10.0


class TestChainMethodsWithMask:
    """Tests for chain-specific methods with mask parameter."""

    def test_get_chain_accuracy_with_mask(self):
        """Test chain-specific accuracy with mask."""
        seq_len = 20
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.arange(seq_len) % vocab_size

        # Make all positions predict correctly
        for pos in range(seq_len):
            logits[pos, original_ids[pos]] = 10.0

        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        # Without mask: all 9 heavy positions correct
        assert output.get_chain_accuracy("heavy", agg="sum") == 9.0

        # With mask: only first 3 positions
        mask = torch.tensor([True, True, True] + [False] * 6)
        assert output.get_chain_accuracy("heavy", mask=mask, agg="sum") == 3.0

    def test_get_chain_perplexity_with_mask(self):
        """Test chain-specific perplexity with mask."""
        seq_len = 20
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.arange(seq_len) % vocab_size

        # Heavy chain: first 5 positions high confidence, last 4 low confidence
        for pos in range(1, 6):
            logits[pos, original_ids[pos]] = 10.0
        for pos in range(6, 10):
            logits[pos, :] = 1.0

        # Light chain: all high confidence
        for pos in range(11, 19):
            logits[pos, original_ids[pos]] = 10.0

        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        # Mask for first 5 heavy positions (low perplexity)
        mask_low = torch.tensor([True] * 5 + [False] * 4)
        ppl_low = output.get_chain_perplexity("heavy", mask=mask_low, agg="mean")

        # Mask for last 4 heavy positions (high perplexity)
        mask_high = torch.tensor([False] * 5 + [True] * 4)
        ppl_high = output.get_chain_perplexity("heavy", mask=mask_high, agg="mean")

        assert ppl_low < ppl_high

    def test_get_chain_entropy_with_mask(self):
        """Test chain-specific entropy with mask."""
        seq_len = 20
        vocab_size = 50
        logits = torch.zeros(seq_len, vocab_size)
        original_ids = torch.arange(seq_len) % vocab_size

        # Light chain: first 4 positions peaked, last 4 uniform
        for pos in range(11, 15):
            logits[pos, original_ids[pos]] = 100.0
        for pos in range(15, 19):
            logits[pos, :] = 0.0

        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        # Mask for first 4 light positions (low entropy)
        mask_low = torch.tensor([True] * 4 + [False] * 4)
        ent_low = output.get_chain_entropy("light", mask=mask_low, agg="mean")

        # Mask for last 4 light positions (high entropy)
        mask_high = torch.tensor([False] * 4 + [True] * 4)
        ent_high = output.get_chain_entropy("light", mask=mask_high, agg="mean")

        assert ent_low < ent_high

    def test_chain_method_mask_length_mismatch(self):
        """Test error for wrong mask length in chain methods."""
        seq_len = 20
        vocab_size = 50
        logits = torch.randn(seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (seq_len,))
        attn_mask = torch.ones(seq_len, dtype=torch.bool)
        token_offsets = {"heavy": (1, 10), "light": (11, 19)}

        output = MaskScanOutput(
            logits=logits,
            original_token_ids=original_ids,
            attention_mask=attn_mask,
            token_offsets=token_offsets,
        )

        # Heavy chain length is 9, mask length is 5
        wrong_mask = torch.ones(5, dtype=torch.bool)

        with pytest.raises(
            ValueError, match="mask length 5 doesn't match chain length 9"
        ):
            output.get_chain_accuracy("heavy", mask=wrong_mask)

        with pytest.raises(
            ValueError, match="mask length 5 doesn't match chain length 9"
        ):
            output.get_chain_perplexity("heavy", mask=wrong_mask)

        with pytest.raises(
            ValueError, match="mask length 5 doesn't match chain length 9"
        ):
            output.get_chain_entropy("heavy", mask=wrong_mask)
