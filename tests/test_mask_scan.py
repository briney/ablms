"""Tests for mask_scan method and MaskScanOutput class."""

import pytest
import torch

from ablms.outputs import MaskScanOutput
from ablms.core.sequence import AntibodySequence


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
