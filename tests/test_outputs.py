"""Tests for output dataclasses."""

import pytest
import torch

from ablms.outputs import (
    EmbeddingOutput,
    LogitsOutput,
    AttentionOutput,
    GenerationOutput,
)
from ablms.core.sequence import AntibodySequence, Species


class TestEmbeddingOutput:
    """Test EmbeddingOutput dataclass."""

    def test_basic_properties(self):
        """Test basic properties of EmbeddingOutput."""
        embeddings = torch.randn(2, 10, 768)
        output = EmbeddingOutput(embeddings=embeddings)

        assert output.shape == torch.Size([2, 10, 768])
        assert output.batch_size == 2
        assert output.hidden_dim == 768
        assert not output.is_pooled

    def test_pooled_embeddings(self):
        """Test pooled embeddings detection."""
        embeddings = torch.randn(2, 768)
        output = EmbeddingOutput(embeddings=embeddings)

        assert output.is_pooled
        assert output.batch_size == 2
        assert output.hidden_dim == 768

    def test_get_chain_embeddings(self):
        """Test extracting chain-specific embeddings."""
        embeddings = torch.randn(1, 20, 768)
        token_offsets = [{"heavy": (1, 10), "light": (11, 19)}]
        output = EmbeddingOutput(
            embeddings=embeddings,
            token_offsets=token_offsets,
        )

        heavy_emb = output.get_chain_embeddings(0, "heavy")
        assert heavy_emb.shape == torch.Size([9, 768])

        light_emb = output.get_chain_embeddings(0, "light")
        assert light_emb.shape == torch.Size([8, 768])

    def test_get_chain_embeddings_pooled_error(self):
        """Test that chain extraction raises error for pooled output."""
        embeddings = torch.randn(1, 768)
        output = EmbeddingOutput(embeddings=embeddings)

        with pytest.raises(ValueError):
            output.get_chain_embeddings(0, "heavy")

    def test_to_device(self):
        """Test moving to device."""
        embeddings = torch.randn(2, 10, 768)
        output = EmbeddingOutput(embeddings=embeddings)

        cpu_output = output.to(torch.device("cpu"))
        assert cpu_output.embeddings.device.type == "cpu"

    def test_single_layer_output_is_not_multi_layer(self):
        """Defaults are unchanged: no layers field means the old behaviour."""
        output = EmbeddingOutput(embeddings=torch.randn(2, 10, 768))

        assert not output.is_multi_layer
        assert output.num_layers == 1
        assert output.layers is None
        assert output.layer == -1

    def test_multi_layer_token_level_properties(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 10, 768), layer=None, layers=[0, 6, 12]
        )

        assert output.is_multi_layer
        assert output.num_layers == 3
        assert output.batch_size == 2
        assert output.hidden_dim == 768
        assert not output.is_pooled

    def test_multi_layer_pooled_is_detected(self):
        """[batch, layers, hidden] is pooled even though it has three dims."""
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768),
            pooled=torch.randn(2, 3, 768),
            layer=None,
            layers=[0, 6, 12],
        )

        assert output.is_pooled
        assert output.hidden_dim == 768

    def test_get_layer_returns_the_requested_layer(self):
        embeddings = torch.randn(2, 3, 768)
        output = EmbeddingOutput(embeddings=embeddings, layer=None, layers=[0, 6, 12])

        assert torch.equal(output.get_layer(6), embeddings[:, 1])
        assert torch.equal(output.get_layer(0), embeddings[:, 0])

    def test_get_layer_rejects_an_unselected_layer(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768), layer=None, layers=[0, 6, 12]
        )
        with pytest.raises(ValueError, match="not selected"):
            output.get_layer(7)

    def test_get_layer_rejects_single_layer_output(self):
        output = EmbeddingOutput(embeddings=torch.randn(2, 10, 768))
        with pytest.raises(ValueError, match="single layer"):
            output.get_layer(0)

    def test_concat_layers_on_pooled_output(self):
        embeddings = torch.randn(2, 3, 768)
        output = EmbeddingOutput(
            embeddings=embeddings, pooled=embeddings, layer=None, layers=[0, 6, 12]
        )

        concatenated = output.concat_layers()

        assert concatenated.shape == (2, 3 * 768)
        expected = torch.cat([embeddings[:, i] for i in range(3)], dim=-1)
        assert torch.equal(concatenated, expected)

    def test_concat_layers_on_token_level_output(self):
        embeddings = torch.randn(2, 3, 10, 768)
        output = EmbeddingOutput(embeddings=embeddings, layer=None, layers=[0, 6, 12])

        concatenated = output.concat_layers()

        assert concatenated.shape == (2, 10, 3 * 768)
        expected = torch.cat([embeddings[:, i] for i in range(3)], dim=-1)
        assert torch.equal(concatenated, expected)

    def test_concat_layers_rejects_single_layer_output(self):
        output = EmbeddingOutput(embeddings=torch.randn(2, 10, 768))
        with pytest.raises(ValueError, match="single layer"):
            output.concat_layers()

    def test_multi_layer_chain_embeddings_keep_the_layer_axis(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(1, 3, 20, 768),
            token_offsets=[{"heavy": (1, 10), "light": (11, 19)}],
            layer=None,
            layers=[0, 6, 12],
        )

        heavy = output.get_chain_embeddings(0, "heavy")

        assert heavy.shape == (3, 9, 768)

    def test_multi_layer_sequence_tokens_strip_padding(self):
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        output = EmbeddingOutput(
            embeddings=torch.randn(1, 3, 5, 768),
            attention_mask=mask,
            layer=None,
            layers=[0, 6, 12],
        )

        tokens = output.get_sequence_tokens(0)

        assert tokens.shape == (3, 3, 768)

    def test_layers_survive_a_device_move(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768), layer=None, layers=[0, 6, 12]
        )
        assert output.cpu().layers == [0, 6, 12]
        assert output.numpy().layers == [0, 6, 12]

    def test_repr_reports_the_selected_layers(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768), layer=None, layers=[0, 6, 12]
        )
        assert "layers=[0, 6, 12]" in repr(output)


class TestLogitsOutput:
    """Test LogitsOutput dataclass."""

    def test_basic_properties(self):
        """Test basic properties of LogitsOutput."""
        logits = torch.randn(2, 10, 1000)
        output = LogitsOutput(logits=logits)

        assert output.shape == torch.Size([2, 10, 1000])
        assert output.batch_size == 2
        assert output.vocab_size == 1000

    def test_probabilities(self):
        """Test probability computation."""
        logits = torch.randn(2, 10, 100)
        output = LogitsOutput(logits=logits)

        probs = output.probabilities
        assert probs.shape == logits.shape
        # Probabilities should sum to 1 along vocab dimension
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 10), atol=1e-5)

    def test_predictions(self):
        """Test prediction computation."""
        logits = torch.randn(2, 10, 100)
        output = LogitsOutput(logits=logits)

        predictions = output.predictions
        assert predictions.shape == torch.Size([2, 10])
        assert predictions.dtype == torch.int64

    def test_top_k_predictions(self):
        """Test top-k predictions."""
        logits = torch.randn(2, 10, 100)
        output = LogitsOutput(logits=logits)

        values, indices = output.top_k_predictions(k=5)
        assert values.shape == torch.Size([2, 10, 5])
        assert indices.shape == torch.Size([2, 10, 5])


class TestAttentionOutput:
    """Test AttentionOutput dataclass."""

    def test_basic_properties(self):
        """Test basic properties of AttentionOutput."""
        attention = torch.randn(2, 12, 8, 20, 20)
        output = AttentionOutput(attention_weights=attention)

        assert output.shape == torch.Size([2, 12, 8, 20, 20])
        assert output.batch_size == 2
        assert output.num_layers == 12
        assert output.num_heads == 8
        assert output.seq_len == 20

    def test_get_layer(self):
        """Test extracting specific layer attention."""
        attention = torch.randn(2, 12, 8, 20, 20)
        output = AttentionOutput(attention_weights=attention)

        layer_attn = output.get_layer(0)
        assert layer_attn.shape == torch.Size([2, 8, 20, 20])

        # Test negative indexing
        last_layer = output.get_layer(-1)
        assert last_layer.shape == torch.Size([2, 8, 20, 20])

    def test_get_head(self):
        """Test extracting specific head attention."""
        attention = torch.randn(2, 12, 8, 20, 20)
        output = AttentionOutput(attention_weights=attention)

        head_attn = output.get_head(0, 0)
        assert head_attn.shape == torch.Size([2, 20, 20])

    def test_get_mean_attention(self):
        """Test mean attention across layers and heads."""
        attention = torch.randn(2, 12, 8, 20, 20)
        output = AttentionOutput(attention_weights=attention)

        mean_attn = output.get_mean_attention()
        assert mean_attn.shape == torch.Size([2, 20, 20])


class TestGenerationOutput:
    """Test GenerationOutput dataclass."""

    def test_basic_properties(self):
        """Test basic properties of GenerationOutput."""
        sequences = [
            AntibodySequence(heavy="EVQLVESGGGLVQ"),
            AntibodySequence(heavy="QVQLVESGGGLVQ"),
        ]
        scores = [-10.5, -12.3]
        output = GenerationOutput(sequences=sequences, scores=scores)

        assert output.num_sequences == 2
        assert len(output) == 2

    def test_get_sequence(self):
        """Test getting specific sequence."""
        sequences = [
            AntibodySequence(heavy="EVQLVESGGGLVQ"),
            AntibodySequence(heavy="QVQLVESGGGLVQ"),
        ]
        output = GenerationOutput(sequences=sequences)

        seq = output.get_sequence(0)
        assert seq.heavy_chain == "EVQLVESGGGLVQ"

    def test_get_score(self):
        """Test getting specific score."""
        sequences = [
            AntibodySequence(heavy="EVQLVESGGGLVQ"),
        ]
        scores = [-10.5]
        output = GenerationOutput(sequences=sequences, scores=scores)

        assert output.get_score(0) == -10.5

    def test_get_top_k(self):
        """Test getting top-k sequences."""
        sequences = [
            AntibodySequence(heavy="EVQLVESGGGLVQ"),
            AntibodySequence(heavy="QVQLVESGGGLVQ"),
            AntibodySequence(heavy="DVQLVESGGGLVQ"),
        ]
        scores = [-10.5, -5.0, -15.0]  # Middle one is best (highest)
        output = GenerationOutput(sequences=sequences, scores=scores)

        top_2 = output.get_top_k(k=2)
        assert top_2.num_sequences == 2
        assert top_2.scores[0] == -5.0  # Best score first

    def test_filter_by_score(self):
        """Test filtering by score."""
        sequences = [
            AntibodySequence(heavy="EVQLVESGGGLVQ"),
            AntibodySequence(heavy="QVQLVESGGGLVQ"),
            AntibodySequence(heavy="DVQLVESGGGLVQ"),
        ]
        scores = [-10.5, -5.0, -15.0]
        output = GenerationOutput(sequences=sequences, scores=scores)

        filtered = output.filter_by_score(min_score=-11.0)
        assert filtered.num_sequences == 2

    def test_iteration(self):
        """Test iterating over sequences."""
        sequences = [
            AntibodySequence(heavy="EVQLVESGGGLVQ"),
            AntibodySequence(heavy="QVQLVESGGGLVQ"),
        ]
        output = GenerationOutput(sequences=sequences)

        for i, seq in enumerate(output):
            assert seq == sequences[i]
