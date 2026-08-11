"""Tests for ESM-2 family encoder models."""

import pytest
import torch

from ablms import AntibodySequence, PairedSequenceError, list_models, load_model
from ablms.core.config import MODEL_REGISTRY, get_model_config
from ablms.encoders.esm2 import ESM2, ESM2_CONFIGS


class TestESM2Creation:
    """Test ESM-2 model instantiation and attributes."""

    def test_model_name(self):
        """Test model name attribute."""
        assert ESM2.model_name == "esm2"

    def test_supports_paired(self):
        """Test that ESM-2 does NOT support paired sequences."""
        assert ESM2.supports_paired is False

    def test_max_length(self):
        """Test max sequence length."""
        assert ESM2.max_length == 1024

    def test_mask_token(self):
        """Test mask token."""
        assert ESM2.mask_token == "<mask>"

    def test_separator_is_none(self):
        """Test that separator is None for unpaired model."""
        assert ESM2.separator is None

    def test_has_mlm_head(self):
        """Test that ESM-2 has MLM head."""
        assert ESM2.has_mlm_head is True

    @pytest.mark.slow
    def test_model_initialization_default(self):
        """Test model can be initialized with default (650M) variant."""
        model = ESM2(devices="cpu")
        assert model is not None
        assert model._model is not None
        assert model._tokenizer is not None
        assert model._model_id == "facebook/esm2_t33_650M_UR50D"
        assert model.embedding_dim == 1280

    @pytest.mark.slow
    def test_model_initialization_8m(self):
        """Test model can be initialized with 8M variant."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        assert model is not None
        assert model._model_id == "facebook/esm2_t6_8M_UR50D"
        assert model.embedding_dim == 320

    def test_esm2_configs(self):
        """Test that all ESM-2 configs are defined."""
        expected_models = [
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
            "facebook/esm2_t33_650M_UR50D",
            "facebook/esm2_t36_3B_UR50D",
            "facebook/esm2_t48_15B_UR50D",
        ]
        for model_id in expected_models:
            assert model_id in ESM2_CONFIGS
            assert "embedding_dim" in ESM2_CONFIGS[model_id]
            assert "num_layers" in ESM2_CONFIGS[model_id]


class TestESM2PairedRejection:
    """Test that ESM-2 correctly rejects paired sequences."""

    @pytest.mark.slow
    def test_paired_sequence_raises_error(self):
        """Test that paired sequences raise PairedSequenceError."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG", light="DIQMTQSPS")

        with pytest.raises(PairedSequenceError):
            model.get_embeddings([seq])

    @pytest.mark.slow
    def test_single_heavy_chain_accepted(self):
        """Test that single heavy chain sequences are accepted."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_embeddings([seq])
        assert output.embeddings.shape[0] == 1

    @pytest.mark.slow
    def test_single_light_chain_accepted(self):
        """Test that single light chain sequences are accepted."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(light="DIQMTQSPS")
        output = model.get_embeddings([seq])
        assert output.embeddings.shape[0] == 1


class TestESM2Formatting:
    """Test sequence formatting for ESM-2."""

    @pytest.mark.slow
    def test_format_single_heavy(self):
        """Test formatting a single heavy chain."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        formatted = model._format_for_model([seq])
        assert len(formatted) == 1
        assert formatted[0] == "EVQLVESGG"

    @pytest.mark.slow
    def test_format_single_light(self):
        """Test formatting a single light chain."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(light="DIQMTQSPS")
        formatted = model._format_for_model([seq])
        assert len(formatted) == 1
        assert formatted[0] == "DIQMTQSPS"

    @pytest.mark.slow
    def test_format_mask_token_conversion(self):
        """Test that <MASK> is converted to <mask>."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        formatted = model._format_for_model([seq])
        assert "<mask>" in formatted[0]
        assert "<MASK>" not in formatted[0]

    @pytest.mark.slow
    def test_format_no_spaces_between_residues(self):
        """Test that ESM model does not add spaces between residues."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        formatted = model._format_for_model([seq])
        # No spaces should be added for ESM models
        assert " " not in formatted[0]


class TestESM2Tokenization:
    """Test ESM-2 tokenization."""

    @pytest.mark.slow
    def test_tokenize_returns_tensor_dict(self):
        """Test that tokenization returns expected tensor dictionary."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        formatted = ["EVQLVESGG"]
        tokenized = model._tokenize(formatted)
        assert "input_ids" in tokenized
        assert "attention_mask" in tokenized
        assert isinstance(tokenized["input_ids"], torch.Tensor)
        assert isinstance(tokenized["attention_mask"], torch.Tensor)

    @pytest.mark.slow
    def test_tokenize_batch(self):
        """Test tokenization of multiple sequences."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        formatted = ["EVQLVESGG", "QVQLVQSGA"]
        tokenized = model._tokenize(formatted)
        assert tokenized["input_ids"].shape[0] == 2


class TestESM2TokenOffsets:
    """Test token offset computation for ESM-2."""

    @pytest.mark.slow
    def test_offsets_single_heavy(self):
        """Test offsets for single heavy chain."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        formatted = model._format_for_model([seq])
        tokenized = model._tokenize(formatted)
        offsets = model._compute_token_offsets([seq], tokenized)

        assert len(offsets) == 1
        assert "heavy" in offsets[0]
        start, end = offsets[0]["heavy"]
        assert start == 1  # After [CLS]
        assert end - start == len("EVQLVESGG")

    @pytest.mark.slow
    def test_offsets_single_light(self):
        """Test offsets for single light chain."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(light="DIQMTQSPS")
        formatted = model._format_for_model([seq])
        tokenized = model._tokenize(formatted)
        offsets = model._compute_token_offsets([seq], tokenized)

        assert len(offsets) == 1
        assert "light" in offsets[0]
        start, end = offsets[0]["light"]
        assert start == 1  # After [CLS]
        assert end - start == len("DIQMTQSPS")


class TestESM2Embeddings:
    """Test ESM-2 embedding extraction."""

    @pytest.mark.slow
    def test_embeddings_shape_single_8m(self):
        """Test embedding shape for 8M model."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_embeddings([seq])

        assert output.embeddings.ndim == 3
        assert output.embeddings.shape[0] == 1  # batch size
        assert output.embeddings.shape[2] == 320  # 8M embedding dim

    @pytest.mark.slow
    def test_embeddings_batch(self):
        """Test embeddings for batch of sequences."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        sequences = [
            AntibodySequence(heavy="EVQLVESGG"),
            AntibodySequence(heavy="QVQLVQSGA"),
        ]
        output = model.get_embeddings(sequences)

        assert output.embeddings.shape[0] == 2

    @pytest.mark.slow
    def test_chain_embeddings_extraction(self):
        """Test extracting chain-specific embeddings."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_embeddings([seq])

        heavy_emb = output.get_chain_embeddings(0, "heavy")
        assert heavy_emb is not None
        assert heavy_emb.shape[0] == len("EVQLVESGG")


class TestESM2Attention:
    """Test ESM-2 attention extraction."""

    @pytest.mark.slow
    def test_attention_output(self):
        """Test attention weight extraction."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_attention([seq])

        assert output.attention_weights.ndim == 5  # [batch, layers, heads, seq, seq]
        assert output.attention_weights.shape[0] == 1
        # 8M model has 6 layers
        assert output.attention_weights.shape[1] == 6


class TestESM2Logits:
    """Test ESM-2 MLM logits."""

    @pytest.mark.slow
    def test_logits_output(self):
        """Test MLM logits extraction."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_logits([seq])

        assert output.logits.ndim == 3  # [batch, seq_len, vocab_size]
        assert output.logits.shape[0] == 1

    @pytest.mark.slow
    def test_logits_with_mask(self):
        """Test logits for masked sequence."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        output = model.get_logits([seq])

        assert output.logits.ndim == 3


class TestESM2MaskFilling:
    """Test ESM-2 mask filling."""

    @pytest.mark.slow
    def test_fill_single_mask(self):
        """Test filling a single mask position."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        results = model.fill_mask([seq], top_k=3)

        assert len(results) == 1
        assert len(results[0]) <= 3
        for pred in results[0]:
            assert isinstance(pred, AntibodySequence)
            assert "<MASK>" not in pred.heavy_chain
            assert "<mask>" not in pred.heavy_chain

    @pytest.mark.slow
    def test_fill_mask_no_mask(self):
        """Test fill_mask with no mask token returns original."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        results = model.fill_mask([seq], top_k=3)

        assert len(results) == 1
        assert len(results[0]) == 1


class TestESM2PseudoLogLikelihood:
    """Test ESM-2 pseudo log-likelihood computation."""

    @pytest.mark.slow
    def test_pseudo_ll_returns_float(self):
        """Test that pseudo log-likelihood returns a float."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        scores = model.pseudo_log_likelihood([seq])

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    @pytest.mark.slow
    def test_pseudo_ll_batch(self):
        """Test pseudo log-likelihood for batch."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        sequences = [
            AntibodySequence(heavy="EVQLVESGG"),
            AntibodySequence(heavy="QVQLVQSGA"),
        ]
        scores = model.pseudo_log_likelihood(sequences)

        assert len(scores) == 2
        for score in scores:
            assert isinstance(score, float)


class TestESM2MaskScan:
    """Test ESM-2 mask scanning."""

    @pytest.mark.slow
    def test_mask_scan_output(self):
        """Test mask scan returns expected output."""
        model = ESM2(devices="cpu", model_id="facebook/esm2_t6_8M_UR50D")
        seq = AntibodySequence(heavy="EVQLVESGG")
        results = model.mask_scan([seq])

        assert len(results) == 1
        result = results[0]
        assert result.logits is not None
        assert result.original_token_ids is not None
        assert result.vocab is not None


class TestESM2Registration:
    """Test ESM-2 model registration."""

    def test_esm2_8m_in_registry(self):
        """Test that ESM2-8M is registered."""
        assert "esm2-8m" in MODEL_REGISTRY

    def test_esm2_35m_in_registry(self):
        """Test that ESM2-35M is registered."""
        assert "esm2-35m" in MODEL_REGISTRY

    def test_esm2_150m_in_registry(self):
        """Test that ESM2-150M is registered."""
        assert "esm2-150m" in MODEL_REGISTRY

    def test_esm2_650m_in_registry(self):
        """Test that ESM2-650M is registered."""
        assert "esm2-650m" in MODEL_REGISTRY

    def test_esm2_3b_in_registry(self):
        """Test that ESM2-3B is registered."""
        assert "esm2-3b" in MODEL_REGISTRY

    def test_esm2_15b_in_registry(self):
        """Test that ESM2-15B is registered."""
        assert "esm2-15b" in MODEL_REGISTRY

    def test_esm2_8m_config(self):
        """Test ESM2-8M configuration."""
        config = get_model_config("esm2-8m")
        assert config.name == "esm2-8m"
        assert config.model_class == ESM2
        assert config.model_id == "facebook/esm2_t6_8M_UR50D"
        assert config.supports_paired is False
        assert config.max_length == 1024
        assert config.embedding_dim == 320
        assert config.mask_token == "<mask>"
        assert config.separator is None
        assert config.has_mlm_head is True
        assert config.model_type == "encoder"
        assert config.extra_kwargs == {"model_id": "facebook/esm2_t6_8M_UR50D"}

    def test_esm2_650m_config(self):
        """Test ESM2-650M configuration."""
        config = get_model_config("esm2-650m")
        assert config.name == "esm2-650m"
        assert config.model_id == "facebook/esm2_t33_650M_UR50D"
        assert config.embedding_dim == 1280

    def test_esm2_3b_config(self):
        """Test ESM2-3B configuration."""
        config = get_model_config("esm2-3b")
        assert config.name == "esm2-3b"
        assert config.model_id == "facebook/esm2_t36_3B_UR50D"
        assert config.embedding_dim == 2560

    def test_esm2_15b_config(self):
        """Test ESM2-15B configuration."""
        config = get_model_config("esm2-15b")
        assert config.name == "esm2-15b"
        assert config.model_id == "facebook/esm2_t48_15B_UR50D"
        assert config.embedding_dim == 5120

    def test_esm2_in_list_models(self):
        """Test that ESM-2 variants appear in list_models."""
        models = list_models()
        assert "esm2-8m" in models
        assert "esm2-35m" in models
        assert "esm2-150m" in models
        assert "esm2-650m" in models
        assert "esm2-3b" in models
        assert "esm2-15b" in models
        for name in [
            "esm2-8m",
            "esm2-35m",
            "esm2-150m",
            "esm2-650m",
            "esm2-3b",
            "esm2-15b",
        ]:
            assert models[name] == "encoder"

    @pytest.mark.slow
    def test_load_model_esm2_8m(self):
        """Test loading ESM2-8M via load_model."""
        model = load_model("esm2-8m", devices="cpu")
        assert isinstance(model, ESM2)
        assert model._model_id == "facebook/esm2_t6_8M_UR50D"
        assert model.embedding_dim == 320
