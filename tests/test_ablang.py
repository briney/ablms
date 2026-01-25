"""Tests for AbLang (v1) encoder model."""

import pytest
import torch

from ablms import AntibodySequence, load_model, list_models
from ablms.core.config import get_model_config, MODEL_REGISTRY
from ablms.encoders.ablang import AbLang
from ablms.exceptions import PairedSequenceError


class TestAbLangCreation:
    """Test AbLang model instantiation and attributes."""

    def test_model_name(self):
        """Test model name attribute."""
        assert AbLang.model_name == "ablang"

    def test_supports_paired(self):
        """Test that AbLang does NOT support paired sequences."""
        assert AbLang.supports_paired is False

    def test_max_length(self):
        """Test max sequence length."""
        assert AbLang.max_length == 160

    def test_embedding_dim(self):
        """Test embedding dimension."""
        assert AbLang.embedding_dim == 768

    def test_mask_token(self):
        """Test mask token."""
        assert AbLang.mask_token == "*"

    def test_separator(self):
        """Test chain separator is None."""
        assert AbLang.separator is None

    def test_has_mlm_head(self):
        """Test that AbLang has MLM head."""
        assert AbLang.has_mlm_head is True

    @pytest.mark.slow
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = AbLang(devices="cpu")
        assert model is not None
        # Models are lazy-loaded
        assert model._ablang_module is not None


class TestAbLangPairedRejection:
    """Test that AbLang rejects paired sequences."""

    @pytest.mark.slow
    def test_paired_sequence_raises_error(self):
        """Test that paired sequences raise PairedSequenceError."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG", light="DIQMTQSPS")
        with pytest.raises(PairedSequenceError):
            model.get_embeddings([seq])


class TestAbLangFormatting:
    """Test sequence formatting for AbLang."""

    @pytest.mark.slow
    def test_format_single_heavy(self):
        """Test formatting a single heavy chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        formatted = model._format_for_model([seq])
        assert len(formatted) == 1
        assert formatted[0] == "EVQLVESGG"

    @pytest.mark.slow
    def test_format_single_light(self):
        """Test formatting a single light chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        formatted = model._format_for_model([seq])
        assert len(formatted) == 1
        assert formatted[0] == "DIQMTQSPS"

    @pytest.mark.slow
    def test_format_mask_token_conversion(self):
        """Test that <MASK> is converted to *."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        formatted = model._format_for_model([seq])
        assert "*" in formatted[0]
        assert "<MASK>" not in formatted[0]


class TestAbLangLazyLoading:
    """Test lazy model loading."""

    @pytest.mark.slow
    def test_heavy_model_lazy_loaded(self):
        """Test heavy model is loaded on first use."""
        model = AbLang(devices="cpu")
        assert model._heavy_model is None
        _ = model._get_heavy_model()
        assert model._heavy_model is not None

    @pytest.mark.slow
    def test_light_model_lazy_loaded(self):
        """Test light model is loaded on first use."""
        model = AbLang(devices="cpu")
        assert model._light_model is None
        _ = model._get_light_model()
        assert model._light_model is not None


class TestAbLangTokenization:
    """Test AbLang tokenization."""

    @pytest.mark.slow
    def test_tokenize_heavy_returns_tensor_dict(self):
        """Test that tokenization returns expected tensor dictionary."""
        model = AbLang(devices="cpu")
        heavy_model = model._get_heavy_model()
        formatted = ["EVQLVESGG"]
        tokenized = model._tokenize_with_model(formatted, heavy_model)
        assert "input_ids" in tokenized
        assert isinstance(tokenized["input_ids"], torch.Tensor)

    @pytest.mark.slow
    def test_tokenize_light_returns_tensor_dict(self):
        """Test that tokenization returns expected tensor dictionary."""
        model = AbLang(devices="cpu")
        light_model = model._get_light_model()
        formatted = ["DIQMTQSPS"]
        tokenized = model._tokenize_with_model(formatted, light_model)
        assert "input_ids" in tokenized
        assert isinstance(tokenized["input_ids"], torch.Tensor)


class TestAbLangTokenOffsets:
    """Test token offset computation for AbLang."""

    @pytest.mark.slow
    def test_offsets_single_heavy(self):
        """Test offsets for single heavy chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        heavy_model = model._get_heavy_model()
        formatted = model._format_for_model([seq])
        tokenized = model._tokenize_with_model(formatted, heavy_model)
        offsets = model._compute_token_offsets([seq], tokenized)

        assert len(offsets) == 1
        assert "heavy" in offsets[0]
        start, end = offsets[0]["heavy"]
        assert start == 1  # After start token
        assert end - start == len("EVQLVESGG")

    @pytest.mark.slow
    def test_offsets_single_light(self):
        """Test offsets for single light chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        light_model = model._get_light_model()
        formatted = model._format_for_model([seq])
        tokenized = model._tokenize_with_model(formatted, light_model)
        offsets = model._compute_token_offsets([seq], tokenized)

        assert len(offsets) == 1
        assert "light" in offsets[0]
        start, end = offsets[0]["light"]
        assert start == 1  # After start token
        assert end - start == len("DIQMTQSPS")


class TestAbLangEmbeddings:
    """Test AbLang embedding extraction."""

    @pytest.mark.slow
    def test_embeddings_shape_heavy(self):
        """Test embedding shape for heavy chain sequence."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_embeddings([seq])

        assert output.embeddings.ndim == 3
        assert output.embeddings.shape[0] == 1  # batch size
        assert output.embeddings.shape[2] == 768  # embedding dim

    @pytest.mark.slow
    def test_embeddings_shape_light(self):
        """Test embedding shape for light chain sequence."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        output = model.get_embeddings([seq])

        assert output.embeddings.ndim == 3
        assert output.embeddings.shape[0] == 1  # batch size
        assert output.embeddings.shape[2] == 768  # embedding dim

    @pytest.mark.slow
    def test_embeddings_batch_heavy(self):
        """Test embeddings for batch of heavy chain sequences."""
        model = AbLang(devices="cpu")
        sequences = [
            AntibodySequence(heavy="EVQLVESGG"),
            AntibodySequence(heavy="QVQLVQSGA"),
        ]
        output = model.get_embeddings(sequences)

        assert output.embeddings.shape[0] == 2

    @pytest.mark.slow
    def test_embeddings_batch_light(self):
        """Test embeddings for batch of light chain sequences."""
        model = AbLang(devices="cpu")
        sequences = [
            AntibodySequence(light="DIQMTQSPS"),
            AntibodySequence(light="EIVLTQSPA"),
        ]
        output = model.get_embeddings(sequences)

        assert output.embeddings.shape[0] == 2

    @pytest.mark.slow
    def test_embeddings_mixed_batch(self):
        """Test embeddings for mixed batch of heavy and light chains."""
        model = AbLang(devices="cpu")
        sequences = [
            AntibodySequence(heavy="EVQLVESGG"),
            AntibodySequence(light="DIQMTQSPS"),
            AntibodySequence(heavy="QVQLVQSGA"),
            AntibodySequence(light="EIVLTQSPA"),
        ]
        output = model.get_embeddings(sequences)

        assert output.embeddings.shape[0] == 4
        assert output.embeddings.shape[2] == 768

    @pytest.mark.slow
    def test_chain_embeddings_extraction_heavy(self):
        """Test extracting heavy chain-specific embeddings."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_embeddings([seq])

        heavy_emb = output.get_chain_embeddings(0, "heavy")

        assert heavy_emb is not None
        assert heavy_emb.shape[0] == len("EVQLVESGG")

    @pytest.mark.slow
    def test_chain_embeddings_extraction_light(self):
        """Test extracting light chain-specific embeddings."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        output = model.get_embeddings([seq])

        light_emb = output.get_chain_embeddings(0, "light")

        assert light_emb is not None
        assert light_emb.shape[0] == len("DIQMTQSPS")


class TestAbLangAttention:
    """Test AbLang attention extraction."""

    @pytest.mark.slow
    def test_attention_output_heavy(self):
        """Test attention weight extraction for heavy chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_attention([seq])

        assert output.attention_weights.ndim == 5  # [batch, layers, heads, seq, seq]
        assert output.attention_weights.shape[0] == 1

    @pytest.mark.slow
    def test_attention_output_light(self):
        """Test attention weight extraction for light chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        output = model.get_attention([seq])

        assert output.attention_weights.ndim == 5
        assert output.attention_weights.shape[0] == 1


class TestAbLangLogits:
    """Test AbLang MLM logits."""

    @pytest.mark.slow
    def test_logits_output_heavy(self):
        """Test MLM logits extraction for heavy chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_logits([seq])

        assert output.logits.ndim == 3  # [batch, seq_len, vocab_size]
        assert output.logits.shape[0] == 1

    @pytest.mark.slow
    def test_logits_output_light(self):
        """Test MLM logits extraction for light chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        output = model.get_logits([seq])

        assert output.logits.ndim == 3
        assert output.logits.shape[0] == 1

    @pytest.mark.slow
    def test_logits_with_mask(self):
        """Test logits for masked sequence."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        output = model.get_logits([seq])

        assert output.logits.ndim == 3


class TestAbLangMaskFilling:
    """Test AbLang mask filling."""

    @pytest.mark.slow
    def test_fill_single_mask_heavy(self):
        """Test filling a single mask position in heavy chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        results = model.fill_mask([seq], top_k=3)

        assert len(results) == 1
        assert len(results[0]) <= 3
        for pred in results[0]:
            assert isinstance(pred, AntibodySequence)
            assert "<MASK>" not in pred.heavy_chain
            assert "*" not in pred.heavy_chain

    @pytest.mark.slow
    def test_fill_single_mask_light(self):
        """Test filling a single mask position in light chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQM<MASK>QSPS")
        results = model.fill_mask([seq], top_k=3)

        assert len(results) == 1
        assert len(results[0]) <= 3
        for pred in results[0]:
            assert isinstance(pred, AntibodySequence)
            assert "<MASK>" not in pred.light_chain
            assert "*" not in pred.light_chain

    @pytest.mark.slow
    def test_fill_mask_no_mask(self):
        """Test fill_mask with no mask token returns original."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        results = model.fill_mask([seq], top_k=3)

        assert len(results) == 1
        assert len(results[0]) == 1


class TestAbLangPseudoLogLikelihood:
    """Test AbLang pseudo log-likelihood computation."""

    @pytest.mark.slow
    def test_pseudo_ll_returns_float_heavy(self):
        """Test that pseudo log-likelihood returns a float for heavy chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        scores = model.pseudo_log_likelihood([seq])

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    @pytest.mark.slow
    def test_pseudo_ll_returns_float_light(self):
        """Test that pseudo log-likelihood returns a float for light chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        scores = model.pseudo_log_likelihood([seq])

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    @pytest.mark.slow
    def test_pseudo_ll_batch(self):
        """Test pseudo log-likelihood for batch."""
        model = AbLang(devices="cpu")
        sequences = [
            AntibodySequence(heavy="EVQLVESGG"),
            AntibodySequence(light="DIQMTQSPS"),
        ]
        scores = model.pseudo_log_likelihood(sequences)

        assert len(scores) == 2
        for score in scores:
            assert isinstance(score, float)


class TestAbLangMaskScan:
    """Test AbLang mask scanning."""

    @pytest.mark.slow
    def test_mask_scan_heavy(self):
        """Test mask scanning for heavy chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        results = model.mask_scan([seq])

        assert len(results) == 1
        assert results[0].logits is not None
        assert results[0].sequence == seq

    @pytest.mark.slow
    def test_mask_scan_light(self):
        """Test mask scanning for light chain."""
        model = AbLang(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        results = model.mask_scan([seq])

        assert len(results) == 1
        assert results[0].logits is not None
        assert results[0].sequence == seq


class TestAbLangRegistration:
    """Test AbLang model registration."""

    def test_ablang_in_registry(self):
        """Test that AbLang is registered."""
        assert "ablang" in MODEL_REGISTRY

    def test_ablang_config(self):
        """Test AbLang configuration."""
        config = get_model_config("ablang")
        assert config.name == "ablang"
        assert config.model_class == AbLang
        assert config.model_id == "ablang"
        assert config.supports_paired is False
        assert config.max_length == 160
        assert config.embedding_dim == 768
        assert config.mask_token == "*"
        assert config.separator is None
        assert config.has_mlm_head is True
        assert config.model_type == "encoder"

    def test_ablang_in_list_models(self):
        """Test that AbLang appears in list_models."""
        models = list_models()
        assert "ablang" in models
        assert models["ablang"] == "encoder"

    @pytest.mark.slow
    def test_load_model_ablang(self):
        """Test loading AbLang via load_model."""
        model = load_model("ablang", devices="cpu")
        assert isinstance(model, AbLang)
