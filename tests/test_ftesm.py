"""Tests for ft-ESM encoder model."""

import pytest
import torch

from ablms import AntibodySequence, load_model, list_models
from ablms.core.config import get_model_config, MODEL_REGISTRY
from ablms.encoders.ftesm import FtESM


class TestFtESMCreation:
    """Test ft-ESM model instantiation and attributes."""

    def test_model_name(self):
        """Test model name attribute."""
        assert FtESM.model_name == "ftesm"

    def test_supports_paired(self):
        """Test that ft-ESM supports paired sequences."""
        assert FtESM.supports_paired is True

    def test_max_length(self):
        """Test max sequence length."""
        assert FtESM.max_length == 1024

    def test_embedding_dim(self):
        """Test embedding dimension."""
        assert FtESM.embedding_dim == 1280

    def test_mask_token(self):
        """Test mask token."""
        assert FtESM.mask_token == "<mask>"

    def test_separator(self):
        """Test chain separator."""
        assert FtESM.separator == "<cls><cls>"

    def test_has_mlm_head(self):
        """Test that ft-ESM has MLM head."""
        assert FtESM.has_mlm_head is True

    @pytest.mark.slow
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = FtESM(devices="cpu")
        assert model is not None
        assert model._model is not None
        assert model._tokenizer is not None


class TestFtESMFormatting:
    """Test sequence formatting for ft-ESM."""

    @pytest.mark.slow
    def test_format_single_heavy(self):
        """Test formatting a single heavy chain."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        formatted = model._format_for_model([seq])
        assert len(formatted) == 1
        assert formatted[0] == "EVQLVESGG"

    @pytest.mark.slow
    def test_format_single_light(self):
        """Test formatting a single light chain."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(light="DIQMTQSPS")
        formatted = model._format_for_model([seq])
        assert len(formatted) == 1
        assert formatted[0] == "DIQMTQSPS"

    @pytest.mark.slow
    def test_format_paired_uses_cls_cls_separator(self):
        """Test that paired sequences use <cls><cls> separator."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG", light="DIQMTQSPS")
        formatted = model._format_for_model([seq])
        assert len(formatted) == 1
        assert formatted[0] == "EVQLVESGG<cls><cls>DIQMTQSPS"

    @pytest.mark.slow
    def test_format_mask_token_conversion(self):
        """Test that <MASK> is converted to <mask>."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        formatted = model._format_for_model([seq])
        assert "<mask>" in formatted[0]
        assert "<MASK>" not in formatted[0]

    @pytest.mark.slow
    def test_format_no_spaces_between_residues(self):
        """Test that ESM model does not add spaces between residues."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        formatted = model._format_for_model([seq])
        # No spaces should be added for ESM models
        assert " " not in formatted[0]


class TestFtESMTokenization:
    """Test ft-ESM tokenization."""

    @pytest.mark.slow
    def test_tokenize_returns_tensor_dict(self):
        """Test that tokenization returns expected tensor dictionary."""
        model = FtESM(devices="cpu")
        formatted = ["EVQLVESGG"]
        tokenized = model._tokenize(formatted)
        assert "input_ids" in tokenized
        assert "attention_mask" in tokenized
        assert isinstance(tokenized["input_ids"], torch.Tensor)
        assert isinstance(tokenized["attention_mask"], torch.Tensor)

    @pytest.mark.slow
    def test_tokenize_batch(self):
        """Test tokenization of multiple sequences."""
        model = FtESM(devices="cpu")
        formatted = ["EVQLVESGG", "QVQLVQSGA"]
        tokenized = model._tokenize(formatted)
        assert tokenized["input_ids"].shape[0] == 2


class TestFtESMTokenOffsets:
    """Test token offset computation for ft-ESM."""

    @pytest.mark.slow
    def test_offsets_single_heavy(self):
        """Test offsets for single heavy chain."""
        model = FtESM(devices="cpu")
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
    def test_offsets_paired_sequence(self):
        """Test offsets for paired sequence."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG", light="DIQMTQSPS")
        formatted = model._format_for_model([seq])
        tokenized = model._tokenize(formatted)
        offsets = model._compute_token_offsets([seq], tokenized)

        assert len(offsets) == 1
        assert "heavy" in offsets[0]
        assert "light" in offsets[0]

        heavy_start, heavy_end = offsets[0]["heavy"]
        light_start, light_end = offsets[0]["light"]

        # Heavy chain starts after [CLS]
        assert heavy_start == 1
        # Light chain starts after heavy + <cls><cls> separator
        assert light_start > heavy_end
        # The separator is 2 CLS tokens
        assert light_start == heavy_end + 2


class TestFtESMEmbeddings:
    """Test ft-ESM embedding extraction."""

    @pytest.mark.slow
    def test_embeddings_shape_single(self):
        """Test embedding shape for single sequence."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_embeddings([seq])

        assert output.embeddings.ndim == 3
        assert output.embeddings.shape[0] == 1  # batch size
        assert output.embeddings.shape[2] == 1280  # embedding dim

    @pytest.mark.slow
    def test_embeddings_shape_paired(self):
        """Test embedding shape for paired sequence."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG", light="DIQMTQSPS")
        output = model.get_embeddings([seq])

        assert output.embeddings.ndim == 3
        assert output.embeddings.shape[0] == 1
        assert output.embeddings.shape[2] == 1024

    @pytest.mark.slow
    def test_embeddings_batch(self):
        """Test embeddings for batch of sequences."""
        model = FtESM(devices="cpu")
        sequences = [
            AntibodySequence(heavy="EVQLVESGG"),
            AntibodySequence(heavy="QVQLVQSGA"),
        ]
        output = model.get_embeddings(sequences)

        assert output.embeddings.shape[0] == 2

    @pytest.mark.slow
    def test_chain_embeddings_extraction(self):
        """Test extracting chain-specific embeddings."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG", light="DIQMTQSPS")
        output = model.get_embeddings([seq])

        heavy_emb = output.get_chain_embeddings(0, "heavy")
        light_emb = output.get_chain_embeddings(0, "light")

        assert heavy_emb is not None
        assert light_emb is not None
        assert heavy_emb.shape[0] == len("EVQLVESGG")
        assert light_emb.shape[0] == len("DIQMTQSPS")


class TestFtESMAttention:
    """Test ft-ESM attention extraction."""

    @pytest.mark.slow
    def test_attention_output(self):
        """Test attention weight extraction."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_attention([seq])

        assert output.attention_weights.ndim == 5  # [batch, layers, heads, seq, seq]
        assert output.attention_weights.shape[0] == 1


class TestFtESMLogits:
    """Test ft-ESM MLM logits."""

    @pytest.mark.slow
    def test_logits_output(self):
        """Test MLM logits extraction."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        output = model.get_logits([seq])

        assert output.logits.ndim == 3  # [batch, seq_len, vocab_size]
        assert output.logits.shape[0] == 1

    @pytest.mark.slow
    def test_logits_with_mask(self):
        """Test logits for masked sequence."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQL<MASK>ESGG")
        output = model.get_logits([seq])

        assert output.logits.ndim == 3


class TestFtESMMaskFilling:
    """Test ft-ESM mask filling."""

    @pytest.mark.slow
    def test_fill_single_mask(self):
        """Test filling a single mask position."""
        model = FtESM(devices="cpu")
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
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        results = model.fill_mask([seq], top_k=3)

        assert len(results) == 1
        assert len(results[0]) == 1


class TestFtESMPseudoLogLikelihood:
    """Test ft-ESM pseudo log-likelihood computation."""

    @pytest.mark.slow
    def test_pseudo_ll_returns_float(self):
        """Test that pseudo log-likelihood returns a float."""
        model = FtESM(devices="cpu")
        seq = AntibodySequence(heavy="EVQLVESGG")
        scores = model.pseudo_log_likelihood([seq])

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    @pytest.mark.slow
    def test_pseudo_ll_batch(self):
        """Test pseudo log-likelihood for batch."""
        model = FtESM(devices="cpu")
        sequences = [
            AntibodySequence(heavy="EVQLVESGG"),
            AntibodySequence(heavy="QVQLVQSGA"),
        ]
        scores = model.pseudo_log_likelihood(sequences)

        assert len(scores) == 2
        for score in scores:
            assert isinstance(score, float)


class TestFtESMRegistration:
    """Test ft-ESM model registration."""

    def test_ftesm_in_registry(self):
        """Test that ft-ESM is registered."""
        assert "ftesm" in MODEL_REGISTRY

    def test_ftesm_config(self):
        """Test ft-ESM configuration."""
        config = get_model_config("ftesm")
        assert config.name == "ftesm"
        assert config.model_class == FtESM
        assert config.model_id == "brineylab/ft-ESM"
        assert config.supports_paired is True
        assert config.max_length == 1024
        assert config.embedding_dim == 1280
        assert config.mask_token == "<mask>"
        assert config.separator == "<cls><cls>"
        assert config.has_mlm_head is True
        assert config.model_type == "encoder"

    def test_ftesm_in_list_models(self):
        """Test that ft-ESM appears in list_models."""
        models = list_models()
        assert "ftesm" in models
        assert models["ftesm"] == "encoder"

    @pytest.mark.slow
    def test_load_model_ftesm(self):
        """Test loading ft-ESM via load_model."""
        model = load_model("ftesm", devices="cpu")
        assert isinstance(model, FtESM)
