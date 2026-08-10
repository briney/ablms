"""Tests for bounded-memory embedding extraction."""

from __future__ import annotations

import pytest
import torch

from ablms import AntibodySequence
from ablms.encoders import ESM2

MODEL_ID = "facebook/esm2_t6_8M_UR50D"

HEAVY_CHAINS = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQG",
    "EVQLVESGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKG",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKS",
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKG",
]


@pytest.fixture(scope="session")
def esm2_cpu():
    """Session-scoped real model; loading weights is the expensive part."""
    return ESM2(devices="cpu", model_id=MODEL_ID)


@pytest.fixture(scope="session")
def sequences():
    return [AntibodySequence(heavy=h) for h in HEAVY_CHAINS]


@pytest.fixture(scope="session")
def ragged_sequences():
    """Sequences with one much shorter chain, so batches pad to different lengths."""
    chains = HEAVY_CHAINS + ["EVQLVESGGGLIQ"]  # deliberately much shorter
    return [AntibodySequence(heavy=h) for h in chains]


class TestBatchLevelPooling:
    """_process_embeddings_batch reduces before returning."""

    @pytest.mark.slow
    def test_pooling_returns_two_dimensional_result(self, esm2_cpu, sequences):
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=-1, pooling="mean"
        )
        assert embeddings.dim() == 2
        assert embeddings.shape == (len(sequences), esm2_cpu.embedding_dim)
        assert mask is None
        assert len(offsets) == len(sequences)

    @pytest.mark.slow
    def test_no_pooling_still_returns_token_level(self, esm2_cpu, sequences):
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=-1, pooling=None
        )
        assert embeddings.dim() == 3
        assert embeddings.shape[0] == len(sequences)
        assert embeddings.shape[2] == esm2_cpu.embedding_dim
        assert mask is not None


class TestPooledEmbeddingsBatchInvariant:
    """Pooled values from get_embeddings must not depend on the batch split.

    This compares the current implementation against itself at two batch sizes;
    it is not a comparison against the pre-refactor implementation. What it
    pins down is that per-batch pooling is padding-invariant, which is the
    property that would break if the reduction were sensitive to how much
    padding a given batch happened to carry.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("strategy", ["mean", "max", "cls", "first", "last"])
    def test_multi_batch_matches_single_batch(self, esm2_cpu, sequences, strategy):
        """Splitting into batches must not change pooled values.

        Pooling now runs per batch rather than once over a globally padded
        stack. Processing the same input as one batch and as three batches must
        agree, which is exactly the property that would break if per-batch
        pooling were not padding-invariant.
        """
        one_batch = esm2_cpu.get_embeddings(
            sequences, pooling=strategy, batch_size=len(sequences), show_progress=False
        )
        many_batches = esm2_cpu.get_embeddings(
            sequences, pooling=strategy, batch_size=2, show_progress=False
        )

        assert one_batch.embeddings.shape == (len(sequences), esm2_cpu.embedding_dim)
        assert torch.allclose(one_batch.embeddings, many_batches.embeddings, atol=1e-5)

    @pytest.mark.slow
    def test_pooled_output_fields(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(
            sequences, pooling="mean", batch_size=2, show_progress=False
        )
        assert output.is_pooled
        assert output.pooled is not None
        assert output.attention_mask is None
        assert len(output.token_offsets) == len(sequences)


class TestIterEmbeddings:
    """Streaming embeddings yield per batch and agree with get_embeddings."""

    @pytest.mark.slow
    def test_yields_one_output_per_batch(self, esm2_cpu, sequences):
        outputs = list(
            esm2_cpu.iter_embeddings(sequences, batch_size=2, show_progress=False)
        )
        assert len(outputs) == 3  # 5 sequences at batch_size 2
        assert [len(o) for o in outputs] == [2, 2, 1]
        assert all(o.sequences is not None for o in outputs)
        assert outputs[0].sequences[0] is sequences[0]
        assert outputs[2].sequences[0] is sequences[4]

    @pytest.mark.slow
    def test_pooled_stream_matches_get_embeddings(self, esm2_cpu, sequences):
        streamed = torch.cat(
            [
                o.embeddings
                for o in esm2_cpu.iter_embeddings(
                    sequences, pooling="mean", batch_size=2, show_progress=False
                )
            ]
        )
        combined = esm2_cpu.get_embeddings(
            sequences, pooling="mean", batch_size=2, show_progress=False
        )
        assert torch.allclose(streamed, combined.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_token_level_stream_matches_get_embeddings(self, esm2_cpu, sequences):
        """Compare per sequence, not per tensor.

        get_embeddings pads every batch to a single global maximum length,
        while each streamed batch is padded only to its own maximum. Both
        describe the same embeddings, so compare with get_sequence_tokens(),
        which strips padding via the attention mask.
        """
        streamed = [
            tokens
            for output in esm2_cpu.iter_embeddings(
                sequences, batch_size=2, show_progress=False
            )
            for tokens in output
        ]
        combined = esm2_cpu.get_embeddings(sequences, batch_size=2, show_progress=False)

        assert len(streamed) == len(sequences)
        for i, tokens in enumerate(streamed):
            assert torch.allclose(tokens, combined.get_sequence_tokens(i), atol=1e-6)

    @pytest.mark.slow
    def test_validation_is_eager(self, esm2_cpu):
        """Invalid input must raise on call, not on first next()."""
        from ablms import PairedSequenceError

        paired = AntibodySequence(heavy=HEAVY_CHAINS[0], light="DIQMTQSPSSLSASVGDRV")
        with pytest.raises(PairedSequenceError):
            esm2_cpu.iter_embeddings([paired], show_progress=False)

    @pytest.mark.slow
    def test_empty_input_yields_nothing(self, esm2_cpu):
        assert list(esm2_cpu.iter_embeddings([], show_progress=False)) == []


class TestRaggedBatchTokenLevel:
    """get_embeddings(pooling=None) must survive batches padded to different lengths.

    _pad_tensors_to_max_length is applied to every tensor in the result tuple,
    including the [batch, seq_len] attention mask, not just the [batch, seq_len,
    hidden_dim] embeddings. Regression coverage for a guard change that stopped
    the mask from being re-padded before concatenation across batches.
    """

    @pytest.mark.slow
    def test_multi_batch_token_level_concatenates(self, esm2_cpu, ragged_sequences):
        output = esm2_cpu.get_embeddings(
            ragged_sequences, pooling=None, batch_size=2, show_progress=False
        )
        n = len(ragged_sequences)
        max_len = output.embeddings.shape[1]

        assert output.embeddings.shape == (n, max_len, esm2_cpu.embedding_dim)
        assert output.attention_mask is not None
        assert output.attention_mask.shape == (n, max_len)
