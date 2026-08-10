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


class TestLayerCount:
    """num_layers must agree with the model's actual hidden_states tuple."""

    @pytest.mark.slow
    def test_matches_forward_pass(self, esm2_cpu, sequences):
        formatted = esm2_cpu._format_for_model(sequences[:1])
        tokenized = esm2_cpu._tokenize(formatted)
        hidden_states, _ = esm2_cpu._forward_all_hidden_states(tokenized)

        assert esm2_cpu.num_layers + 1 == len(hidden_states)

    @pytest.mark.slow
    def test_matches_the_checkpoint_variant(self, esm2_cpu):
        """The t6 checkpoint has 6 blocks; a hardcoded constant would not track this."""
        assert esm2_cpu.num_layers == 6


class TestMultiLayerBatchProcessing:
    """The layer axis sits at dimension 1, and pooling still reduces per batch."""

    @pytest.mark.slow
    def test_token_level_stacks_layers(self, esm2_cpu, sequences):
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=[0, 3, 6], pooling=None
        )

        assert embeddings.dim() == 4
        assert embeddings.shape[0] == len(sequences)
        assert embeddings.shape[1] == 3
        assert embeddings.shape[3] == esm2_cpu.embedding_dim
        assert mask is not None

    @pytest.mark.slow
    def test_pooling_reduces_before_transfer(self, esm2_cpu, sequences):
        """The [batch, layers, seq, hidden] tensor must never reach the queue."""
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=[0, 3, 6], pooling="mean"
        )

        assert embeddings.shape == (len(sequences), 3, esm2_cpu.embedding_dim)
        assert mask is None

    @pytest.mark.slow
    def test_each_layer_matches_a_single_layer_call(self, esm2_cpu, sequences):
        stacked, _, _ = esm2_cpu._process_embeddings_batch(
            sequences, layer=[0, 3, 6], pooling="mean"
        )

        for position, layer in enumerate([0, 3, 6]):
            single, _, _ = esm2_cpu._process_embeddings_batch(
                sequences, layer=layer, pooling="mean"
            )
            assert torch.allclose(stacked[:, position], single, atol=1e-6)

    @pytest.mark.slow
    def test_single_int_path_is_unchanged(self, esm2_cpu, sequences):
        """An int layer must not gain a layer axis."""
        embeddings, _, _ = esm2_cpu._process_embeddings_batch(
            sequences, layer=-1, pooling=None
        )
        assert embeddings.dim() == 3

    @pytest.mark.slow
    def test_a_wrong_layer_count_is_caught(self, esm2_cpu, sequences, monkeypatch):
        """A missing num_layers override must fail loudly, not mis-index.

        This is the guard for a newly added encoder whose config does not spell
        its depth `num_hidden_layers`.
        """
        monkeypatch.setattr(type(esm2_cpu), "num_layers", property(lambda self: 99))

        with pytest.raises(RuntimeError, match="num_layers"):
            esm2_cpu._process_embeddings_batch(sequences, layer=[0, 1], pooling="mean")


class TestMultiLayerGetEmbeddings:
    """The public API: layer accepts an int, a list, or "all"."""

    @pytest.mark.slow
    def test_default_call_is_unchanged(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(sequences, show_progress=False)

        assert output.embeddings.dim() == 3
        assert not output.is_multi_layer
        assert output.layer == -1
        assert output.layers is None

    @pytest.mark.slow
    def test_all_layers_pooled(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(
            sequences, layer="all", pooling="cls", batch_size=2, show_progress=False
        )

        assert output.is_multi_layer
        assert output.layers == list(range(esm2_cpu.num_layers + 1))
        assert output.layer is None
        assert output.is_pooled
        assert output.embeddings.shape == (
            len(sequences),
            esm2_cpu.num_layers + 1,
            esm2_cpu.embedding_dim,
        )

    @pytest.mark.slow
    def test_concat_layers_gives_one_vector_per_sequence(self, esm2_cpu, sequences):
        """The dimensionality-reduction use case."""
        output = esm2_cpu.get_embeddings(
            sequences, layer="all", pooling="cls", show_progress=False
        )
        features = output.concat_layers()

        assert features.shape == (
            len(sequences),
            (esm2_cpu.num_layers + 1) * esm2_cpu.embedding_dim,
        )

    @pytest.mark.slow
    def test_explicit_list_preserves_order(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(
            sequences, layer=[6, 0], pooling="mean", show_progress=False
        )
        assert output.layers == [6, 0]

    @pytest.mark.slow
    def test_get_layer_matches_a_single_layer_call(self, esm2_cpu, sequences):
        multi = esm2_cpu.get_embeddings(
            sequences, layer=[0, 3, 6], pooling="mean", show_progress=False
        )
        single = esm2_cpu.get_embeddings(
            sequences, layer=3, pooling="mean", show_progress=False
        )

        assert torch.allclose(multi.get_layer(3), single.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_single_element_list_keeps_the_layer_axis(self, esm2_cpu, sequences):
        """The argument's type decides the shape, not its length."""
        listed = esm2_cpu.get_embeddings(
            sequences, layer=[-1], pooling="mean", show_progress=False
        )
        scalar = esm2_cpu.get_embeddings(
            sequences, layer=-1, pooling="mean", show_progress=False
        )

        assert listed.embeddings.shape == (len(sequences), 1, esm2_cpu.embedding_dim)
        assert torch.allclose(listed.embeddings[:, 0], scalar.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_ragged_multi_batch_token_level_concatenates(
        self, esm2_cpu, ragged_sequences
    ):
        """Batches padded to different lengths must concatenate across the layer axis.

        This is the case that fails if _pad_tensors_to_max_length assumes
        dimension 1 is the sequence axis.
        """
        output = esm2_cpu.get_embeddings(
            ragged_sequences,
            layer=[0, 3],
            pooling=None,
            batch_size=2,
            show_progress=False,
        )
        n = len(ragged_sequences)
        max_len = output.embeddings.shape[2]

        assert output.embeddings.shape == (n, 2, max_len, esm2_cpu.embedding_dim)
        assert output.attention_mask.shape == (n, max_len)

    @pytest.mark.slow
    def test_invalid_layer_fails_at_the_call_site(self, esm2_cpu, sequences):
        with pytest.raises(ValueError, match="out of range"):
            esm2_cpu.get_embeddings(sequences, layer=999, show_progress=False)

    @pytest.mark.slow
    def test_empty_input_reports_the_layer_axis(self, esm2_cpu):
        output = esm2_cpu.get_embeddings([], layer="all", pooling="mean")

        assert output.layers == list(range(esm2_cpu.num_layers + 1))
        assert output.embeddings.shape == (
            0,
            esm2_cpu.num_layers + 1,
            esm2_cpu.embedding_dim,
        )


class TestMultiLayerIterEmbeddings:
    @pytest.mark.slow
    def test_each_batch_carries_its_layers(self, esm2_cpu, sequences):
        outputs = list(
            esm2_cpu.iter_embeddings(
                sequences,
                layer=[0, 3],
                pooling="mean",
                batch_size=2,
                show_progress=False,
            )
        )

        assert len(outputs) == 3
        assert all(o.layers == [0, 3] for o in outputs)
        assert outputs[0].embeddings.shape == (2, 2, esm2_cpu.embedding_dim)

    @pytest.mark.slow
    def test_stream_matches_get_embeddings(self, esm2_cpu, sequences):
        streamed = torch.cat(
            [
                o.embeddings
                for o in esm2_cpu.iter_embeddings(
                    sequences,
                    layer="all",
                    pooling="mean",
                    batch_size=2,
                    show_progress=False,
                )
            ]
        )
        combined = esm2_cpu.get_embeddings(
            sequences, layer="all", pooling="mean", batch_size=2, show_progress=False
        )

        assert torch.allclose(streamed, combined.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_invalid_layer_fails_eagerly(self, esm2_cpu, sequences):
        """Like sequence validation, this must raise on call, not on first next()."""
        with pytest.raises(ValueError, match="out of range"):
            esm2_cpu.iter_embeddings(sequences, layer=999, show_progress=False)
