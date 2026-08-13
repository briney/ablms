"""Offline smoke tests for backends that bundle their own weights.

`iglm` and `antiberty` ship weights inside the package, so these run with no
network and no GPU. They exist because every test that loads real weights is
marked `slow` and deselected in CI, which is how these two models came to be
broken under transformers 5.x without the suite noticing.

These go through `ablms.load_model(...)` and the library's own public API
(`generate()` for IgLM, `get_embeddings()` for AntiBERTy) rather than the
third-party packages directly. That matters: calling `iglm.IgLM(...)` or
`antiberty.AntiBERTyRunner()` straight would test the dependency rather than
this library's wrapper code.

The assertions here deliberately check *meaning* rather than shape. An earlier
version asserted only `embeddings.shape[-1] == 512`, which passes even when
every residue has silently become `[UNK]`. That is not hypothetical: it is
precisely what hid one of the five bugs these tests now cover, three upstream
and two in this library's own AntiBERTy wrapper. A model that loads and returns
correctly shaped garbage is the failure mode to defend against, so each test
asserts something a broken model would get wrong - different sequences must
embed differently, probabilities must sum to one, and a real VH must outscore
poly-alanine.

See `ablms.utils.compat` for the upstream shims and issue #5 for the history.
"""

from __future__ import annotations

import pytest
import torch

import ablms
from ablms.core.sequence import AntibodySequence, ChainType, Species

pytestmark = [pytest.mark.smoke, pytest.mark.slow]

# The 20 canonical amino acids. A generated sequence containing anything else
# means the tokenizer decoded to junk even if generation itself succeeded.
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Two distinct human VH framework openings of equal length. Equal length is the
# point: under a degenerate all-`[UNK]` vocabulary these tokenize identically,
# so identical embeddings prove the tokenizer is broken.
VH_A = "EVQLVESGGGLVQPGRSLRLSCAAS"
VH_B = "QVQLVQSGAEVKKPGASVKVSCKAS"

# AntiBERTy's vocabulary: the 20 amino acids plus [PAD] [UNK] [CLS] [SEP] [MASK].
ANTIBERTY_VOCAB_SIZE = 25


def test_iglm_generates_a_plausible_sequence():
    """IgLM's wrapper should generate a real antibody sequence, not junk."""
    model = ablms.load_model("iglm")
    output = model.generate(
        num_sequences=1,
        chain_type=ChainType.HEAVY,
        species=Species.HUMAN,
        temperature=1.0,
    )

    assert len(output.sequences) == 1
    generated = output.sequences[0].heavy_chain

    # Meaning, not shape: a degenerate vocabulary cannot produce a residue
    # string of realistic length drawn only from the amino-acid alphabet.
    assert len(generated) >= 20, f"implausibly short sequence: {generated!r}"
    unexpected = set(generated) - AMINO_ACIDS
    assert not unexpected, f"non-amino-acid characters {unexpected} in {generated!r}"

    # Scores come from IgLM's log_likelihood, so a finite value confirms the
    # scoring path ran rather than silently defaulting.
    assert output.scores is not None
    assert len(output.scores) == 1
    score = output.scores[0]
    assert score == score, "score is NaN"  # NaN is the only value != itself
    assert score < 0, f"a mean per-token log-probability must be negative: {score}"


def test_antiberty_distinguishes_two_sequences():
    """AntiBERTy must produce *different* embeddings for different sequences.

    This is the assertion that catches a degenerate tokenizer. `VH_A` and
    `VH_B` are the same length, so if every residue maps to `[UNK]` the two
    token sequences are identical and so are their embeddings.
    """
    model = ablms.load_model("antiberty")
    output = model.get_embeddings([VH_A, VH_B])

    assert output.embeddings.shape[0] == 2
    assert output.embeddings.shape[-1] == 512

    first, second = output.embeddings[0], output.embeddings[1]
    assert not torch.allclose(first, second), (
        "identical embeddings for two different sequences - the tokenizer is "
        "collapsing residues to [UNK]"
    )
    assert torch.isfinite(first).all(), "non-finite values in embedding"


def test_antiberty_embeds_a_single_sequence():
    """The single-sequence path should also produce finite, sane embeddings."""
    model = ablms.load_model("antiberty")
    output = model.get_embeddings([VH_A])

    assert output.embeddings.shape[0] == 1
    assert output.embeddings.shape[-1] == 512
    assert torch.isfinite(output.embeddings).all()
    # An all-`[UNK]` run collapses to one repeated row; real embeddings vary
    # position to position.
    per_position = output.embeddings[0]
    assert not torch.allclose(
        per_position[0], per_position[1]
    ), "all positions identical - residues are not being distinguished"


def test_antiberty_logits_form_a_distribution_over_the_vocabulary():
    """AntiBERTy's MLM head should yield real logits, one row per token.

    AntiBERTy names its MLM output `prediction_logits`, not the `logits` most
    HuggingFace heads use, and its `loss` field defaults to the integer 0 rather
    than None - so a positional fallback to `outputs[0]` silently returns that 0.
    """
    model = ablms.load_model("antiberty")
    output = model.get_logits([VH_A])

    # One row per token: the residues plus [CLS] and [SEP].
    assert output.logits.shape[0] == 1
    assert output.logits.shape[1] == len(VH_A) + 2
    assert output.logits.shape[-1] == ANTIBERTY_VOCAB_SIZE
    assert torch.isfinite(output.logits).all()

    probabilities = output.probabilities[0]
    row_sums = probabilities.sum(dim=-1)
    assert torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-4
    ), f"probabilities do not sum to 1: {row_sums[:5].tolist()}"


def test_antiberty_fills_a_mask_with_an_amino_acid():
    """A masked position should be filled with a real residue."""
    model = ablms.load_model("antiberty")
    masked = AntibodySequence(heavy="EVQLVESGGG<MASK>VQPGRSLRLSCAAS")

    filled = model.fill_mask([masked], top_k=3)

    assert len(filled) == 1
    predictions = filled[0]
    assert len(predictions) == 3, f"expected 3 candidates, got {len(predictions)}"
    for candidate in predictions:
        sequence = candidate.heavy_chain
        assert len(sequence) == len("EVQLVESGGGXVQPGRSLRLSCAAS")
        unexpected = set(sequence) - AMINO_ACIDS
        assert not unexpected, f"non-amino-acid characters {unexpected} in {sequence!r}"


def test_antiberty_prefers_a_real_antibody_to_poly_alanine():
    """Pseudo-log-likelihood must rank a real VH above a degenerate sequence.

    This is the assertion that distinguishes a working MLM head from one
    returning arbitrary numbers: a model trained on antibodies should find a
    genuine framework far more probable than a run of alanines of equal length.
    """
    model = ablms.load_model("antiberty")
    real, poly_alanine = VH_A, "A" * len(VH_A)

    scores = model.pseudo_log_likelihood([real, poly_alanine])

    assert len(scores) == 2
    real_score, degenerate_score = scores
    assert real_score == real_score, "score is NaN"
    assert real_score < 0, f"a summed log-probability must be negative: {real_score}"
    assert real_score > degenerate_score, (
        f"real VH scored {real_score:.2f} but poly-alanine scored "
        f"{degenerate_score:.2f}; the MLM head is not discriminating"
    )
