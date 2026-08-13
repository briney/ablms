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
every residue has silently become `[UNK]` — the exact failure mode that hid one
of the three upstream bugs these tests now cover. See
`EncoderAbLM`/`GenerativeAbLM` `_load_model` compatibility shims and issue #5.
"""

from __future__ import annotations

import pytest
import torch

import ablms
from ablms.core.sequence import ChainType, Species

pytestmark = [pytest.mark.smoke, pytest.mark.slow]

# The 20 canonical amino acids. A generated sequence containing anything else
# means the tokenizer decoded to junk even if generation itself succeeded.
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Two distinct human VH framework openings of equal length. Equal length is the
# point: under a degenerate all-`[UNK]` vocabulary these tokenize identically,
# so identical embeddings prove the tokenizer is broken.
VH_A = "EVQLVESGGGLVQPGRSLRLSCAAS"
VH_B = "QVQLVQSGAEVKKPGASVKVSCKAS"


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
