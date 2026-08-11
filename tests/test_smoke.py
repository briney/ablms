"""Offline smoke tests for backends that bundle their own weights.

`iglm` and `antiberty` ship weights inside the package, so these run with no
network and no GPU. They exist because every test that loads real weights is
marked `slow` and deselected in CI, which is how these two models came to be
broken under transformers 5.x without the suite noticing.

Both are expected to fail until that incompatibility is resolved. The CI job
running them is non-blocking for exactly that reason.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def test_iglm_generates_one_sequence():
    """IgLM's small checkpoint should produce a sequence from control tokens."""
    iglm = pytest.importorskip("iglm")
    model = iglm.IgLM(model_name="IgLM-S")
    generated = model.generate(
        chain_token="[HEAVY]",
        species_token="[HUMAN]",
        num_to_generate=1,
        temperature=1.0,
    )
    assert len(generated) == 1
    assert generated[0].isalpha()


def test_antiberty_embeds_one_sequence():
    """AntiBERTy should return an embedding for a single heavy chain."""
    antiberty = pytest.importorskip("antiberty")
    runner = antiberty.AntiBERTyRunner()
    embeddings = runner.embed(["EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"])
    assert len(embeddings) == 1
    assert embeddings[0].shape[-1] == 512
