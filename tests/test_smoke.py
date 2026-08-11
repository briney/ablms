"""Offline smoke tests for backends that bundle their own weights.

`iglm` and `antiberty` ship weights inside the package, so these run with no
network and no GPU. They exist because every test that loads real weights is
marked `slow` and deselected in CI, which is how these two models came to be
broken under transformers 5.x without the suite noticing.

These go through `ablms.load_model(...)` and the library's own public API
(`generate()` for IgLM, `get_embeddings()` for AntiBERTy) rather than the
third-party packages directly. That matters: if these called `iglm.IgLM(...)`
or `antiberty.AntiBERTyRunner()` straight, the job would flip green the
moment the upstream incompatibility is fixed, without ever having executed
this library's wrapper code. Routing through `ablms` means the wrapper is
still exercised even while it's unreachable for the upstream reason below.

Both are expected to fail until that incompatibility is resolved (tracking
issue #5): `BertTokenizerFast(vocab_file=...)` no longer loads the vocabulary
under transformers 5, so IgLM's control tokens all map to `[UNK]` and its own
assertion trips; AntiBERTy separately hits a missing `all_tied_weights_keys`
attribute during `from_pretrained`. The CI job running them is non-blocking
for exactly this reason.
"""

from __future__ import annotations

import pytest

import ablms
from ablms.core.sequence import ChainType, Species

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def test_iglm_generates_one_sequence():
    """IgLM's wrapper should produce a sequence from control tokens."""
    model = ablms.load_model("iglm")
    output = model.generate(
        num_sequences=1,
        chain_type=ChainType.HEAVY,
        species=Species.HUMAN,
        temperature=1.0,
    )
    assert len(output.sequences) == 1
    assert output.sequences[0].heavy_chain.isalpha()


def test_antiberty_embeds_one_sequence():
    """AntiBERTy's wrapper should return an embedding for a single heavy chain."""
    model = ablms.load_model("antiberty")
    output = model.get_embeddings(["EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"])
    assert output.embeddings.shape[-1] == 512
