"""Contract tests for the IgLM wrapper's use of the `iglm` package API.

These do not instantiate the model. They check that the wrapper's calls match
the installed package's signatures and vocabulary, which is enough to catch
upstream API drift and needs no weights, network, or GPU.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ablms.core.sequence import ChainType, Species
from ablms.generators.iglm import CHAIN_TYPE_MAP, SPECIES_MAP


@pytest.fixture(scope="session")
def iglm_vocab() -> set[str]:
    """Every token in the installed iglm package's vocabulary."""
    iglm_model = pytest.importorskip("iglm.model.IgLM")
    return set(Path(iglm_model.VOCAB_FILE).read_text().split())


@pytest.fixture(scope="session")
def iglm_class():
    """The installed IgLM class, without instantiating it."""
    return pytest.importorskip("iglm").IgLM


class TestTokenMaps:
    def test_species_map_values_are_vocab_tokens(self, iglm_vocab):
        for species, token in SPECIES_MAP.items():
            assert token in iglm_vocab, f"{species} maps to {token!r}, not in vocab"

    def test_chain_map_values_are_vocab_tokens(self, iglm_vocab):
        for chain, token in CHAIN_TYPE_MAP.items():
            assert token in iglm_vocab, f"{chain} maps to {token!r}, not in vocab"

    def test_every_enum_member_is_mapped(self):
        assert set(SPECIES_MAP) == set(Species)
        assert set(CHAIN_TYPE_MAP) == set(ChainType)


class TestCallSignatures:
    """The wrapper's keyword arguments must bind against the real signatures."""

    def test_generate_kwargs_bind(self, iglm_class):
        sig = inspect.signature(iglm_class.generate)
        sig.bind(
            None,  # self
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
            prompt_sequence=None,
            num_to_generate=1,
            top_p=1.0,
            temperature=1.0,
        )

    def test_infill_kwargs_bind(self, iglm_class):
        sig = inspect.signature(iglm_class.infill)
        sig.bind(
            None,  # self
            sequence="EVQL",
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
            infill_range=(1, 2),
            num_to_generate=1,
            temperature=1.0,
        )

    def test_log_likelihood_kwargs_bind(self, iglm_class):
        sig = inspect.signature(iglm_class.log_likelihood)
        sig.bind(
            None,  # self
            sequence="EVQL",
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
        )

    @pytest.mark.parametrize("method", ["generate", "infill", "log_likelihood"])
    def test_wrapper_does_not_pass_our_enum_names(self, iglm_class, method):
        """`chain_type` and `species` are our names; iglm uses `*_token`."""
        params = inspect.signature(getattr(iglm_class, method)).parameters
        assert "chain_token" in params
        assert "species_token" in params
        assert "chain_type" not in params
        assert "species" not in params
