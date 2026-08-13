"""Contract tests for the IgLM wrapper's use of the `iglm` package API.

These do not instantiate the real model (no weights, network, or GPU
required). The `TestWrapperCallContract` tests drive the wrapper's own
`_generate`/`_infill`/`_compute_log_likelihood` methods against a
`create_autospec` double of `iglm.IgLM`, so a call is only accepted if it
actually binds against the real third-party signature. Passing the wrong
keyword (e.g. reintroducing `chain_type=` in place of `chain_token=`) raises
`TypeError` at call time, which is exactly the regression these tests exist
to catch. They also cover the wrapper's handling of `iglm`'s list-of-strings
and bare-float return values.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from ablms.core.sequence import AntibodySequence, ChainType, Species
from ablms.generators.iglm import CHAIN_TYPE_MAP, SPECIES_MAP, IgLM


@pytest.fixture(scope="session")
def iglm_vocab() -> set[str]:
    """Every token in the installed iglm package's vocabulary."""
    # `importlib.import_module` (rather than `import iglm.model.IgLM as ...`)
    # is required here: `iglm/model/__init__.py` rebinds the name `IgLM` in
    # its own namespace to the class, which would otherwise shadow the
    # submodule when accessed via attribute-style import.
    iglm_model_module = importlib.import_module("iglm.model.IgLM")

    return set(Path(iglm_model_module.VOCAB_FILE).read_text().split())


@pytest.fixture(scope="session")
def iglm_class():
    """The installed IgLM class, without instantiating it."""
    import iglm

    return iglm.IgLM


@pytest.fixture
def iglm_wrapper(iglm_class) -> IgLM:
    """An `IgLM` wrapper instance with a mocked `_iglm` backend.

    Bypasses `IgLM.__init__`/`_load_model()`, which would otherwise
    instantiate the real (and, under transformers 5.x, broken) model. Only
    `_iglm` is set, since that is the only attribute `_generate`, `_infill`,
    and `_compute_log_likelihood` touch on `self`.
    """
    wrapper = object.__new__(IgLM)
    wrapper._iglm = create_autospec(iglm_class, instance=True)
    return wrapper


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


class TestWrapperCallContract:
    """Exercise the wrapper's own methods against an autospecced backend."""

    def test_generate_calls_backend_and_handles_list_return(self, iglm_wrapper):
        iglm_wrapper._iglm.generate.return_value = ["EVQLVESGG", "EVQLVESGH"]
        iglm_wrapper._iglm.log_likelihood.return_value = -1.5

        sequences, scores = iglm_wrapper._generate(
            num_sequences=2,
            chain_type=ChainType.HEAVY,
            species=Species.HUMAN,
            prompt=None,
            temperature=1.0,
            top_k=None,
            top_p=None,
            max_length=None,
        )

        iglm_wrapper._iglm.generate.assert_called_once_with(
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
            prompt_sequence=None,
            num_to_generate=2,
            top_p=1.0,
            temperature=1.0,
        )
        assert len(sequences) == 2
        assert all(isinstance(s, AntibodySequence) for s in sequences)
        assert [s.heavy_chain for s in sequences] == ["EVQLVESGG", "EVQLVESGH"]
        assert scores == [-1.5, -1.5]

    def test_infill_with_mask_range_calls_backend_and_handles_list_return(
        self, iglm_wrapper
    ):
        iglm_wrapper._iglm.infill.return_value = ["EVQLW"]
        iglm_wrapper._iglm.log_likelihood.return_value = -3.25

        sequence = AntibodySequence(heavy="EVQLX")
        sequences, scores = iglm_wrapper._infill(
            sequence=sequence,
            mask_range=(1, 2),
            num_sequences=1,
            chain_type=ChainType.HEAVY,
            species=Species.HUMAN,
            temperature=1.0,
        )

        iglm_wrapper._iglm.infill.assert_called_once_with(
            sequence="EVQLX",
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
            infill_range=(1, 2),
            temperature=1.0,
            num_to_generate=1,
        )
        assert [s.heavy_chain for s in sequences] == ["EVQLW"]
        assert scores == [-3.25]

    def test_infill_with_mask_token_calls_backend(self, iglm_wrapper):
        """Locks in the wrapper's current call shape for the `<MASK>`-token path.

        The coordinates passed here have a pre-existing, out-of-scope coordinate
        bug tracked in issue #5: they are computed against positions in the
        masked string but then applied to a copy of the string with the mask
        token already deleted. This test asserts only that the call shape is
        unchanged, not that it is correct.
        """
        iglm_wrapper._iglm.infill.return_value = ["ACDWFG"]
        iglm_wrapper._iglm.log_likelihood.return_value = -0.5

        sequence = AntibodySequence(heavy="ACD<MASK>EFG")
        sequences, scores = iglm_wrapper._infill(
            sequence=sequence,
            mask_range=None,
            num_sequences=1,
            chain_type=ChainType.HEAVY,
            species=Species.HUMAN,
            temperature=1.0,
        )

        iglm_wrapper._iglm.infill.assert_called_once_with(
            sequence="ACDEFG",
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
            infill_range=(3, 4),
            temperature=1.0,
            num_to_generate=1,
        )
        assert [s.heavy_chain for s in sequences] == ["ACDWFG"]
        assert scores == [-0.5]

    def test_compute_log_likelihood_calls_backend_and_returns_float(self, iglm_wrapper):
        iglm_wrapper._iglm.log_likelihood.return_value = -4.0

        score = iglm_wrapper._compute_log_likelihood(
            sequence=AntibodySequence(heavy="EVQL"),
            chain_type=ChainType.HEAVY,
            species=Species.HUMAN,
        )

        iglm_wrapper._iglm.log_likelihood.assert_called_once_with(
            sequence="EVQL",
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
        )
        assert score == -4.0
