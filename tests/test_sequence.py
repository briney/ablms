"""Tests for AntibodySequence class."""

import pytest

from ablms.core.sequence import AntibodySequence, ChainType, Species
from ablms.exceptions import InvalidAminoAcidError, InvalidSequenceError


class TestAntibodySequenceCreation:
    """Test AntibodySequence creation methods."""

    def test_heavy_only(self):
        """Test creating sequence from heavy chain."""
        seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        assert seq.heavy_chain == "EVQLVESGGGLVQ"
        assert seq.light_chain is None
        assert seq.species == Species.UNKNOWN
        assert not seq.is_paired

    def test_light_only(self):
        """Test creating sequence from light chain."""
        seq = AntibodySequence(light="DIQMTQSPSSLSA")
        assert seq.heavy_chain is None
        assert seq.light_chain == "DIQMTQSPSSLSA"
        assert not seq.is_paired

    def test_paired(self):
        """Test creating paired sequence."""
        seq = AntibodySequence(
            heavy="EVQLVESGGGLVQ",
            light="DIQMTQSPSSLSA",
            species=Species.HUMAN,
        )
        assert seq.heavy_chain == "EVQLVESGGGLVQ"
        assert seq.light_chain == "DIQMTQSPSSLSA"
        assert seq.species == Species.HUMAN
        assert seq.is_paired

    def test_requires_at_least_one_chain(self):
        """Test that at least one chain is required."""
        with pytest.raises(InvalidSequenceError):
            AntibodySequence()

    def test_empty_chain_raises_error(self):
        """Test that empty chain raises error."""
        with pytest.raises(InvalidSequenceError):
            AntibodySequence(heavy="")

    def test_keyword_only_arguments(self):
        """Test that positional arguments are not allowed."""
        with pytest.raises(TypeError):
            AntibodySequence("EVQLVESGGGLVQ")  # type: ignore


class TestAntibodySequenceValidation:
    """Test AntibodySequence validation."""

    def test_valid_amino_acids(self):
        """Test that valid amino acids are accepted."""
        # All standard amino acids
        seq = AntibodySequence(heavy="ACDEFGHIKLMNPQRSTVWY")
        assert seq.heavy_chain == "ACDEFGHIKLMNPQRSTVWY"

    def test_invalid_amino_acid(self):
        """Test that invalid amino acids raise error."""
        with pytest.raises(InvalidAminoAcidError):
            AntibodySequence(heavy="EVQL1ESGGGLVQ")

    def test_lowercase_invalid(self):
        """Test that lowercase amino acids are invalid."""
        with pytest.raises(InvalidAminoAcidError):
            AntibodySequence(heavy="evqlvesggglvq")

    def test_mask_token_allowed(self):
        """Test that mask tokens are allowed."""
        seq = AntibodySequence(heavy="EVQ<MASK>VESGGGLVQ")
        assert "<MASK>" in seq.heavy_chain


class TestAntibodySequenceMasking:
    """Test AntibodySequence masking functionality."""

    def test_is_masked(self):
        """Test is_masked property."""
        seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        assert not seq.is_masked

        masked_seq = AntibodySequence(heavy="EVQ<MASK>VESGGGLVQ")
        assert masked_seq.is_masked

    def test_masked_positions(self):
        """Test masked_positions property."""
        seq = AntibodySequence(heavy="EVQ<MASK>VES<MASK>GLVQ")
        positions = seq.masked_positions
        assert "heavy" in positions
        assert len(positions["heavy"]) == 2

    def test_with_mask(self):
        """Test with_mask method."""
        seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        masked = seq.with_mask("heavy", [3, 4, 5])

        assert "<MASK>" in masked.heavy_chain
        assert masked.is_masked

    def test_with_mask_invalid_chain(self):
        """Test with_mask raises error for invalid chain."""
        seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        with pytest.raises(ValueError):
            seq.with_mask("invalid", [0])

    def test_with_mask_position_out_of_range(self):
        """Test with_mask raises error for out of range position."""
        seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        with pytest.raises(ValueError):
            seq.with_mask("heavy", [100])


class TestAntibodySequenceProperties:
    """Test AntibodySequence properties."""

    def test_length_single_chain(self):
        """Test length property for single chain."""
        seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        assert seq.length == {"heavy": 13}
        assert seq.total_length == 13

    def test_length_paired(self):
        """Test length property for paired chains."""
        seq = AntibodySequence(
            heavy="EVQLVESGGGLVQ",
            light="DIQMTQ",
        )
        assert seq.length == {"heavy": 13, "light": 6}
        assert seq.total_length == 19

    def test_length_with_mask(self):
        """Test that mask tokens count as single position."""
        seq = AntibodySequence(heavy="EVQ<MASK>VESGGGLVQ")
        # 13 amino acids with <MASK> replacing one position
        assert seq.length["heavy"] == 13

    def test_get_sequence(self):
        """Test get_sequence method."""
        seq = AntibodySequence(
            heavy="EVQLVESGGGLVQ",
            light="DIQMTQ",
        )
        assert seq.get_sequence("heavy") == "EVQLVESGGGLVQ"
        assert seq.get_sequence("light") == "DIQMTQ"

    def test_get_sequence_invalid(self):
        """Test get_sequence raises error for invalid chain."""
        seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        with pytest.raises(ValueError):
            seq.get_sequence("invalid")


class TestAntibodySequenceEquality:
    """Test AntibodySequence equality and hashing."""

    def test_equality(self):
        """Test that identical sequences are equal."""
        seq1 = AntibodySequence(heavy="EVQLVESGGGLVQ")
        seq2 = AntibodySequence(heavy="EVQLVESGGGLVQ")
        assert seq1 == seq2

    def test_inequality_different_sequence(self):
        """Test that different sequences are not equal."""
        seq1 = AntibodySequence(heavy="EVQLVESGGGLVQ")
        seq2 = AntibodySequence(heavy="QVQLVESGGGLVQ")
        assert seq1 != seq2

    def test_inequality_different_chain(self):
        """Test that same sequence in different chain is not equal."""
        seq1 = AntibodySequence(heavy="EVQLVESGGGLVQ")
        seq2 = AntibodySequence(light="EVQLVESGGGLVQ")
        assert seq1 != seq2

    def test_hashable(self):
        """Test that sequences can be used in sets."""
        seq1 = AntibodySequence(heavy="EVQLVESGGGLVQ")
        seq2 = AntibodySequence(heavy="EVQLVESGGGLVQ")
        seq3 = AntibodySequence(heavy="QVQLVESGGGLVQ")

        seq_set = {seq1, seq2, seq3}
        assert len(seq_set) == 2


class TestSpeciesEnum:
    """Test Species enumeration."""

    def test_species_values(self):
        """Test that all expected species are available."""
        assert Species.HUMAN.value == "human"
        assert Species.MOUSE.value == "mouse"
        assert Species.CAMEL.value == "camel"
        assert Species.UNKNOWN.value == "unknown"


class TestChainTypeEnum:
    """Test ChainType enumeration."""

    def test_chain_type_values(self):
        """Test that all expected chain types are available."""
        assert ChainType.HEAVY.value == "heavy"
        assert ChainType.LIGHT.value == "light"
        assert ChainType.UNKNOWN.value == "unknown"
