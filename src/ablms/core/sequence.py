"""Antibody sequence representations and enums."""

from __future__ import annotations

import re
from enum import Enum

from ablms.exceptions import InvalidAminoAcidError, InvalidSequenceError


class ChainType(Enum):
    """Antibody chain type enumeration."""

    HEAVY = "heavy"
    LIGHT = "light"
    UNKNOWN = "unknown"


class Species(Enum):
    """Species enumeration for antibody sequences."""

    HUMAN = "human"
    MOUSE = "mouse"
    CAMEL = "camel"
    RAT = "rat"
    RABBIT = "rabbit"
    RHESUS = "rhesus"
    UNKNOWN = "unknown"


# Valid amino acid characters (standard 20 + X for unknown)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWXY")


class AntibodySequence:
    """
    Unified representation of antibody sequences.

    Supports single chain (heavy or light) and paired sequences.
    Uses a unified mask token (<MASK>) that gets converted to model-specific
    tokens during processing.

    All arguments must be passed as keyword arguments to ensure chain type
    is always explicit.

    Args:
        heavy: Heavy chain amino acid sequence, or None.
        light: Light chain amino acid sequence, or None.
        species: Species of origin for the antibody.
        name: Optional name/identifier for the sequence.

    Raises:
        InvalidSequenceError: If neither heavy nor light chain is provided.
        InvalidAminoAcidError: If sequences contain invalid characters.

    Examples:
        >>> seq = AntibodySequence(heavy="EVQLVESGGGLVQ")
        >>> seq = AntibodySequence(light="DIQMTQSPSSLSA")
        >>> seq = AntibodySequence(heavy="EVQLVESGGGLVQ", light="DIQMTQSPSSLSA")
        >>> seq = AntibodySequence(heavy="EVQLVESGGGLVQ", species=Species.HUMAN)
    """

    # Unified mask token used internally
    MASK_TOKEN: str = "<MASK>"

    __slots__ = ("heavy_chain", "light_chain", "species", "name")

    def __init__(
        self,
        *,  # Force keyword-only arguments
        heavy: str | None = None,
        light: str | None = None,
        species: Species = Species.UNKNOWN,
        name: str | None = None,
    ) -> None:
        """
        Initialize an AntibodySequence.

        Args:
            heavy: Heavy chain amino acid sequence.
            light: Light chain amino acid sequence.
            species: Species of origin for the antibody.
            name: Optional name/identifier for the sequence.
        """
        self.heavy_chain = heavy
        self.light_chain = light
        self.species = species
        self.name = name
        self._validate()

    def _validate(self) -> None:
        """Validate that sequences contain only valid amino acids and mask tokens."""
        if self.heavy_chain is None and self.light_chain is None:
            raise InvalidSequenceError(
                "At least one of heavy or light chain must be provided"
            )

        for chain_name, sequence in [
            ("heavy", self.heavy_chain),
            ("light", self.light_chain),
        ]:
            if sequence is not None:
                self._validate_sequence(sequence, chain_name)

    def _validate_sequence(self, sequence: str, chain_name: str) -> None:
        """Validate a single sequence."""
        if not sequence:
            raise InvalidSequenceError(f"{chain_name} cannot be empty string")

        # Remove mask tokens for validation
        seq_without_masks = sequence.replace(self.MASK_TOKEN, "")

        # Check for invalid characters
        invalid_chars = set(seq_without_masks) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise InvalidAminoAcidError(
                f"Invalid amino acid(s) in {chain_name}: {sorted(invalid_chars)}"
            )

    @property
    def is_paired(self) -> bool:
        """Check if this is a paired sequence (both chains present)."""
        return self.heavy_chain is not None and self.light_chain is not None

    @property
    def is_masked(self) -> bool:
        """Check if any chain contains mask tokens."""
        if self.heavy_chain and self.MASK_TOKEN in self.heavy_chain:
            return True
        if self.light_chain and self.MASK_TOKEN in self.light_chain:
            return True
        return False

    @property
    def masked_positions(self) -> dict[str, list[int]]:
        """
        Get positions of mask tokens in each chain.

        Returns:
            Dictionary mapping chain names to lists of mask positions.
        """
        positions = {}

        for chain_name, sequence in [
            ("heavy", self.heavy_chain),
            ("light", self.light_chain),
        ]:
            if sequence is not None:
                chain_positions = []
                # Find all occurrences of the mask token
                idx = 0
                pos = 0
                while idx < len(sequence):
                    if sequence[idx:].startswith(self.MASK_TOKEN):
                        chain_positions.append(pos)
                        idx += len(self.MASK_TOKEN)
                    else:
                        idx += 1
                    pos += 1
                if chain_positions:
                    positions[chain_name] = chain_positions

        return positions

    @property
    def length(self) -> dict[str, int]:
        """
        Get the length of each chain (mask tokens count as 1).

        Returns:
            Dictionary mapping chain names to their lengths.
        """
        lengths = {}

        for chain_name, sequence in [
            ("heavy", self.heavy_chain),
            ("light", self.light_chain),
        ]:
            if sequence is not None:
                # Count mask tokens as single positions
                seq_len = len(sequence.replace(self.MASK_TOKEN, "X"))
                lengths[chain_name] = seq_len

        return lengths

    @property
    def total_length(self) -> int:
        """Get total length across all chains."""
        return sum(self.length.values())

    def with_mask(
        self, chain: str, positions: list[int]
    ) -> AntibodySequence:
        """
        Create a new AntibodySequence with mask tokens at specified positions.

        Args:
            chain: Chain to mask ("heavy" or "light").
            positions: List of positions to mask (0-indexed).

        Returns:
            New AntibodySequence with masks inserted.

        Raises:
            ValueError: If chain is invalid or positions are out of range.
        """
        if chain not in ("heavy", "light"):
            raise ValueError(f"chain must be 'heavy' or 'light', got '{chain}'")

        sequence = self.heavy_chain if chain == "heavy" else self.light_chain
        if sequence is None:
            raise ValueError(f"Cannot mask {chain} chain: it is None")

        # Convert to list for easier manipulation
        seq_list = list(sequence)
        seq_len = len(seq_list)

        # Validate positions
        for pos in positions:
            if pos < 0 or pos >= seq_len:
                raise ValueError(
                    f"Position {pos} is out of range for sequence of length {seq_len}"
                )

        # Replace positions with mask tokens (in reverse order to preserve indices)
        for pos in sorted(positions, reverse=True):
            seq_list[pos] = self.MASK_TOKEN

        masked_sequence = "".join(seq_list)

        # Create new instance with masked sequence
        if chain == "heavy":
            return AntibodySequence(
                heavy=masked_sequence,
                light=self.light_chain,
                species=self.species,
                name=self.name,
            )
        else:
            return AntibodySequence(
                heavy=self.heavy_chain,
                light=masked_sequence,
                species=self.species,
                name=self.name,
            )

    def get_sequence(self, chain: str) -> str | None:
        """
        Get the sequence for a specific chain.

        Args:
            chain: Chain to get ("heavy" or "light").

        Returns:
            The sequence string or None if not present.
        """
        if chain == "heavy":
            return self.heavy_chain
        elif chain == "light":
            return self.light_chain
        else:
            raise ValueError(f"chain must be 'heavy' or 'light', got '{chain}'")

    def __repr__(self) -> str:
        """Return a string representation."""
        parts = []
        if self.name:
            parts.append(f"name='{self.name}'")
        if self.heavy_chain:
            h_len = self.length.get("heavy", 0)
            parts.append(f"heavy={h_len}aa")
        if self.light_chain:
            l_len = self.length.get("light", 0)
            parts.append(f"light={l_len}aa")
        if self.species != Species.UNKNOWN:
            parts.append(f"species={self.species.value}")
        return f"AntibodySequence({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another AntibodySequence."""
        if not isinstance(other, AntibodySequence):
            return NotImplemented
        return (
            self.heavy_chain == other.heavy_chain
            and self.light_chain == other.light_chain
            and self.species == other.species
        )

    def __hash__(self) -> int:
        """Return hash of the sequence."""
        return hash((self.heavy_chain, self.light_chain, self.species))
