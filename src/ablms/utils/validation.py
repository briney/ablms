"""Input validation utilities."""


from ablms.exceptions import InvalidAminoAcidError, SequenceTooLongError


# Valid amino acid characters:
# - Standard 20 amino acids: ACDEFGHIKLMNPQRSTVWY
# - X: ambiguous/unknown residue
# - *: stop codon
# - -: gap
VALID_AMINO_ACIDS: set[str] = set("ACDEFGHIKLMNPQRSTVWXY*-")


def validate_amino_acids(
    sequence: str,
    mask_token: str = "<MASK>",
    context: str = "sequence",
) -> None:
    """
    Validate that a sequence contains only valid amino acid characters.

    Args:
        sequence: Amino acid sequence to validate.
        mask_token: Mask token to exclude from validation.
        context: Context string for error messages.

    Raises:
        InvalidAminoAcidError: If invalid characters are found.
    """
    # Remove mask tokens for validation
    seq_without_masks = sequence.replace(mask_token, "")

    invalid_chars = set(seq_without_masks) - VALID_AMINO_ACIDS
    if invalid_chars:
        raise InvalidAminoAcidError(
            f"Invalid amino acid(s) in {context}: {sorted(invalid_chars)}"
        )


def validate_sequence_length(
    sequence: str,
    max_length: int,
    mask_token: str = "<MASK>",
    context: str = "sequence",
) -> None:
    """
    Validate that a sequence does not exceed the maximum length.

    Args:
        sequence: Sequence to validate.
        max_length: Maximum allowed length.
        mask_token: Mask token (counts as 1 character).
        context: Context string for error messages.

    Raises:
        SequenceTooLongError: If sequence exceeds maximum length.
    """
    # Count mask tokens as single positions
    seq_len = len(sequence.replace(mask_token, "X"))

    if seq_len > max_length:
        raise SequenceTooLongError(
            f"{context} length ({seq_len}) exceeds maximum ({max_length})"
        )
