"""IgLM generative model wrapper."""

from __future__ import annotations

import torch

from ablms.core.generative import GenerativeAbLM
from ablms.core.sequence import AntibodySequence, ChainType, Species
from ablms.exceptions import ModelLoadError, UnsupportedOperationError

# Mapping from our Species enum to IgLM species control tokens. These are fed
# to IgLM's tokenizer, which asserts that every token is in its vocabulary.
SPECIES_MAP = {
    Species.HUMAN: "[HUMAN]",
    Species.MOUSE: "[MOUSE]",
    Species.CAMEL: "[CAMEL]",
    Species.RAT: "[RAT]",
    Species.RABBIT: "[RABBIT]",
    Species.RHESUS: "[RHESUS]",
    Species.UNKNOWN: "[HUMAN]",  # Default to human
}

# Mapping from our ChainType enum to IgLM chain control tokens.
CHAIN_TYPE_MAP = {
    ChainType.HEAVY: "[HEAVY]",
    ChainType.LIGHT: "[LIGHT]",
    ChainType.UNKNOWN: "[HEAVY]",  # Default to heavy
}


class IgLM(GenerativeAbLM):
    """
    IgLM generative model for antibody sequences.

    IgLM is a GPT-2 based model for antibody sequence generation.
    It can generate full sequences, infill masked regions, and
    compute sequence likelihoods.

    Package: iglm
    Paper: https://www.biorxiv.org/content/10.1101/2022.12.20.521029

    Attributes:
        model_name: "iglm"
        supports_paired: False
        max_length: 512
    """

    model_name = "iglm"
    supports_paired = False
    max_length = 512
    mask_token = None
    separator = None
    has_mlm_head = False

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize IgLM model.

        Args:
            device: (DEPRECATED) Device for inference. Use 'devices' instead.
            devices: Device(s) for inference. Auto-selects all GPUs if None.
        """
        super().__init__(device=device, devices=devices)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from the iglm package."""
        try:
            from iglm import IgLM as IgLMModel
        except ImportError as e:
            raise ModelLoadError(
                "Failed to import iglm package. " "Install it with: pip install iglm"
            ) from e

        self._iglm = IgLMModel()
        # Move model to device if possible
        if hasattr(self._iglm, "model"):
            self._iglm.model = self._iglm.model.to(self._primary_device)
        self._model = self._iglm

    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """Format sequences for IgLM (returns raw sequences)."""
        return [seq.primary_chain for seq in sequences]

    def _tokenize(self, formatted_sequences: list[str]) -> dict[str, list[str]]:
        """Tokenization is handled internally by IgLM."""
        return {"sequences": formatted_sequences}

    def _generate(
        self,
        num_sequences: int,
        chain_type: ChainType,
        species: Species,
        prompt: str | None,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        max_length: int | None,
        **kwargs,
    ) -> tuple[list[AntibodySequence], list[float]]:
        """Generate new antibody sequences using IgLM."""
        # Map enums to IgLM parameters
        iglm_chain = CHAIN_TYPE_MAP.get(chain_type, "[HEAVY]")
        iglm_species = SPECIES_MAP.get(species, "[HUMAN]")

        if top_k is not None:
            raise UnsupportedOperationError(
                "IgLM does not support top_k sampling; use top_p instead."
            )
        if max_length is not None:
            raise UnsupportedOperationError(
                "IgLM does not support a max_length argument."
            )

        # IgLM returns a de-duplicated list of sequences and no scores, so ask
        # for all of them in one call and score them separately.
        generated_seqs = self._iglm.generate(
            chain_token=iglm_chain,
            species_token=iglm_species,
            prompt_sequence=prompt,
            num_to_generate=num_sequences,
            top_p=top_p if top_p is not None else 1.0,
            temperature=temperature,
            **kwargs,
        )

        sequences = []
        scores = []
        for generated_seq in generated_seqs:
            if chain_type == ChainType.LIGHT:
                ab_seq = AntibodySequence(light=generated_seq, species=species)
            else:
                ab_seq = AntibodySequence(heavy=generated_seq, species=species)
            sequences.append(ab_seq)
            scores.append(
                self._iglm.log_likelihood(
                    sequence=generated_seq,
                    chain_token=iglm_chain,
                    species_token=iglm_species,
                )
            )

        return sequences, scores

    def _infill(
        self,
        sequence: AntibodySequence,
        mask_range: tuple[int, int] | None,
        num_sequences: int,
        chain_type: ChainType,
        species: Species,
        temperature: float,
        **kwargs,
    ) -> tuple[list[AntibodySequence], list[float]]:
        """Infill masked regions in a sequence."""
        # Get the sequence string
        seq_str = sequence.primary_chain

        # Map enums
        iglm_chain = CHAIN_TYPE_MAP.get(chain_type, "[HEAVY]")
        iglm_species = SPECIES_MAP.get(species, "[HUMAN]")

        sequences = []
        scores = []

        if mask_range is not None:
            # Use IgLM's infill functionality
            start, end = mask_range

            infilled_seqs = self._iglm.infill(
                sequence=seq_str,
                chain_token=iglm_chain,
                species_token=iglm_species,
                infill_range=(start, end),
                temperature=temperature,
                num_to_generate=num_sequences,
                **kwargs,
            )

            for infilled_seq in infilled_seqs:
                if chain_type == ChainType.LIGHT:
                    ab_seq = AntibodySequence(light=infilled_seq, species=species)
                else:
                    ab_seq = AntibodySequence(heavy=infilled_seq, species=species)
                sequences.append(ab_seq)
                scores.append(
                    self._iglm.log_likelihood(
                        sequence=infilled_seq,
                        chain_token=iglm_chain,
                        species_token=iglm_species,
                    )
                )

        elif sequence.is_masked:
            # Find mask positions and use IgLM infill
            mask_token = AntibodySequence.MASK_TOKEN
            masked_positions = sequence.masked_positions

            chain = "heavy" if sequence.heavy_chain else "light"
            positions = masked_positions.get(chain, [])

            if positions:
                # Find contiguous mask region
                start = positions[0]
                end = positions[-1] + 1

                infilled_seqs = self._iglm.infill(
                    sequence=seq_str.replace(mask_token, ""),
                    chain_token=iglm_chain,
                    species_token=iglm_species,
                    infill_range=(start, end),
                    temperature=temperature,
                    num_to_generate=num_sequences,
                    **kwargs,
                )

                for infilled_seq in infilled_seqs:
                    if chain_type == ChainType.LIGHT:
                        ab_seq = AntibodySequence(light=infilled_seq, species=species)
                    else:
                        ab_seq = AntibodySequence(heavy=infilled_seq, species=species)
                    sequences.append(ab_seq)
                    scores.append(
                        self._iglm.log_likelihood(
                            sequence=infilled_seq,
                            chain_token=iglm_chain,
                            species_token=iglm_species,
                        )
                    )
            else:
                # No masks found, return original
                sequences = [sequence]
                scores = [0.0]
        else:
            # No mask range or mask tokens, return original
            sequences = [sequence]
            scores = [0.0]

        return sequences, scores

    def _compute_log_likelihood(
        self,
        sequence: AntibodySequence,
        chain_type: ChainType,
        species: Species,
    ) -> float:
        """Compute log-likelihood for a single sequence."""
        seq_str = sequence.primary_chain

        iglm_chain = CHAIN_TYPE_MAP.get(chain_type, "[HEAVY]")
        iglm_species = SPECIES_MAP.get(species, "[HUMAN]")

        # Use IgLM's log_likelihood method
        score = self._iglm.log_likelihood(
            sequence=seq_str,
            chain_token=iglm_chain,
            species_token=iglm_species,
        )

        return float(score) if score is not None else 0.0
