"""IgLM generative model wrapper."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ablms.core.generative import GenerativeAbLM
from ablms.core.sequence import AntibodySequence, ChainType, Species
from ablms.exceptions import ModelLoadError


# Mapping from our Species enum to IgLM species names
SPECIES_MAP = {
    Species.HUMAN: "human",
    Species.MOUSE: "mouse",
    Species.CAMEL: "camel",
    Species.RAT: "rat",
    Species.RABBIT: "rabbit",
    Species.RHESUS: "rhesus",
    Species.UNKNOWN: "human",  # Default to human
}

# Mapping from our ChainType enum to IgLM chain names
CHAIN_TYPE_MAP = {
    ChainType.HEAVY: "heavy",
    ChainType.LIGHT: "light",
    ChainType.UNKNOWN: "heavy",  # Default to heavy
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
                "Failed to import iglm package. "
                "Install it with: pip install iglm"
            ) from e

        self._iglm = IgLMModel()
        # Move model to device if possible
        if hasattr(self._iglm, "model"):
            self._iglm.model = self._iglm.model.to(self._primary_device)
        self._model = self._iglm

    def _format_for_model(
        self, sequences: List[AntibodySequence]
    ) -> List[str]:
        """Format sequences for IgLM (returns raw sequences)."""
        formatted = []
        for seq in sequences:
            sequence = seq.heavy_chain or seq.light_chain
            formatted.append(sequence)
        return formatted

    def _tokenize(
        self, formatted_sequences: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize is handled internally by IgLM."""
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
    ) -> Tuple[List[AntibodySequence], List[float]]:
        """Generate new antibody sequences using IgLM."""
        # Map enums to IgLM parameters
        iglm_chain = CHAIN_TYPE_MAP.get(chain_type, "heavy")
        iglm_species = SPECIES_MAP.get(species, "human")

        sequences = []
        scores = []

        for _ in range(num_sequences):
            # Generate sequence using IgLM
            generated_seq, score = self._iglm.generate(
                chain_type=iglm_chain,
                species=iglm_species,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p if top_p is not None else 1.0,
                num_to_generate=1,
                **kwargs,
            )

            # Handle IgLM output format
            if isinstance(generated_seq, list):
                generated_seq = generated_seq[0]
            if isinstance(score, list):
                score = score[0] if score else 0.0

            # Create AntibodySequence
            if chain_type == ChainType.LIGHT:
                ab_seq = AntibodySequence(light=generated_seq, species=species)
            else:
                ab_seq = AntibodySequence(heavy=generated_seq, species=species)

            sequences.append(ab_seq)
            scores.append(float(score) if score is not None else 0.0)

        return sequences, scores

    def _infill(
        self,
        sequence: AntibodySequence,
        mask_range: Tuple[int, int] | None,
        num_sequences: int,
        chain_type: ChainType,
        species: Species,
        temperature: float,
        **kwargs,
    ) -> Tuple[List[AntibodySequence], List[float]]:
        """Infill masked regions in a sequence."""
        # Get the sequence string
        seq_str = sequence.heavy_chain or sequence.light_chain

        # Map enums
        iglm_chain = CHAIN_TYPE_MAP.get(chain_type, "heavy")
        iglm_species = SPECIES_MAP.get(species, "human")

        sequences = []
        scores = []

        if mask_range is not None:
            # Use IgLM's infill functionality
            start, end = mask_range
            prefix = seq_str[:start]
            suffix = seq_str[end:]

            for _ in range(num_sequences):
                infilled_seq, score = self._iglm.infill(
                    sequence=seq_str,
                    chain_type=iglm_chain,
                    species=iglm_species,
                    infill_range=(start, end),
                    temperature=temperature,
                    num_to_generate=1,
                    **kwargs,
                )

                if isinstance(infilled_seq, list):
                    infilled_seq = infilled_seq[0]
                if isinstance(score, list):
                    score = score[0] if score else 0.0

                if chain_type == ChainType.LIGHT:
                    ab_seq = AntibodySequence(light=infilled_seq, species=species)
                else:
                    ab_seq = AntibodySequence(heavy=infilled_seq, species=species)

                sequences.append(ab_seq)
                scores.append(float(score) if score is not None else 0.0)

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

                for _ in range(num_sequences):
                    infilled_seq, score = self._iglm.infill(
                        sequence=seq_str.replace(mask_token, ""),
                        chain_type=iglm_chain,
                        species=iglm_species,
                        infill_range=(start, end),
                        temperature=temperature,
                        num_to_generate=1,
                        **kwargs,
                    )

                    if isinstance(infilled_seq, list):
                        infilled_seq = infilled_seq[0]
                    if isinstance(score, list):
                        score = score[0] if score else 0.0

                    if chain_type == ChainType.LIGHT:
                        ab_seq = AntibodySequence(light=infilled_seq, species=species)
                    else:
                        ab_seq = AntibodySequence(heavy=infilled_seq, species=species)

                    sequences.append(ab_seq)
                    scores.append(float(score) if score is not None else 0.0)
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
        seq_str = sequence.heavy_chain or sequence.light_chain

        iglm_chain = CHAIN_TYPE_MAP.get(chain_type, "heavy")
        iglm_species = SPECIES_MAP.get(species, "human")

        # Use IgLM's log_likelihood method
        score = self._iglm.log_likelihood(
            sequence=seq_str,
            chain_type=iglm_chain,
            species=iglm_species,
        )

        return float(score) if score is not None else 0.0
