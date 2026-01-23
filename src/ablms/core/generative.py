"""Base class for generative antibody language models."""

from __future__ import annotations

from abc import abstractmethod

import torch

from ablms.core.base import BaseAbLM
from ablms.core.sequence import AntibodySequence, ChainType, Species
from ablms.outputs import GenerationOutput


class GenerativeAbLM(BaseAbLM):
    """
    Base class for generative (autoregressive) antibody language models.

    Generative models can produce new antibody sequences, perform infilling,
    and compute sequence likelihoods.

    This class defines the unified API that all generative implementations
    must follow. Subclasses should override the abstract methods.
    """

    def generate(
        self,
        num_sequences: int = 1,
        chain_type: ChainType = ChainType.HEAVY,
        species: Species = Species.HUMAN,
        prompt: str | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate new antibody sequences.

        Args:
            num_sequences: Number of sequences to generate.
            chain_type: Type of chain to generate (HEAVY or LIGHT).
            species: Species for the generated sequence.
            prompt: Optional prompt to condition generation.
            temperature: Sampling temperature (higher = more diverse).
            top_k: If set, sample from top-k tokens only.
            top_p: If set, sample using nucleus sampling.
            max_length: Maximum sequence length to generate.
            **kwargs: Additional model-specific parameters.

        Returns:
            GenerationOutput containing generated sequences and scores.
        """
        generation_params = {
            "chain_type": chain_type.value,
            "species": species.value,
            "prompt": prompt,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_length": max_length,
            **kwargs,
        }

        sequences, scores = self._generate(
            num_sequences=num_sequences,
            chain_type=chain_type,
            species=species,
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            **kwargs,
        )

        return GenerationOutput(
            sequences=sequences,
            scores=scores,
            generation_params=generation_params,
        )

    def infill(
        self,
        sequence: str | AntibodySequence,
        mask_range: tuple[int, int] | None = None,
        num_sequences: int = 1,
        chain_type: ChainType = ChainType.HEAVY,
        species: Species = Species.HUMAN,
        temperature: float = 1.0,
        **kwargs,
    ) -> GenerationOutput:
        """
        Infill masked regions in a sequence.

        Args:
            sequence: Input sequence with mask tokens or region to infill.
            mask_range: Optional (start, end) tuple specifying region to infill.
                If not provided, uses mask tokens in the sequence.
            num_sequences: Number of infilled sequences to generate.
            chain_type: Type of chain.
            species: Species for the sequence.
            temperature: Sampling temperature.
            **kwargs: Additional model-specific parameters.

        Returns:
            GenerationOutput containing infilled sequences.
        """
        # Normalize input
        if isinstance(sequence, str):
            if chain_type == ChainType.HEAVY:
                sequence = AntibodySequence(heavy=sequence, species=species)
            else:
                sequence = AntibodySequence(light=sequence, species=species)

        generation_params = {
            "chain_type": chain_type.value,
            "species": species.value,
            "mask_range": mask_range,
            "temperature": temperature,
            **kwargs,
        }

        sequences, scores = self._infill(
            sequence=sequence,
            mask_range=mask_range,
            num_sequences=num_sequences,
            chain_type=chain_type,
            species=species,
            temperature=temperature,
            **kwargs,
        )

        return GenerationOutput(
            sequences=sequences,
            scores=scores,
            generation_params=generation_params,
        )

    def log_likelihood(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        chain_type: ChainType = ChainType.HEAVY,
        species: Species = Species.HUMAN,
    ) -> list[float]:
        """
        Compute log-likelihood scores for sequences.

        Args:
            sequences: Input sequences to score.
            chain_type: Type of chain.
            species: Species for the sequences.

        Returns:
            List of log-likelihood scores for each sequence.
        """
        sequences = self._normalize_input(sequences)

        scores = []
        for seq in sequences:
            score = self._compute_log_likelihood(seq, chain_type, species)
            scores.append(score)

        return scores

    # Abstract methods that subclasses must implement

    @abstractmethod
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
        """
        Internal generation method.

        Args:
            num_sequences: Number of sequences to generate.
            chain_type: Type of chain to generate.
            species: Species for generation.
            prompt: Optional conditioning prompt.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            max_length: Maximum sequence length.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (list of sequences, list of scores).
        """
        pass

    @abstractmethod
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
        """
        Internal infill method.

        Args:
            sequence: Sequence to infill.
            mask_range: Region to infill.
            num_sequences: Number of infilled sequences.
            chain_type: Chain type.
            species: Species.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (list of infilled sequences, list of scores).
        """
        pass

    @abstractmethod
    def _compute_log_likelihood(
        self,
        sequence: AntibodySequence,
        chain_type: ChainType,
        species: Species,
    ) -> float:
        """
        Compute log-likelihood for a single sequence.

        Args:
            sequence: Sequence to score.
            chain_type: Chain type.
            species: Species.

        Returns:
            Log-likelihood score.
        """
        pass
