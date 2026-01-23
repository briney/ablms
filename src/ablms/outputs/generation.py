"""Generation output dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ablms.core.sequence import AntibodySequence, ChainType, Species


@dataclass
class GenerationOutput:
    """
    Output container for generated sequences.

    Attributes:
        sequences: List of generated AntibodySequence objects.
        scores: Log-likelihood scores for each generated sequence.
        generation_params: Dictionary of parameters used for generation.
    """

    sequences: List[AntibodySequence]
    scores: List[float] | None = None
    generation_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_sequences(self) -> int:
        """Return the number of generated sequences."""
        return len(self.sequences)

    def get_sequence(self, idx: int) -> AntibodySequence:
        """
        Get a specific generated sequence.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            The AntibodySequence at the specified index.
        """
        return self.sequences[idx]

    def get_score(self, idx: int) -> float | None:
        """
        Get the score for a specific sequence.

        Args:
            idx: Index of the sequence.

        Returns:
            The log-likelihood score, or None if scores are unavailable.
        """
        if self.scores is None:
            return None
        return self.scores[idx]

    def get_top_k(self, k: int = 5) -> GenerationOutput:
        """
        Get the top-k sequences by score.

        Args:
            k: Number of top sequences to return.

        Returns:
            New GenerationOutput with only the top-k sequences.

        Raises:
            ValueError: If scores are not available.
        """
        if self.scores is None:
            raise ValueError("Scores are not available for ranking")

        # Sort by score descending
        sorted_indices = sorted(
            range(len(self.scores)), key=lambda i: self.scores[i], reverse=True
        )[:k]

        return GenerationOutput(
            sequences=[self.sequences[i] for i in sorted_indices],
            scores=[self.scores[i] for i in sorted_indices],
            generation_params=self.generation_params,
        )

    def filter_by_score(self, min_score: float) -> GenerationOutput:
        """
        Filter sequences by minimum score.

        Args:
            min_score: Minimum log-likelihood score threshold.

        Returns:
            New GenerationOutput with only sequences meeting the threshold.

        Raises:
            ValueError: If scores are not available.
        """
        if self.scores is None:
            raise ValueError("Scores are not available for filtering")

        filtered = [
            (seq, score)
            for seq, score in zip(self.sequences, self.scores)
            if score >= min_score
        ]

        if not filtered:
            return GenerationOutput(
                sequences=[],
                scores=[],
                generation_params=self.generation_params,
            )

        sequences, scores = zip(*filtered)
        return GenerationOutput(
            sequences=list(sequences),
            scores=list(scores),
            generation_params=self.generation_params,
        )

    def __iter__(self):
        """Iterate over sequences."""
        return iter(self.sequences)

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> AntibodySequence:
        """Get sequence by index."""
        return self.sequences[idx]

    def __repr__(self) -> str:
        """Return a string representation."""
        scores_str = ""
        if self.scores is not None and self.scores:
            min_score = min(self.scores)
            max_score = max(self.scores)
            scores_str = f", scores=[{min_score:.2f}, {max_score:.2f}]"
        return f"GenerationOutput(n={self.num_sequences}{scores_str})"
