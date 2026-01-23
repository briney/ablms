"""Base class for encoder antibody language models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from ablms.core.base import BaseAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import UnsupportedOperationError
from ablms.outputs import AttentionOutput, EmbeddingOutput, LogitsOutput, MaskScanOutput
from ablms.utils.pooling import apply_pooling


class EncoderAbLM(BaseAbLM):
    """
    Base class for encoder-based antibody language models.

    Encoder models process sequences bidirectionally and can produce
    embeddings, attention weights, and masked language model predictions.

    This class defines the unified API that all encoder implementations
    must follow. Subclasses should override the abstract methods and
    may override other methods for model-specific behavior.
    """

    # Whether the model has a masked language modeling head
    has_mlm_head: bool = True

    def get_embeddings(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        layer: int = -1,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingOutput:
        """
        Get token-level embeddings for sequences.

        Args:
            sequences: Input sequences in various formats.
            layer: Layer index to extract embeddings from (-1 for last layer).
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            EmbeddingOutput containing token-level embeddings with shape
            [batch, seq_len, hidden_dim].
        """
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return EmbeddingOutput(
                embeddings=torch.empty(0, 0, self.embedding_dim),
                attention_mask=None,
                token_offsets=[],
                sequences=[],
                layer=layer,
            )

        executor = self._get_executor()
        all_embeddings, all_masks, all_offsets = executor.execute(
            method_name="_process_embeddings_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing embeddings",
            layer=layer,
        )

        return EmbeddingOutput(
            embeddings=all_embeddings,
            attention_mask=all_masks,
            token_offsets=all_offsets,
            sequences=sequences,
            layer=layer,
        )

    def _process_embeddings_batch(
        self,
        sequences: list[AntibodySequence],
        layer: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """
        Process a single batch of sequences for embeddings.

        This method is called by workers and should NOT be parallelized further.

        Args:
            sequences: Batch of sequences (already batched by executor).
            layer: Layer index to extract embeddings from.

        Returns:
            Tuple of (embeddings, attention_mask, token_offsets).
        """
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize(formatted)
        offsets = self._compute_token_offsets(sequences, tokenized)
        embeddings, mask = self._forward_embeddings(tokenized, layer)

        # Move results to CPU for cross-process transfer
        embeddings = embeddings.cpu()
        if mask is not None:
            mask = mask.cpu()

        return embeddings, mask, offsets

    def get_sequence_embeddings(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        pooling: str = "mean",
        layer: int = -1,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingOutput:
        """
        Get sequence-level embeddings using pooling.

        Args:
            sequences: Input sequences in various formats.
            pooling: Pooling strategy ("mean", "max", "cls", "first", "last").
            layer: Layer index to extract embeddings from.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            EmbeddingOutput containing pooled embeddings with shape
            [batch, hidden_dim].
        """
        # Get token-level embeddings first
        token_output = self.get_embeddings(
            sequences, layer=layer, batch_size=batch_size, show_progress=show_progress
        )

        # Apply pooling
        pooled = apply_pooling(
            token_output.embeddings,
            strategy=pooling,
            attention_mask=token_output.attention_mask,
        )

        return EmbeddingOutput(
            embeddings=pooled,
            attention_mask=None,
            token_offsets=token_output.token_offsets,
            pooled=pooled,
            sequences=token_output.sequences,
            layer=layer,
        )

    def get_hidden_states(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[EmbeddingOutput]:
        """
        Get embeddings from all layers.

        Args:
            sequences: Input sequences in various formats.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            List of EmbeddingOutput objects, one per layer.
        """
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return []

        executor = self._get_executor()
        all_hidden_states, all_masks, all_offsets = executor.execute(
            method_name="_process_hidden_states_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing hidden states",
        )

        # Create EmbeddingOutput for each layer
        outputs = []
        for layer_idx, layer_embeddings in enumerate(all_hidden_states):
            outputs.append(
                EmbeddingOutput(
                    embeddings=layer_embeddings,
                    attention_mask=all_masks,
                    token_offsets=all_offsets,
                    sequences=sequences,
                    layer=layer_idx,
                )
            )

        return outputs

    def _process_hidden_states_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """
        Process a single batch for hidden states from all layers.

        Args:
            sequences: Batch of sequences.

        Returns:
            Tuple of (list of hidden states per layer, attention_mask, token_offsets).
        """
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize(formatted)
        offsets = self._compute_token_offsets(sequences, tokenized)
        hidden_states, mask = self._forward_all_hidden_states(tokenized)

        # Move to CPU
        hidden_states = [h.cpu() for h in hidden_states]
        if mask is not None:
            mask = mask.cpu()

        return hidden_states, mask, offsets

    def get_attention(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> AttentionOutput:
        """
        Get attention weights for sequences.

        Args:
            sequences: Input sequences in various formats.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            AttentionOutput containing attention weights with shape
            [batch, layers, heads, seq_len, seq_len].
        """
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return AttentionOutput(
                attention_weights=torch.empty(0, 0, 0, 0, 0),
                attention_mask=None,
                token_offsets=[],
                sequences=[],
            )

        executor = self._get_executor()
        all_attention, all_masks, all_offsets = executor.execute(
            method_name="_process_attention_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing attention",
        )

        return AttentionOutput(
            attention_weights=all_attention,
            attention_mask=all_masks,
            token_offsets=all_offsets,
            sequences=sequences,
        )

    def _process_attention_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """
        Process a single batch for attention weights.

        Args:
            sequences: Batch of sequences.

        Returns:
            Tuple of (attention_weights, attention_mask, token_offsets).
        """
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize(formatted)
        offsets = self._compute_token_offsets(sequences, tokenized)
        attention, mask = self._forward_attention(tokenized)

        # Move to CPU
        attention = attention.cpu()
        if mask is not None:
            mask = mask.cpu()

        return attention, mask, offsets

    def get_logits(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> LogitsOutput:
        """
        Get masked language model logits for sequences.

        Args:
            sequences: Input sequences in various formats.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            LogitsOutput containing logits with shape [batch, seq_len, vocab_size].

        Raises:
            UnsupportedOperationError: If the model doesn't have an MLM head.
        """
        if not self.has_mlm_head:
            raise UnsupportedOperationError(
                f"Model '{self.model_name}' does not have a masked language "
                "modeling head and cannot produce logits."
            )

        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return LogitsOutput(
                logits=torch.empty(0, 0, 0),
                attention_mask=None,
                token_offsets=[],
                vocab=self._get_vocab(),
                sequences=[],
            )

        executor = self._get_executor()
        all_logits, all_masks, all_offsets = executor.execute(
            method_name="_process_logits_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing logits",
        )

        return LogitsOutput(
            logits=all_logits,
            attention_mask=all_masks,
            token_offsets=all_offsets,
            vocab=self._get_vocab(),
            sequences=sequences,
        )

    def _process_logits_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """
        Process a single batch for MLM logits.

        Args:
            sequences: Batch of sequences.

        Returns:
            Tuple of (logits, attention_mask, token_offsets).
        """
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize(formatted)
        offsets = self._compute_token_offsets(sequences, tokenized)
        logits, mask = self._forward_logits(tokenized)

        # Move to CPU
        logits = logits.cpu()
        if mask is not None:
            mask = mask.cpu()

        return logits, mask, offsets

    def pseudo_log_likelihood(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[float]:
        """
        Compute pseudo log-likelihood scores for sequences.

        Uses masked language model predictions to compute a pseudo
        log-likelihood by masking each position and summing log probs.

        Args:
            sequences: Input sequences in various formats.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            List of pseudo log-likelihood scores for each sequence.

        Raises:
            UnsupportedOperationError: If the model doesn't have an MLM head.
        """
        if not self.has_mlm_head:
            raise UnsupportedOperationError(
                f"Model '{self.model_name}' does not have a masked language "
                "modeling head and cannot compute pseudo log-likelihood."
            )

        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return []

        executor = self._get_executor()
        scores = executor.execute(
            method_name="_process_pseudo_ll_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing pseudo log-likelihood",
        )

        return scores

    def _process_pseudo_ll_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> list[float]:
        """
        Process a batch of sequences for pseudo log-likelihood.

        Args:
            sequences: Batch of sequences.

        Returns:
            List of pseudo log-likelihood scores.
        """
        scores = []
        for seq in sequences:
            score = self._compute_pseudo_ll(seq)
            scores.append(score)
        return scores

    def fill_mask(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        top_k: int = 1,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[AntibodySequence]]:
        """
        Fill masked positions in sequences.

        Args:
            sequences: Input sequences with mask tokens.
            top_k: Number of top predictions to return per sequence.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            List of lists of AntibodySequence objects. Each inner list
            contains top_k predictions for the corresponding input sequence.

        Raises:
            UnsupportedOperationError: If the model doesn't have an MLM head.
        """
        if not self.has_mlm_head:
            raise UnsupportedOperationError(
                f"Model '{self.model_name}' does not have a masked language "
                "modeling head and cannot fill masks."
            )

        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return []

        executor = self._get_executor()
        results = executor.execute(
            method_name="_process_fill_mask_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Filling masks",
            top_k=top_k,
        )

        return results

    def _process_fill_mask_batch(
        self,
        sequences: list[AntibodySequence],
        top_k: int = 1,
    ) -> list[list[AntibodySequence]]:
        """
        Process a batch for mask filling.

        Args:
            sequences: Batch of sequences with masks.
            top_k: Number of top predictions per sequence.

        Returns:
            List of lists of predicted sequences.
        """
        return self._fill_mask_batch(sequences, top_k)

    def mask_scan(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[MaskScanOutput]:
        """
        Scan each position by masking it and collecting model predictions.

        For each position in the sequence, masks that position and performs
        a forward pass to get the model's prediction distribution. Useful for
        computing per-position metrics like accuracy, perplexity, and entropy.

        Args:
            sequences: Input sequences (strings or AntibodySequence objects).
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            List of MaskScanOutput objects, one per input sequence.

        Raises:
            UnsupportedOperationError: If model has no MLM head.
        """
        if not self.has_mlm_head:
            raise UnsupportedOperationError(
                f"Model '{self.model_name}' does not have a masked language "
                "modeling head and cannot perform mask scanning."
            )

        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return []

        executor = self._get_executor()
        results = executor.execute(
            method_name="_process_mask_scan_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Scanning masks",
        )

        return results

    def _process_mask_scan_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> list[MaskScanOutput]:
        """
        Process a batch of sequences for mask scanning.

        Args:
            sequences: Batch of sequences.

        Returns:
            List of MaskScanOutput objects.
        """
        return self._mask_scan_batch(sequences)

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _forward_embeddings(
        self,
        tokenized: dict[str, torch.Tensor],
        layer: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass to get embeddings from a specific layer.

        Args:
            tokenized: Tokenized input tensors.
            layer: Layer index to extract embeddings from.

        Returns:
            Tuple of (embeddings, attention_mask).
        """
        pass

    @abstractmethod
    def _forward_all_hidden_states(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """
        Forward pass to get hidden states from all layers.

        Args:
            tokenized: Tokenized input tensors.

        Returns:
            Tuple of (list of hidden states, attention_mask).
        """
        pass

    @abstractmethod
    def _forward_attention(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass to get attention weights.

        Args:
            tokenized: Tokenized input tensors.

        Returns:
            Tuple of (attention weights [batch, layers, heads, seq, seq], attention_mask).
        """
        pass

    @abstractmethod
    def _forward_logits(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass to get MLM logits.

        Args:
            tokenized: Tokenized input tensors.

        Returns:
            Tuple of (logits, attention_mask).
        """
        pass

    @abstractmethod
    def _get_vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping."""
        pass

    @abstractmethod
    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """Compute pseudo log-likelihood for a single sequence."""
        pass

    @abstractmethod
    def _fill_mask_batch(
        self,
        sequences: list[AntibodySequence],
        top_k: int,
    ) -> list[list[AntibodySequence]]:
        """Fill masks for a batch of sequences."""
        pass

    @abstractmethod
    def _mask_scan_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> list[MaskScanOutput]:
        """
        Scan each position by masking it and collecting predictions.

        Model-specific implementation for mask scanning.

        Args:
            sequences: Batch of sequences.

        Returns:
            List of MaskScanOutput objects.
        """
        pass
