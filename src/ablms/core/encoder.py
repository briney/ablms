"""Base class for encoder antibody language models."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator

import torch

from ablms.core.base import BaseAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import UnsupportedOperationError
from ablms.outputs import AttentionOutput, EmbeddingOutput, LogitsOutput, MaskScanOutput
from ablms.utils.layers import resolve_layer_selection
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

    # Whether the model can return layers other than its final one
    supports_intermediate_layers: bool = True

    @property
    def num_layers(self) -> int:
        """
        Number of transformer blocks in the loaded model.

        Selectable layer indices run from 0 to num_layers inclusive: index 0 is
        the embedding-layer output and index i is the output of block i. This
        matches HuggingFace, where len(hidden_states) == num_hidden_layers + 1.

        Subclasses must override this if their model object does not expose a
        HuggingFace config with `num_hidden_layers`.

        Returns:
            Count of transformer blocks.
        """
        return self._model.config.num_hidden_layers

    def get_embeddings(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        layer: int | list[int] | str = -1,
        pooling: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingOutput:
        """
        Get embeddings for sequences.

        Args:
            sequences: Input sequences in various formats.
            layer: Which layer(s) to extract. One of:
                - an int (default -1, the final layer). Index 0 is the
                  embedding layer and index i is the output of block i.
                - a list of ints, which adds a layer axis at dimension 1.
                - "all", for every layer in ascending order.
                A list of length one still adds the layer axis, so a
                programmatically built selection has a stable shape.
            pooling: Optional pooling strategy for sequence-level embeddings.
                If None (default), returns token-level embeddings. Pooling is
                applied within each batch on the model's device, and per layer
                before layers are stacked, so pooled runs never materialize the
                full token-level tensor.
                Valid options: "mean", "max", "cls", "first", "last".
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            EmbeddingOutput containing embeddings. Shape is
            [batch, seq_len, hidden_dim] for token-level (pooling=None) or
            [batch, hidden_dim] for sequence-level (pooling specified), with a
            layer axis inserted at dimension 1 when several layers are selected.

        Raises:
            ValueError: If the layer selection is malformed or out of range.
            UnsupportedOperationError: If a non-final layer is requested from a
                model that exposes only its final layer.

        Note:
            Token-level output for many layers is large: "all" on a 12-block
            model with hidden_dim 1024 is roughly 13x the single-layer payload.
            Use iter_embeddings() for anything that will not fit in memory.

        Example:
            >>> # Every layer, CLS-pooled, as one feature vector per sequence
            >>> out = model.get_embeddings(seqs, layer="all", pooling="cls")
            >>> features = out.concat_layers()  # [batch, n_layers * hidden_dim]
        """
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        selection = resolve_layer_selection(
            layer,
            self.num_layers,
            model_name=self.model_name,
            supports_intermediate_layers=self.supports_intermediate_layers,
        )
        layers = None if isinstance(selection, int) else selection
        single_layer = selection if isinstance(selection, int) else None

        if len(sequences) == 0:
            return self._empty_embedding_output(layers, single_layer, pooling)

        executor = self._get_executor()
        all_embeddings, all_masks, all_offsets = executor.execute(
            method_name="_process_embeddings_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing embeddings",
            layer=selection,
            pooling=pooling,
        )

        return EmbeddingOutput(
            embeddings=all_embeddings,
            # Pooling already reduced each batch, so there is no mask to carry.
            attention_mask=None if pooling is not None else all_masks,
            token_offsets=all_offsets,
            pooled=all_embeddings if pooling is not None else None,
            sequences=sequences,
            layer=single_layer,
            layers=layers,
        )

    def _empty_embedding_output(
        self,
        layers: list[int] | None,
        single_layer: int | None,
        pooling: str | None,
    ) -> EmbeddingOutput:
        """Build the zero-sequence result with the shape a real run would produce."""
        shape: tuple[int, ...]
        if pooling is None:
            shape = (0, 0, self.embedding_dim)
        else:
            shape = (0, self.embedding_dim)

        if layers is not None:
            shape = (0, len(layers), *shape[1:])

        return EmbeddingOutput(
            embeddings=torch.empty(*shape),
            attention_mask=None,
            token_offsets=[],
            sequences=[],
            layer=single_layer,
            layers=layers,
        )

    def _process_embeddings_batch(
        self,
        sequences: list[AntibodySequence],
        layer: int | list[int] = -1,
        pooling: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """
        Process a single batch of sequences for embeddings.

        This method is called by workers and should NOT be parallelized further.

        Pooling is applied on the model's device, before the result is moved to
        the host. This keeps the large [batch, seq_len, hidden_dim] tensor off
        the host entirely and out of the inter-process queue, which is what
        makes large multi-GPU runs viable.

        Args:
            sequences: Batch of sequences (already batched by executor).
            layer: A single layer index, or a list of resolved non-negative
                indices. A list adds a layer axis at dimension 1.
            pooling: Optional pooling strategy applied within this batch.
                One of "mean", "max", "cls", "first", "last", or None for
                token-level output.

        Returns:
            Tuple of (embeddings, attention_mask, token_offsets). The mask is
            None whenever pooling was applied. Embeddings are
            [batch, seq_len, hidden_dim], or [batch, hidden_dim] when pooled;
            a list `layer` inserts a layer axis at dimension 1.
        """
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize(formatted)
        offsets = self._compute_token_offsets(sequences, tokenized)

        if isinstance(layer, int):
            embeddings, mask = self._forward_embeddings(tokenized, layer)
            if pooling is not None:
                embeddings = apply_pooling(
                    embeddings, strategy=pooling, attention_mask=mask
                )
                mask = None
        else:
            embeddings, mask = self._forward_selected_layers(tokenized, layer, pooling)

        # Move results to CPU for cross-process transfer
        embeddings = embeddings.cpu()
        if mask is not None:
            mask = mask.cpu()

        return embeddings, mask, offsets

    def _forward_selected_layers(
        self,
        tokenized: dict[str, torch.Tensor],
        layers: list[int],
        pooling: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Stack several layers, pooling each one before it is stacked.

        Every encoder already implements `_forward_all_hidden_states`, and the
        HuggingFace-backed ones compute every layer regardless, so selecting
        from its output costs nothing over a single-layer forward pass.

        Pooling per layer rather than after stacking means the
        [batch, layers, seq_len, hidden_dim] tensor is never allocated on
        pooled runs - only [batch, layers, hidden_dim] survives to cross the
        result queue. See the "reduce before transfer" note in CLAUDE.md.

        Args:
            tokenized: Tokenized batch.
            layers: Resolved non-negative indices, in the order requested.
            pooling: Optional pooling strategy, applied to each layer.

        Returns:
            Tuple of (embeddings, attention_mask). Embeddings are
            [batch, len(layers), hidden_dim] when pooled, else
            [batch, len(layers), seq_len, hidden_dim]. The mask is None
            whenever pooling was applied.

        Raises:
            RuntimeError: If the model's reported num_layers disagrees with the
                number of hidden states its forward pass returned.
        """
        hidden_states, mask = self._forward_all_hidden_states(tokenized)

        expected = self.num_layers + 1
        if len(hidden_states) != expected:
            raise RuntimeError(
                f"{self.model_name} reports num_layers={self.num_layers} "
                f"({expected} selectable layers), but its forward pass returned "
                f"{len(hidden_states)} hidden states. The num_layers property "
                f"needs an override for this model."
            )

        if pooling is not None:
            pooled = [
                apply_pooling(hidden_states[i], strategy=pooling, attention_mask=mask)
                for i in layers
            ]
            return torch.stack(pooled, dim=1), None

        return torch.stack([hidden_states[i] for i in layers], dim=1), mask

    def iter_embeddings(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        layer: int | list[int] | str = -1,
        pooling: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> Iterator[EmbeddingOutput]:
        """
        Stream embeddings one batch at a time.

        Unlike get_embeddings(), nothing is accumulated: each batch is yielded
        as soon as it is ready and released once the caller is done with it.
        Use this when the full token-level output for the dataset would not fit
        in memory, writing each batch to HDF5, zarr, or npy as it arrives.

        Args:
            sequences: Input sequences in various formats.
            layer: Which layer(s) to extract. One of:
                - an int (default -1, the final layer). Index 0 is the
                  embedding layer and index i is the output of block i.
                - a list of ints, which adds a layer axis at dimension 1.
                - "all", for every layer in ascending order.
                A list of length one still adds the layer axis, so a
                programmatically built selection has a stable shape.
            pooling: Optional pooling strategy applied within each batch.
                Valid options: "mean", "max", "cls", "first", "last".
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Yields:
            One EmbeddingOutput per batch, in input order. Each carries its own
            slice of the input sequences and its own token offsets, so batches
            are self-describing.

        Raises:
            PairedSequenceError: If paired sequences are provided but the model
                does not support them.
            SequenceTooLongError: If a sequence exceeds the model's max length.
            ValueError: If the layer selection is malformed or out of range.
            UnsupportedOperationError: If a non-final layer is requested from a
                model that exposes only its final layer.

        Example:
            >>> for batch in model.iter_embeddings(sequences, pooling="mean"):
            ...     writer.append(batch.embeddings.numpy())
        """
        # Validate eagerly rather than on first next(), so bad input fails at
        # the call site.
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        selection = resolve_layer_selection(
            layer,
            self.num_layers,
            model_name=self.model_name,
            supports_intermediate_layers=self.supports_intermediate_layers,
        )

        return self._iter_embeddings(
            sequences=sequences,
            layer=selection,
            pooling=pooling,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def _iter_embeddings(
        self,
        sequences: list[AntibodySequence],
        layer: int | list[int],
        pooling: str | None,
        batch_size: int,
        show_progress: bool,
    ) -> Iterator[EmbeddingOutput]:
        """Generator backing iter_embeddings(); assumes validated input."""
        if not sequences:
            return

        layers = None if isinstance(layer, int) else layer
        single_layer = layer if isinstance(layer, int) else None

        executor = self._get_executor()
        for batch_idx, (embeddings, mask, offsets) in executor.execute_iter(
            method_name="_process_embeddings_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing embeddings",
            layer=layer,
            pooling=pooling,
        ):
            start = batch_idx * batch_size
            yield EmbeddingOutput(
                embeddings=embeddings,
                attention_mask=mask,
                token_offsets=offsets,
                pooled=embeddings if pooling is not None else None,
                sequences=sequences[start : start + batch_size],
                layer=single_layer,
                layers=layers,
            )

    def get_hidden_states(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[EmbeddingOutput]:
        """
        Get embeddings from all layers.

        A thin wrapper over `get_embeddings(layer="all")`, kept for backwards
        compatibility. Prefer `layer="all"` directly: it returns one output with
        a layer axis, supports pooling, and streams through `iter_embeddings()`.
        This method materializes the full token-level output for every layer.

        Args:
            sequences: Input sequences in various formats.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            List of EmbeddingOutput objects, one per layer, in ascending layer
            order. Models that expose only their final layer return a
            single-element list.
        """
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return []

        # A final-layer-only model (AbLang) would raise on "all".
        selection = "all" if self.supports_intermediate_layers else -1
        stacked = self.get_embeddings(
            sequences,
            layer=selection,
            pooling=None,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        if not stacked.is_multi_layer:
            return [stacked]

        return [
            EmbeddingOutput(
                embeddings=stacked.embeddings[:, position],
                attention_mask=stacked.attention_mask,
                token_offsets=stacked.token_offsets,
                sequences=stacked.sequences,
                layer=layer_index,
            )
            for position, layer_index in enumerate(stacked.layers)
        ]

    def _process_hidden_states_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> tuple[
        list[torch.Tensor], torch.Tensor | None, list[dict[str, tuple[int, int]]]
    ]:
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
            batch_size: Number of masked variants to process in a single forward
                pass. For a sequence of length L, L masked variants are created
                (one per position). These variants are batched together using
                this batch_size for efficient GPU utilization.
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
        # For mask_scan, each sequence is processed individually but the batch_size
        # controls how many masked variants are batched together for each sequence.
        # We set executor batch_size=1 so each sequence is processed one at a time,
        # and pass the user's batch_size to control internal batching of masked variants.
        results = executor.execute(
            method_name="_process_mask_scan_batch",
            sequences=sequences,
            batch_size=1,  # Process one sequence at a time
            show_progress=show_progress,
            progress_desc="Scanning masks",
            variants_batch_size=batch_size,  # Batch size for masked variants
        )

        return results

    def _process_mask_scan_batch(
        self,
        sequences: list[AntibodySequence],
        variants_batch_size: int = 32,
    ) -> list[MaskScanOutput]:
        """
        Process a batch of sequences for mask scanning.

        Args:
            sequences: Batch of sequences (typically just one sequence at a time
                since mask_scan processes sequences individually).
            variants_batch_size: Number of masked variants to process in one
                forward pass for GPU efficiency.

        Returns:
            List of MaskScanOutput objects.
        """
        return self._mask_scan_batch(sequences, variants_batch_size)

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
        batch_size: int = 32,
    ) -> list[MaskScanOutput]:
        """
        Scan each position by masking it and collecting predictions.

        Model-specific implementation for mask scanning. Creates masked variants
        of each input sequence (one variant per position) and batches them together
        for efficient GPU processing.

        Args:
            sequences: Input sequences to scan.
            batch_size: Number of masked variants to process in one forward pass.
                For a sequence of length L, there are L masked variants. This
                parameter controls how many of those variants are batched together.

        Returns:
            List of MaskScanOutput objects, one per input sequence.
        """
        pass
