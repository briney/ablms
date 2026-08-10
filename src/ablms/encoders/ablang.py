"""AbLang (v1) encoder model wrapper."""

from __future__ import annotations


import torch
import torch.nn.functional as F

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import ModelLoadError
from ablms.outputs import MaskScanOutput
from ablms.utils.pooling import apply_pooling


class AbLang(EncoderAbLM):
    """
    AbLang (v1) encoder model for antibody sequences.

    AbLang v1 uses two separate models: one for heavy chains and one for
    light chains. This implementation auto-selects the appropriate model
    based on the input sequence type and supports mixed batches containing
    both heavy and light chain sequences.

    Package: ablang
    Paper: https://www.biorxiv.org/content/10.1101/2021.11.10.468064

    Attributes:
        model_name: "ablang"
        supports_paired: False (separate models for each chain type)
        max_length: 160
        embedding_dim: 768
        mask_token: "*"
    """

    model_name = "ablang"
    supports_paired = False
    max_length = 160
    embedding_dim = 768
    mask_token = "*"
    separator = None
    has_mlm_head = True
    supports_intermediate_layers = False

    @property
    def num_layers(self) -> int:
        """
        AbRep's depth, per the AbLang paper.

        Hardcoded because AbLang's model object exposes no config. Only the
        final layer is reachable (`_forward_embeddings_with_model` can return
        nothing else), so this value affects only which explicit positive index
        is accepted as "final" and the wording of the resulting error. Confirm
        it against a real forward pass if the `ablang` package is ever installed
        in the test environment.
        """
        return 12

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize AbLang model.

        Args:
            device: (DEPRECATED) Device for inference. Use 'devices' instead.
            devices: Device(s) for inference. Auto-selects all GPUs if None.
        """
        super().__init__(device=device, devices=devices)
        # Lazy-loaded models
        self._heavy_model = None
        self._light_model = None
        self._ablang_module = None
        self._load_model()

    def _load_model(self) -> None:
        """Verify ablang package is available; models are loaded lazily."""
        try:
            import ablang
            self._ablang_module = ablang
        except ImportError as e:
            raise ModelLoadError(
                "Failed to import ablang package. "
                "Install it with: pip install ablang"
            ) from e

    def _get_heavy_model(self):
        """Lazy-load and return the heavy chain model."""
        if self._heavy_model is None:
            self._heavy_model = self._ablang_module.pretrained("heavy")
            self._heavy_model.freeze()
            # Move to device
            if hasattr(self._heavy_model, "AbRep"):
                self._heavy_model.AbRep = self._heavy_model.AbRep.to(self._primary_device)
        return self._heavy_model

    def _get_light_model(self):
        """Lazy-load and return the light chain model."""
        if self._light_model is None:
            self._light_model = self._ablang_module.pretrained("light")
            self._light_model.freeze()
            # Move to device
            if hasattr(self._light_model, "AbRep"):
                self._light_model.AbRep = self._light_model.AbRep.to(self._primary_device)
        return self._light_model

    def _get_model_for_sequence(self, seq: AntibodySequence):
        """Get the appropriate model for a sequence."""
        if seq.heavy_chain is not None:
            return self._get_heavy_model()
        else:
            return self._get_light_model()

    def _is_heavy_chain(self, seq: AntibodySequence) -> bool:
        """Determine if sequence is a heavy chain."""
        return seq.heavy_chain is not None

    def _format_for_model(
        self, sequences: list[AntibodySequence]
    ) -> list[str]:
        """
        Format sequences for AbLang.

        AbLang uses "*" as the mask token and expects raw sequences.
        """
        formatted = []
        for seq in sequences:
            sequence = seq.heavy_chain or seq.light_chain

            # Convert unified mask token to AbLang mask token
            sequence = sequence.replace(AntibodySequence.MASK_TOKEN, self.mask_token)

            formatted.append(sequence)

        return formatted

    def _tokenize(
        self, formatted_sequences: list[str]
    ) -> dict[str, torch.Tensor]:
        """Tokenize formatted sequences using AbLang tokenizer.

        Note: This method requires a model to be selected first via
        _tokenize_with_model, as each model has its own tokenizer.
        """
        raise NotImplementedError(
            "AbLang requires model-specific tokenization. "
            "Use _tokenize_with_model instead."
        )

    def _tokenize_with_model(
        self, formatted_sequences: list[str], model
    ) -> dict[str, torch.Tensor]:
        """Tokenize formatted sequences using the given model's tokenizer."""
        # AbLang tokenizer returns tensor directly
        encoded = model.tokenizer(formatted_sequences, pad=True)

        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(self._primary_device)
            # Create attention mask based on padding
            attention_mask = (input_ids != model.tokenizer.pad_token).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        elif isinstance(encoded, dict):
            return {k: v.to(self._primary_device) for k, v in encoded.items()}
        else:
            return {"input_ids": encoded.to(self._primary_device)}

    def _compute_token_offsets(
        self,
        sequences: list[AntibodySequence],
        tokenized: dict[str, torch.Tensor],
    ) -> list[dict[str, tuple[int, int]]]:
        """Compute token offsets for each chain."""
        offsets = []

        for seq in sequences:
            seq_offsets = {}
            start = 1  # Skip [CLS]/start token

            if seq.heavy_chain is not None:
                seq_len = seq.length.get("heavy", 0)
                seq_offsets["heavy"] = (start, start + seq_len)
            elif seq.light_chain is not None:
                seq_len = seq.length.get("light", 0)
                seq_offsets["light"] = (start, start + seq_len)

            offsets.append(seq_offsets)

        return offsets

    # Override batch processing methods to handle mixed heavy/light batches

    def _process_embeddings_batch(
        self,
        sequences: list[AntibodySequence],
        layer: int = -1,
        pooling: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """
        Process a batch of sequences for embeddings, handling mixed chain types.

        The signature must stay bind-compatible with
        `EncoderAbLM._process_embeddings_batch`: the executor forwards
        `pooling` through `**method_kwargs` for every model.

        Pooling is applied to the merged heavy/light result, using the merged
        attention mask, so the reduction happens before the tensor is handed to
        the inter-process queue - which is what keeps the transfer small.

        Args:
            sequences: Batch of sequences (already batched by executor).
            layer: Layer index to extract embeddings from (ignored by AbLang,
                which only exposes its final layer).
            pooling: Optional pooling strategy applied within this batch.
                One of "mean", "max", "cls", "first", "last", or None for
                token-level output.

        Returns:
            Tuple of (embeddings, attention_mask, token_offsets). When pooling
            is applied, embeddings has shape [batch, hidden_dim] and the mask is
            None; otherwise embeddings has shape [batch, seq_len, hidden_dim].
        """
        embeddings, mask, offsets = self._process_mixed_batch(
            sequences,
            process_fn=lambda seqs, model: self._forward_embeddings_with_model(
                seqs, model, layer
            ),
        )

        if pooling is not None:
            embeddings = apply_pooling(
                embeddings, strategy=pooling, attention_mask=mask
            )
            mask = None

        return embeddings, mask, offsets

    def _process_hidden_states_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """Process a batch for hidden states from all layers."""
        return self._process_mixed_batch(
            sequences,
            process_fn=lambda seqs, model: self._forward_all_hidden_states_with_model(
                seqs, model
            ),
            is_hidden_states=True,
        )

    def _process_attention_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """Process a batch for attention weights."""
        return self._process_mixed_batch(
            sequences,
            process_fn=lambda seqs, model: self._forward_attention_with_model(
                seqs, model
            ),
        )

    def _process_logits_batch(
        self,
        sequences: list[AntibodySequence],
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """Process a batch for MLM logits."""
        return self._process_mixed_batch(
            sequences,
            process_fn=lambda seqs, model: self._forward_logits_with_model(
                seqs, model
            ),
        )

    def _process_mixed_batch(
        self,
        sequences: list[AntibodySequence],
        process_fn,
        is_hidden_states: bool = False,
    ):
        """
        Process a batch with potentially mixed heavy/light chains.

        Splits batch by chain type, processes each with appropriate model,
        then merges results maintaining original order.
        """
        # Partition sequences by chain type
        heavy_indices = []
        light_indices = []
        for i, seq in enumerate(sequences):
            if self._is_heavy_chain(seq):
                heavy_indices.append(i)
            else:
                light_indices.append(i)

        # Process each partition
        results = [None] * len(sequences)
        masks = [None] * len(sequences)
        offsets = [None] * len(sequences)

        if heavy_indices:
            heavy_seqs = [sequences[i] for i in heavy_indices]
            model = self._get_heavy_model()
            h_results, h_masks, h_offsets = process_fn(heavy_seqs, model)

            if is_hidden_states:
                # h_results is a list of tensors (one per layer)
                for batch_idx, orig_idx in enumerate(heavy_indices):
                    results[orig_idx] = [layer[batch_idx:batch_idx+1] for layer in h_results]
                    masks[orig_idx] = h_masks[batch_idx:batch_idx+1] if h_masks is not None else None
                    offsets[orig_idx] = h_offsets[batch_idx]
            else:
                for batch_idx, orig_idx in enumerate(heavy_indices):
                    results[orig_idx] = h_results[batch_idx:batch_idx+1]
                    masks[orig_idx] = h_masks[batch_idx:batch_idx+1] if h_masks is not None else None
                    offsets[orig_idx] = h_offsets[batch_idx]

        if light_indices:
            light_seqs = [sequences[i] for i in light_indices]
            model = self._get_light_model()
            l_results, l_masks, l_offsets = process_fn(light_seqs, model)

            if is_hidden_states:
                for batch_idx, orig_idx in enumerate(light_indices):
                    results[orig_idx] = [layer[batch_idx:batch_idx+1] for layer in l_results]
                    masks[orig_idx] = l_masks[batch_idx:batch_idx+1] if l_masks is not None else None
                    offsets[orig_idx] = l_offsets[batch_idx]
            else:
                for batch_idx, orig_idx in enumerate(light_indices):
                    results[orig_idx] = l_results[batch_idx:batch_idx+1]
                    masks[orig_idx] = l_masks[batch_idx:batch_idx+1] if l_masks is not None else None
                    offsets[orig_idx] = l_offsets[batch_idx]

        # Concatenate results
        if is_hidden_states:
            # results[i] is a list of single-sample tensors for each layer
            # We need to return a list where each element is all samples for that layer
            num_layers = len(results[0])
            combined_hidden_states = []
            for layer_idx in range(num_layers):
                layer_tensors = [results[i][layer_idx] for i in range(len(sequences))]
                combined_hidden_states.append(torch.cat(layer_tensors, dim=0).cpu())
            combined_masks = torch.cat([m for m in masks if m is not None], dim=0).cpu() if any(m is not None for m in masks) else None
            return combined_hidden_states, combined_masks, offsets
        else:
            combined_results = torch.cat(results, dim=0).cpu()
            combined_masks = torch.cat([m for m in masks if m is not None], dim=0).cpu() if any(m is not None for m in masks) else None
            return combined_results, combined_masks, offsets

    def _forward_embeddings_with_model(
        self,
        sequences: list[AntibodySequence],
        model,
        layer: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """Forward pass to get embeddings using a specific model."""
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize_with_model(formatted, model)
        offsets = self._compute_token_offsets(sequences, tokenized)

        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            # AbLang model forward pass through AbRep
            outputs = model.AbRep(input_ids)

        # AbLang only provides last_hidden_states (no intermediate layers available)
        # layer parameter is ignored as we can only access the final layer
        embeddings = outputs.last_hidden_states

        # Create attention mask
        attention_mask = (input_ids != model.tokenizer.pad_token).long()

        return embeddings, attention_mask, offsets

    def _forward_all_hidden_states_with_model(
        self,
        sequences: list[AntibodySequence],
        model,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """Forward pass to get all hidden states using a specific model."""
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize_with_model(formatted, model)
        offsets = self._compute_token_offsets(sequences, tokenized)

        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            outputs = model.AbRep(input_ids)

        # AbLang only provides last_hidden_states (no intermediate layers available)
        # Return as a single-element list to match expected interface
        hidden_states = [outputs.last_hidden_states]

        attention_mask = (input_ids != model.tokenizer.pad_token).long()

        return hidden_states, attention_mask, offsets

    def _forward_attention_with_model(
        self,
        sequences: list[AntibodySequence],
        model,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """Forward pass to get attention weights using a specific model."""
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize_with_model(formatted, model)
        offsets = self._compute_token_offsets(sequences, tokenized)

        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            outputs = model.AbRep(input_ids, output_attentions=True)

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            # AbLang returns attentions as list of [num_heads, batch, seq, seq]
            # Transpose to [batch, num_heads, seq, seq] then stack layers
            attentions_list = [a.permute(1, 0, 2, 3) for a in outputs.attentions]
            attentions = torch.stack(attentions_list, dim=1)
        else:
            batch_size, seq_len = input_ids.shape
            attentions = torch.zeros(batch_size, 1, 1, seq_len, seq_len, device=self._primary_device)

        attention_mask = (input_ids != model.tokenizer.pad_token).long()

        return attentions, attention_mask, offsets

    def _forward_logits_with_model(
        self,
        sequences: list[AntibodySequence],
        model,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """Forward pass to get MLM logits using a specific model."""
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize_with_model(formatted, model)
        offsets = self._compute_token_offsets(sequences, tokenized)

        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            # Use AbLang's MLM head - returns tensor directly
            logits = model.AbLang(input_ids)

        attention_mask = (input_ids != model.tokenizer.pad_token).long()

        return logits, attention_mask, offsets

    # Abstract method implementations (these are not directly called due to overrides)

    def _forward_embeddings(
        self,
        tokenized: dict[str, torch.Tensor],
        layer: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get embeddings - not directly used."""
        raise NotImplementedError("Use _process_embeddings_batch instead")

    def _forward_all_hidden_states(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Forward pass to get all hidden states - not directly used."""
        raise NotImplementedError("Use _process_hidden_states_batch instead")

    def _forward_attention(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get attention weights - not directly used."""
        raise NotImplementedError("Use _process_attention_batch instead")

    def _forward_logits(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get MLM logits - not directly used."""
        raise NotImplementedError("Use _process_logits_batch instead")

    def _get_vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping."""
        # Use heavy model's tokenizer by default (both should have same vocab)
        model = self._get_heavy_model()
        tokenizer = model.tokenizer
        # AbLang uses vocab_to_token dict (token -> id mapping)
        if hasattr(tokenizer, "vocab_to_token"):
            return tokenizer.vocab_to_token
        elif hasattr(tokenizer, "get_vocab"):
            return tokenizer.get_vocab()
        elif hasattr(tokenizer, "vocab"):
            return tokenizer.vocab
        else:
            return {str(i): i for i in range(len(tokenizer.vocab_to_aa))}

    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """Compute pseudo log-likelihood for a single sequence."""
        model = self._get_model_for_sequence(sequence)
        formatted = self._format_for_model([sequence])[0]
        tokenized = self._tokenize_with_model([formatted], model)
        input_ids = tokenized["input_ids"][0]

        mask_token_id = model.tokenizer.vocab_to_token[self.mask_token]
        pad_token_id = model.tokenizer.pad_token
        total_ll = 0.0

        for i in range(len(input_ids)):
            if input_ids[i] == pad_token_id:
                continue

            masked_ids = input_ids.clone()
            original_token = input_ids[i].item()
            masked_ids[i] = mask_token_id

            inputs = {"input_ids": masked_ids.unsqueeze(0)}

            with torch.no_grad():
                # AbLang returns logits tensor directly
                logits = model.AbLang(inputs["input_ids"])

                log_probs = F.log_softmax(logits[0, i], dim=-1)
                total_ll += log_probs[original_token].item()

        return total_ll

    def _fill_mask_batch(
        self,
        sequences: list[AntibodySequence],
        top_k: int,
    ) -> list[list[AntibodySequence]]:
        """Fill masks for a batch of sequences."""
        results = []

        for seq in sequences:
            model = self._get_model_for_sequence(seq)
            mask_token_id = model.tokenizer.vocab_to_token[self.mask_token]

            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize_with_model([formatted], model)
            input_ids = tokenized["input_ids"][0]

            with torch.no_grad():
                # AbLang returns logits tensor directly [batch, seq, vocab]
                all_logits = model.AbLang(tokenized["input_ids"])
                logits = all_logits[0]  # Get first (only) batch element

            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                results.append([seq])
                continue

            seq_results = []
            if len(mask_positions) == 1:
                pos = mask_positions[0].item()
                _, top_k_indices = torch.topk(logits[pos], top_k)

                for idx in top_k_indices:
                    filled_ids = input_ids.clone()
                    filled_ids[pos] = idx
                    filled_seq = self._decode_to_sequence(seq, filled_ids, model)
                    if filled_seq is not None:
                        seq_results.append(filled_seq)
            else:
                for k in range(top_k):
                    filled_ids = input_ids.clone()
                    for pos in mask_positions:
                        if k == 0:
                            pred_idx = logits[pos].argmax()
                        else:
                            _, indices = torch.topk(logits[pos], k + 1)
                            pred_idx = indices[min(k, len(indices) - 1)]
                        filled_ids[pos] = pred_idx

                    filled_seq = self._decode_to_sequence(seq, filled_ids, model)
                    if filled_seq is not None:
                        seq_results.append(filled_seq)

            results.append(seq_results if seq_results else [seq])

        return results

    def _decode_to_sequence(
        self,
        original: AntibodySequence,
        token_ids: torch.Tensor,
        model,
    ) -> AntibodySequence | None:
        """Decode token IDs back to AntibodySequence."""
        decoded = model.tokenizer.decode(token_ids)
        # Strip start (<) and end (>) tokens and any whitespace
        sequence = decoded.strip().lstrip("<").rstrip(">")

        try:
            if original.heavy_chain is not None:
                return AntibodySequence(heavy=sequence, species=original.species)
            else:
                return AntibodySequence(light=sequence, species=original.species)
        except Exception:
            return None

    def _mask_scan_batch(
        self,
        sequences: list[AntibodySequence],
        batch_size: int = 32,
    ) -> list[MaskScanOutput]:
        """Scan each position by masking it and collecting predictions."""
        results = []

        for seq in sequences:
            model = self._get_model_for_sequence(seq)
            mask_token_id = model.tokenizer.vocab_to_token[self.mask_token]
            pad_token_id = model.tokenizer.pad_token

            # Get vocab size
            vocab_size = len(model.tokenizer.vocab_to_token)

            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize_with_model([formatted], model)
            input_ids = tokenized["input_ids"][0]
            tokens = input_ids.tolist()

            seq_len = len(tokens)
            logits = torch.zeros(seq_len, vocab_size, device=self._primary_device)
            valid_mask = torch.zeros(seq_len, dtype=torch.bool, device=self._primary_device)

            # Build list of positions to mask (skip special tokens)
            positions_to_mask = []
            for i in range(1, seq_len - 1):  # Skip start and end tokens
                if tokens[i] != pad_token_id:
                    positions_to_mask.append(i)

            # Process masked variants in batches
            for batch_start in range(0, len(positions_to_mask), batch_size):
                batch_positions = positions_to_mask[batch_start:batch_start + batch_size]

                # Create masked variants for this batch
                masked_variants = []
                for pos in batch_positions:
                    masked_ids = input_ids.clone()
                    masked_ids[pos] = mask_token_id
                    masked_variants.append(masked_ids)

                # Stack into batch tensor
                batch_input_ids = torch.stack(masked_variants, dim=0)

                # Single batched forward pass
                with torch.no_grad():
                    # AbLang returns logits tensor directly
                    output_logits = model.AbLang(batch_input_ids)

                # Extract logits for each masked position
                for batch_idx, pos in enumerate(batch_positions):
                    logits[pos] = output_logits[batch_idx, pos]
                    valid_mask[pos] = True

            # Compute token offsets for this single sequence
            offsets = self._compute_token_offsets([seq], tokenized)[0]

            results.append(
                MaskScanOutput(
                    logits=logits.cpu(),
                    original_token_ids=input_ids.cpu(),
                    attention_mask=valid_mask.cpu(),
                    vocab=self._get_vocab(),
                    sequence=seq,
                    token_offsets=offsets,
                )
            )

        return results
