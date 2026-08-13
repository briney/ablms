"""AbLang2 encoder model wrapper."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import ModelLoadError
from ablms.outputs import MaskScanOutput


class AbLang2(EncoderAbLM):
    """
    AbLang2 encoder model for antibody sequences.

    AbLang2 is trained on paired antibody sequences and supports both
    single chain and paired heavy/light chain inputs.

    Package: ablang2
    Paper: https://www.biorxiv.org/content/10.1101/2024.02.12.579844

    Attributes:
        model_name: "ablang2"
        supports_paired: True
        max_length: 512
        embedding_dim: 480
        mask_token: "*"
        separator: "|"
    """

    model_name = "ablang2"
    supports_paired = True
    max_length = 512
    embedding_dim = 480
    mask_token = "*"
    separator = "|"
    has_mlm_head = True

    @property
    def num_layers(self) -> int:
        """AbLang2 has no HuggingFace config; count the encoder blocks directly."""
        return len(self._model.encoder_blocks)

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize AbLang2 model.

        Args:
            device: (DEPRECATED) Device for inference. Use 'devices' instead.
            devices: Device(s) for inference. Auto-selects all GPUs if None.
        """
        super().__init__(device=device, devices=devices)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from the ablang2 package."""
        try:
            import ablang2
        except ImportError as e:
            raise ModelLoadError(
                "Failed to import ablang2 package. "
                "Install it with: pip install ablang2"
            ) from e

        # AbLang2 provides a pre-built model
        self._ablang = ablang2.pretrained()
        self._ablang.freeze()

        # Move entire model to device (both AbRep and AbLang with AbHead)
        if hasattr(self._ablang, "AbLang"):
            self._ablang.AbLang = self._ablang.AbLang.to(self._primary_device)
        if hasattr(self._ablang, "AbRep"):
            self._ablang.AbRep = self._ablang.AbRep.to(self._primary_device)
            self._model = self._ablang.AbRep
        else:
            self._model = self._ablang

        # Get tokenizer
        self._tokenizer = self._ablang.tokenizer

    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """
        Format sequences for AbLang2.

        AbLang2 uses "*" as the mask token and expects format "<heavy>|<light>".
        We format sequences this way and use w_extra_tkns=False in tokenization.
        """
        formatted = []
        for seq in sequences:
            parts = []

            if seq.heavy_chain is not None:
                heavy = seq.heavy_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                parts.append(f"<{heavy}>")

            if seq.light_chain is not None:
                light = seq.light_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                parts.append(f"<{light}>")

            # AbLang2 uses "|" separator between <heavy> and <light>
            formatted.append(self.separator.join(parts))

        return formatted

    def _tokenize(self, formatted_sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize formatted sequences using AbLang2 tokenizer."""
        # AbLang2 tokenizer doesn't support return_tensors - returns tensor directly
        # Use w_extra_tkns=False since we pre-format sequences as <heavy>|<light>
        encoded = self._tokenizer(formatted_sequences, pad=True, w_extra_tkns=False)

        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(self._primary_device)
            # Create attention mask based on padding
            attention_mask = (input_ids != self._tokenizer.pad_token).long()
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
        input_ids = tokenized["input_ids"]

        # Find separator token ID (ablang2 tokenizer has sep_token attribute)
        sep_token_id = self._tokenizer.sep_token

        for idx, seq in enumerate(sequences):
            seq_offsets = {}
            tokens = input_ids[idx].tolist()

            # Find separator position
            sep_positions = [i for i, t in enumerate(tokens) if t == sep_token_id]

            start = 1  # Skip start token

            if seq.heavy_chain is not None:
                heavy_len = seq.length.get("heavy", 0)
                heavy_end = sep_positions[0] if sep_positions else start + heavy_len
                seq_offsets["heavy"] = (start, min(heavy_end, start + heavy_len))

                if seq.light_chain is not None and sep_positions:
                    light_start = sep_positions[0] + 1
                    light_len = seq.length.get("light", 0)
                    seq_offsets["light"] = (light_start, light_start + light_len)

            elif seq.light_chain is not None:
                light_len = seq.length.get("light", 0)
                seq_offsets["light"] = (start, start + light_len)

            offsets.append(seq_offsets)

        return offsets

    def _forward_embeddings(
        self,
        tokenized: dict[str, torch.Tensor],
        layer: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get embeddings from a specific layer."""
        input_ids = tokenized["input_ids"]
        num_layers = (
            len(self._model.encoder_blocks) + 1
        )  # 0 is embedding, 1-12 are encoder blocks

        # Convert negative layer index to positive
        if layer < 0:
            layer_idx = num_layers + layer
        else:
            layer_idx = layer

        with torch.no_grad():
            # AbLang2 uses return_rep_layers to get specific layer outputs
            outputs = self._model(input_ids, return_rep_layers=[layer_idx])

        # outputs.many_hidden_states is a dict {layer_idx: tensor}
        embeddings = outputs.many_hidden_states[layer_idx]

        # Create attention mask
        attention_mask = (input_ids != self._tokenizer.pad_token).long()

        return embeddings, attention_mask

    def _forward_all_hidden_states(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Forward pass to get all hidden states."""
        input_ids = tokenized["input_ids"]
        num_layers = (
            len(self._model.encoder_blocks) + 1
        )  # 0 is embedding, 1-12 are encoder blocks

        with torch.no_grad():
            # Request all layers (0 = embedding, 1-12 = encoder blocks)
            outputs = self._model(input_ids, return_rep_layers=list(range(num_layers)))

        # outputs.many_hidden_states is a dict {layer_idx: tensor}
        # Convert to list in order
        hidden_states = [outputs.many_hidden_states[i] for i in range(num_layers)]

        attention_mask = (input_ids != self._tokenizer.pad_token).long()

        return hidden_states, attention_mask

    def _forward_attention(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get attention weights."""
        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            # AbLang2 uses return_attn_weights parameter
            outputs = self._model(input_ids, return_attn_weights=True)

        # outputs.attention_weights is a list of attention tensors per layer
        if outputs.attention_weights:
            attentions = torch.stack(outputs.attention_weights, dim=1)
        else:
            # Return empty attention if not available
            batch_size, seq_len = input_ids.shape
            attentions = torch.zeros(
                batch_size, 1, 1, seq_len, seq_len, device=self._primary_device
            )

        attention_mask = (input_ids != self._tokenizer.pad_token).long()

        return attentions, attention_mask

    def _forward_logits(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get MLM logits."""
        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            # AbLang2's AbLang model returns logits directly when called without special args
            logits = self._ablang.AbLang(input_ids)

        attention_mask = (input_ids != self._tokenizer.pad_token).long()

        return logits, attention_mask

    def _get_vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping."""
        if hasattr(self._tokenizer, "get_vocab"):
            return self._tokenizer.get_vocab()
        elif hasattr(self._tokenizer, "aa_to_token"):
            # AbLang2 tokenizer uses aa_to_token
            return self._tokenizer.aa_to_token
        elif hasattr(self._tokenizer, "vocab"):
            return self._tokenizer.vocab
        else:
            # Build vocab from tokenizer attributes
            return {str(i): i for i in range(len(self._tokenizer.aa_to_token))}

    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """Compute pseudo log-likelihood for a single sequence."""
        formatted = self._format_for_model([sequence])[0]
        tokenized = self._tokenize([formatted])
        input_ids = tokenized["input_ids"][0]

        mask_token_id = self._tokenizer.mask_token
        total_ll = 0.0

        for i in range(len(input_ids)):
            if input_ids[i] == self._tokenizer.pad_token:
                continue
            if input_ids[i] == self._tokenizer.sep_token:
                continue

            masked_ids = input_ids.clone()
            original_token = int(input_ids[i].item())
            masked_ids[i] = mask_token_id

            inputs = {"input_ids": masked_ids.unsqueeze(0)}

            with torch.no_grad():
                logits, _ = self._forward_logits(inputs)
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
        mask_token_id = self._tokenizer.mask_token

        for seq in sequences:
            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize([formatted])
            input_ids = tokenized["input_ids"][0]

            with torch.no_grad():
                logits, _ = self._forward_logits(tokenized)
                logits = logits[0]

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
                    filled_seq = self._decode_to_sequence(seq, filled_ids)
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

                    filled_seq = self._decode_to_sequence(seq, filled_ids)
                    if filled_seq is not None:
                        seq_results.append(filled_seq)

            results.append(seq_results if seq_results else [seq])

        return results

    def _decode_to_sequence(
        self,
        original: AntibodySequence,
        token_ids: torch.Tensor,
    ) -> AntibodySequence | None:
        """Decode token IDs back to AntibodySequence."""
        decoded = self._tokenizer.decode(token_ids)

        # Parse the decoded string - format is <heavy>|<light> or <heavy>
        parts = decoded.split(self.separator)
        # Strip whitespace and remove <> brackets from chain sequences
        parts = [p.strip().strip("<>") for p in parts if p.strip()]

        try:
            if original.is_paired and len(parts) >= 2:
                return AntibodySequence(
                    heavy=parts[0], light=parts[1], species=original.species
                )
            elif original.heavy_chain is not None and parts:
                return AntibodySequence(heavy=parts[0], species=original.species)
            elif original.light_chain is not None and parts:
                return AntibodySequence(light=parts[0], species=original.species)
        except Exception:
            pass

        return None

    def _mask_scan_batch(
        self,
        sequences: list[AntibodySequence],
        batch_size: int = 32,
    ) -> list[MaskScanOutput]:
        """Scan each position by masking it and collecting predictions."""
        results = []
        mask_token_id = self._tokenizer.mask_token
        sep_token_id = self._tokenizer.sep_token
        pad_token_id = self._tokenizer.pad_token

        # Get vocab size from model config or tokenizer
        if hasattr(self._model, "config") and hasattr(self._model.config, "vocab_size"):
            vocab_size = self._model.config.vocab_size
        elif hasattr(self._tokenizer, "aa_to_token"):
            vocab_size = len(self._tokenizer.aa_to_token)
        else:
            vocab_size = len(self._get_vocab())

        for seq in sequences:
            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize([formatted])
            input_ids = tokenized["input_ids"][0]
            tokens = input_ids.tolist()

            seq_len = len(tokens)
            logits = torch.zeros(seq_len, vocab_size, device=self._primary_device)
            valid_mask = torch.zeros(
                seq_len, dtype=torch.bool, device=self._primary_device
            )

            # Build list of positions to mask (skip special tokens)
            positions_to_mask = []
            for i in range(1, seq_len - 1):  # Skip start and end tokens
                if tokens[i] != pad_token_id and tokens[i] != sep_token_id:
                    positions_to_mask.append(i)

            # Process masked variants in batches
            for batch_start in range(0, len(positions_to_mask), batch_size):
                batch_positions = positions_to_mask[
                    batch_start : batch_start + batch_size
                ]

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
                    output_logits, _ = self._forward_logits(
                        {"input_ids": batch_input_ids}
                    )

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
