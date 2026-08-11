"""AntiBERTa2 encoder model wrapper."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.outputs import MaskScanOutput


class AntiBERTa2(EncoderAbLM):
    """
    AntiBERTa2 encoder model for antibody sequences.

    AntiBERTa2 is based on the RoFormer architecture with rotary position
    embeddings. It supports both unpaired (heavy or light chain only) and
    paired antibody sequences, where chains are separated by a [SEP] token.

    Model: alchemab/antiberta2
    Paper: https://www.biorxiv.org/content/10.1101/2023.08.08.552446

    Attributes:
        model_name: "antiberta2"
        supports_paired: True
        max_length: 512
        embedding_dim: 1024
        mask_token: "[MASK]"
        separator: "[SEP]"
    """

    model_name = "antiberta2"
    supports_paired = True
    max_length = 512
    embedding_dim = 1024
    mask_token = "[MASK]"
    separator = "[SEP]"
    has_mlm_head = True

    MODEL_ID = "alchemab/antiberta2"

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize AntiBERTa2 model.

        Args:
            device: (DEPRECATED) Device for inference. Use 'devices' instead.
            devices: Device(s) for inference. Auto-selects all GPUs if None.
        """
        super().__init__(device=device, devices=devices)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForMaskedLM.from_pretrained(self.MODEL_ID)
        self._model = self._model.to(self._primary_device)
        self._model.eval()

        # Get the base model attribute name (varies by architecture)
        # RoFormer models typically use 'roformer' as the base attribute
        if hasattr(self._model, "roformer"):
            self._base_model_attr = "roformer"
        elif hasattr(self._model, "roberta"):
            self._base_model_attr = "roberta"
        else:
            # Fallback: try to find the encoder
            self._base_model_attr = None

    def _get_base_model(self):
        """Get the base encoder model."""
        if self._base_model_attr:
            return getattr(self._model, self._base_model_attr)
        return self._model

    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """
        Format sequences for AntiBERTa2 tokenization.

        AntiBERTa2 expects space-separated amino acids. For paired sequences,
        chains are separated by [SEP].
        """
        formatted = []
        for seq in sequences:
            parts = []

            if seq.heavy_chain is not None:
                heavy = seq.heavy_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                # Add spaces between amino acids, preserving mask token
                heavy_spaced = self._add_spaces(heavy)
                parts.append(heavy_spaced)

            if seq.light_chain is not None:
                light = seq.light_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                light_spaced = self._add_spaces(light)
                parts.append(light_spaced)

            # Join with [SEP] for paired sequences
            formatted.append(f" {self.separator} ".join(parts))

        return formatted

    def _add_spaces(self, sequence: str) -> str:
        """Add spaces between amino acids while preserving mask tokens."""
        result = []
        i = 0
        while i < len(sequence):
            if sequence[i:].startswith(self.mask_token):
                result.append(self.mask_token)
                i += len(self.mask_token)
            else:
                result.append(sequence[i])
                i += 1
        return " ".join(result)

    def _tokenize(self, formatted_sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize formatted sequences."""
        encoded = self._tokenizer(
            formatted_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.to(self._primary_device) for k, v in encoded.items()}

    def _compute_token_offsets(
        self,
        sequences: list[AntibodySequence],
        tokenized: dict[str, torch.Tensor],
    ) -> list[dict[str, tuple[int, int]]]:
        """Compute token offsets for each chain."""
        offsets = []
        input_ids = tokenized["input_ids"]

        sep_token_id = self._tokenizer.sep_token_id

        for idx, seq in enumerate(sequences):
            seq_offsets = {}
            tokens = input_ids[idx].tolist()

            # Find [SEP] positions
            sep_positions = [i for i, t in enumerate(tokens) if t == sep_token_id]

            start = 1  # Skip [CLS]

            if seq.heavy_chain is not None:
                heavy_len = seq.length.get("heavy", 0)
                # Heavy chain ends at first [SEP] or calculated length
                heavy_end = sep_positions[0] if sep_positions else start + heavy_len
                seq_offsets["heavy"] = (start, min(heavy_end, start + heavy_len))

                if seq.light_chain is not None and sep_positions:
                    # Light chain starts after [SEP]
                    light_start = sep_positions[0] + 1
                    light_len = seq.length.get("light", 0)
                    light_end = (
                        sep_positions[1]
                        if len(sep_positions) > 1
                        else light_start + light_len
                    )
                    seq_offsets["light"] = (
                        light_start,
                        min(light_end, light_start + light_len),
                    )

            elif seq.light_chain is not None:
                light_len = seq.length.get("light", 0)
                light_end = sep_positions[0] if sep_positions else start + light_len
                seq_offsets["light"] = (start, min(light_end, start + light_len))

            offsets.append(seq_offsets)

        return offsets

    def _forward_embeddings(
        self,
        tokenized: dict[str, torch.Tensor],
        layer: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get embeddings from a specific layer."""
        with torch.no_grad():
            base_model = self._get_base_model()
            outputs = base_model(
                **tokenized,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        embeddings = hidden_states[layer]
        attention_mask = tokenized.get("attention_mask")

        return embeddings, attention_mask

    def _forward_all_hidden_states(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Forward pass to get all hidden states."""
        with torch.no_grad():
            base_model = self._get_base_model()
            outputs = base_model(
                **tokenized,
                output_hidden_states=True,
            )

        hidden_states = list(outputs.hidden_states)
        attention_mask = tokenized.get("attention_mask")

        return hidden_states, attention_mask

    def _forward_attention(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get attention weights."""
        with torch.no_grad():
            base_model = self._get_base_model()
            outputs = base_model(
                **tokenized,
                output_attentions=True,
            )

        attentions = torch.stack(outputs.attentions, dim=1)
        attention_mask = tokenized.get("attention_mask")

        return attentions, attention_mask

    def _forward_logits(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get MLM logits."""
        with torch.no_grad():
            outputs = self._model(**tokenized)

        logits = outputs.logits
        attention_mask = tokenized.get("attention_mask")

        return logits, attention_mask

    def _get_vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping."""
        return self._tokenizer.get_vocab()

    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """Compute pseudo log-likelihood for a single sequence."""
        formatted = self._format_for_model([sequence])[0]
        tokens = self._tokenizer.encode(formatted, add_special_tokens=True)

        total_ll = 0.0
        sep_token_id = self._tokenizer.sep_token_id

        for i in range(1, len(tokens) - 1):
            # Skip [SEP] tokens
            if tokens[i] == sep_token_id:
                continue

            masked_tokens = tokens.copy()
            original_token = tokens[i]
            masked_tokens[i] = self._tokenizer.mask_token_id

            inputs = {
                "input_ids": torch.tensor([masked_tokens], device=self._primary_device),
                "attention_mask": torch.ones(
                    1, len(masked_tokens), device=self._primary_device
                ),
            }

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0, i]
                log_probs = F.log_softmax(logits, dim=-1)
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
            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize([formatted])

            with torch.no_grad():
                outputs = self._model(**tokenized)
                logits = outputs.logits[0]

            mask_token_id = self._tokenizer.mask_token_id
            input_ids = tokenized["input_ids"][0]
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
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=False)

        # Remove special tokens and parse
        decoded = decoded.replace("[CLS]", "").replace("[PAD]", "").strip()

        # Split by [SEP] for paired sequences
        parts = [p.strip().replace(" ", "") for p in decoded.split("[SEP]")]
        parts = [p for p in parts if p]

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
        sep_token_id = self._tokenizer.sep_token_id
        mask_token_id = self._tokenizer.mask_token_id

        for seq in sequences:
            formatted = self._format_for_model([seq])[0]
            tokens = self._tokenizer.encode(formatted, add_special_tokens=True)

            seq_len = len(tokens)
            vocab_size = self._model.config.vocab_size
            logits = torch.zeros(seq_len, vocab_size, device=self._primary_device)
            valid_mask = torch.zeros(
                seq_len, dtype=torch.bool, device=self._primary_device
            )

            # Build list of positions to mask (skip special tokens)
            positions_to_mask = []
            for i in range(1, seq_len - 1):  # Skip [CLS] and final [SEP]
                if tokens[i] != sep_token_id:
                    positions_to_mask.append(i)

            # Process masked variants in batches
            for batch_start in range(0, len(positions_to_mask), batch_size):
                batch_positions = positions_to_mask[
                    batch_start : batch_start + batch_size
                ]
                current_batch_size = len(batch_positions)

                # Create masked variants for this batch
                masked_variants = []
                for pos in batch_positions:
                    masked_tokens = tokens.copy()
                    masked_tokens[pos] = mask_token_id
                    masked_variants.append(masked_tokens)

                # Stack into batch tensors
                input_ids = torch.tensor(masked_variants, device=self._primary_device)
                attention_mask = torch.ones(
                    current_batch_size, seq_len, device=self._primary_device
                )

                # Single batched forward pass
                with torch.no_grad():
                    outputs = self._model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )

                # Extract logits for each masked position
                for batch_idx, pos in enumerate(batch_positions):
                    logits[pos] = outputs.logits[batch_idx, pos]
                    valid_mask[pos] = True

            # Compute token offsets for this single sequence
            tokenized = {
                "input_ids": torch.tensor([tokens], device=self._primary_device)
            }
            offsets = self._compute_token_offsets([seq], tokenized)[0]

            results.append(
                MaskScanOutput(
                    logits=logits.cpu(),
                    original_token_ids=torch.tensor(tokens, device="cpu"),
                    attention_mask=valid_mask.cpu(),
                    vocab=self._get_vocab(),
                    sequence=seq,
                    token_offsets=offsets,
                )
            )

        return results
