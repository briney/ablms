"""IgBERT encoder model wrapper."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence, Species


class IgBERT(EncoderAbLM):
    """
    IgBERT encoder model for antibody sequences.

    IgBERT is a BERT-based model trained on paired antibody sequences.
    It supports both single chain and paired heavy/light chain inputs.

    Model: Exscientia/IgBert
    Paper: https://arxiv.org/abs/2112.00306

    Attributes:
        model_name: "igbert"
        supports_paired: True
        max_length: 512
        embedding_dim: 768
        mask_token: "[MASK]"
        separator: "[SEP]"
    """

    model_name = "igbert"
    supports_paired = True
    max_length = 512
    embedding_dim = 768
    mask_token = "[MASK]"
    separator = "[SEP]"
    has_mlm_head = True

    MODEL_ID = "Exscientia/IgBert"

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize IgBERT model.

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

    def _format_for_model(
        self, sequences: List[AntibodySequence]
    ) -> List[str]:
        """
        Format sequences for IgBERT tokenization.

        Converts <MASK> to [MASK] and joins paired chains with [SEP].
        """
        formatted = []
        for seq in sequences:
            parts = []

            if seq.heavy_chain is not None:
                heavy = seq.heavy_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                # Add spaces between amino acids for IgBERT
                heavy = " ".join(heavy.replace(self.mask_token, " [MASK] ").split())
                parts.append(heavy)

            if seq.light_chain is not None:
                light = seq.light_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                # Add spaces between amino acids
                light = " ".join(light.replace(self.mask_token, " [MASK] ").split())
                parts.append(light)

            # Join with separator for paired sequences
            formatted.append(f" {self.separator} ".join(parts))

        return formatted

    def _tokenize(
        self, formatted_sequences: List[str]
    ) -> Dict[str, torch.Tensor]:
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
        sequences: List[AntibodySequence],
        tokenized: Dict[str, torch.Tensor],
    ) -> List[Dict[str, Tuple[int, int]]]:
        """Compute token offsets for each chain."""
        offsets = []
        input_ids = tokenized["input_ids"]

        sep_token_id = self._tokenizer.sep_token_id
        cls_token_id = self._tokenizer.cls_token_id

        for idx, seq in enumerate(sequences):
            seq_offsets = {}
            tokens = input_ids[idx].tolist()

            # Find positions of special tokens
            start = 1  # Skip [CLS]

            # Find [SEP] positions
            sep_positions = [i for i, t in enumerate(tokens) if t == sep_token_id]

            if seq.heavy_chain is not None:
                # Heavy chain: from after [CLS] to first [SEP]
                heavy_end = sep_positions[0] if sep_positions else len(tokens) - 1
                seq_offsets["heavy"] = (start, heavy_end)

                if seq.light_chain is not None and len(sep_positions) >= 1:
                    # Light chain: from after first [SEP] to second [SEP] or end
                    light_start = sep_positions[0] + 1
                    light_end = (
                        sep_positions[1] if len(sep_positions) > 1 else len(tokens) - 1
                    )
                    seq_offsets["light"] = (light_start, light_end)

            elif seq.light_chain is not None:
                # Only light chain
                light_end = sep_positions[0] if sep_positions else len(tokens) - 1
                seq_offsets["light"] = (start, light_end)

            offsets.append(seq_offsets)

        return offsets

    def _forward_embeddings(
        self,
        tokenized: Dict[str, torch.Tensor],
        layer: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get embeddings from a specific layer."""
        with torch.no_grad():
            outputs = self._model.bert(
                **tokenized,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        embeddings = hidden_states[layer]
        attention_mask = tokenized.get("attention_mask")

        return embeddings, attention_mask

    def _forward_all_hidden_states(
        self,
        tokenized: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor | None]:
        """Forward pass to get all hidden states."""
        with torch.no_grad():
            outputs = self._model.bert(
                **tokenized,
                output_hidden_states=True,
            )

        hidden_states = list(outputs.hidden_states)
        attention_mask = tokenized.get("attention_mask")

        return hidden_states, attention_mask

    def _forward_attention(
        self,
        tokenized: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get attention weights."""
        with torch.no_grad():
            outputs = self._model.bert(
                **tokenized,
                output_attentions=True,
            )

        # Stack attention from all layers: [batch, layers, heads, seq, seq]
        attentions = torch.stack(outputs.attentions, dim=1)
        attention_mask = tokenized.get("attention_mask")

        return attentions, attention_mask

    def _forward_logits(
        self,
        tokenized: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get MLM logits."""
        with torch.no_grad():
            outputs = self._model(**tokenized)

        logits = outputs.logits
        attention_mask = tokenized.get("attention_mask")

        return logits, attention_mask

    def _get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping."""
        return self._tokenizer.get_vocab()

    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """Compute pseudo log-likelihood for a single sequence."""
        formatted = self._format_for_model([sequence])[0]
        tokens = self._tokenizer.encode(formatted, add_special_tokens=True)

        total_ll = 0.0
        num_tokens = 0

        # Mask each position and compute log probability
        for i in range(1, len(tokens) - 1):  # Skip [CLS] and [SEP]
            masked_tokens = tokens.copy()
            original_token = tokens[i]
            masked_tokens[i] = self._tokenizer.mask_token_id

            inputs = {
                "input_ids": torch.tensor([masked_tokens], device=self._primary_device),
                "attention_mask": torch.ones(1, len(masked_tokens), device=self._primary_device),
            }

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0, i]
                log_probs = F.log_softmax(logits, dim=-1)
                total_ll += log_probs[original_token].item()
                num_tokens += 1

        return total_ll

    def _fill_mask_batch(
        self,
        sequences: List[AntibodySequence],
        top_k: int,
    ) -> List[List[AntibodySequence]]:
        """Fill masks for a batch of sequences."""
        results = []

        for seq in sequences:
            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize([formatted])

            with torch.no_grad():
                outputs = self._model(**tokenized)
                logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Find mask positions
            mask_token_id = self._tokenizer.mask_token_id
            input_ids = tokenized["input_ids"][0]
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                results.append([seq])
                continue

            # Get top-k predictions for each mask position
            # For simplicity, we handle single mask position
            # Multiple masks would need more complex logic
            seq_results = []
            if len(mask_positions) == 1:
                pos = mask_positions[0].item()
                top_k_logits, top_k_indices = torch.topk(logits[pos], top_k)

                for idx in top_k_indices:
                    token = self._tokenizer.decode([idx.item()]).strip()
                    # Reconstruct the sequence
                    filled_seq = self._reconstruct_sequence(seq, pos, token, tokenized)
                    if filled_seq is not None:
                        seq_results.append(filled_seq)
            else:
                # For multiple masks, just take argmax at each position
                for k in range(top_k):
                    filled_tokens = input_ids.clone()
                    for pos in mask_positions:
                        if k == 0:
                            pred_idx = logits[pos].argmax()
                        else:
                            # Get k-th best prediction
                            _, indices = torch.topk(logits[pos], k + 1)
                            pred_idx = indices[k]
                        filled_tokens[pos] = pred_idx

                    filled_seq = self._tokens_to_sequence(seq, filled_tokens)
                    if filled_seq is not None:
                        seq_results.append(filled_seq)

            results.append(seq_results if seq_results else [seq])

        return results

    def _reconstruct_sequence(
        self,
        original: AntibodySequence,
        mask_pos: int,
        token: str,
        tokenized: Dict[str, torch.Tensor],
    ) -> AntibodySequence | None:
        """Reconstruct sequence with filled token."""
        # Decode the entire sequence and parse it
        input_ids = tokenized["input_ids"][0].clone()
        input_ids[mask_pos] = self._tokenizer.convert_tokens_to_ids(token)

        decoded = self._tokenizer.decode(input_ids, skip_special_tokens=False)
        # Remove [CLS], [SEP], [PAD] and spaces
        decoded = decoded.replace("[CLS]", "").replace("[PAD]", "").strip()

        # Split by [SEP] for paired sequences
        parts = [p.strip().replace(" ", "") for p in decoded.split("[SEP]")]
        parts = [p for p in parts if p]

        if original.is_paired and len(parts) >= 2:
            return AntibodySequence(
                heavy=parts[0], light=parts[1], species=original.species
            )
        elif original.heavy_chain is not None and parts:
            return AntibodySequence(heavy=parts[0], species=original.species)
        elif original.light_chain is not None and parts:
            return AntibodySequence(light=parts[0], species=original.species)

        return None

    def _tokens_to_sequence(
        self,
        original: AntibodySequence,
        token_ids: torch.Tensor,
    ) -> AntibodySequence | None:
        """Convert token IDs back to AntibodySequence."""
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        decoded = decoded.replace("[CLS]", "").replace("[PAD]", "").strip()

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
