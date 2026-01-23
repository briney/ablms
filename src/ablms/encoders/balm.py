"""BALM-paired encoder model wrapper."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence


class BALM(EncoderAbLM):
    """
    BALM-paired encoder model for antibody sequences.

    BALM-paired is a RoBERTa-based model trained on paired antibody sequences.
    It uses </s></s> as the chain separator.

    Model: BALM/BALM-paired

    Attributes:
        model_name: "balm"
        supports_paired: True
        max_length: 512
        embedding_dim: 1024
        mask_token: "<mask>"
        separator: "</s></s>"
    """

    model_name = "balm"
    supports_paired = True
    max_length = 512
    embedding_dim = 1024
    mask_token = "<mask>"
    separator = "</s></s>"
    has_mlm_head = True

    MODEL_ID = "BALM/BALM-paired"

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize BALM model.

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
        Format sequences for BALM tokenization.

        BALM expects space-separated amino acids with </s></s> between chains.
        """
        formatted = []
        for seq in sequences:
            parts = []

            if seq.heavy_chain is not None:
                heavy = seq.heavy_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                # Add spaces between amino acids, preserving mask token
                result = []
                i = 0
                while i < len(heavy):
                    if heavy[i:].startswith(self.mask_token):
                        result.append(self.mask_token)
                        i += len(self.mask_token)
                    else:
                        result.append(heavy[i])
                        i += 1
                parts.append(" ".join(result))

            if seq.light_chain is not None:
                light = seq.light_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                result = []
                i = 0
                while i < len(light):
                    if light[i:].startswith(self.mask_token):
                        result.append(self.mask_token)
                        i += len(self.mask_token)
                    else:
                        result.append(light[i])
                        i += 1
                parts.append(" ".join(result))

            # Join with separator
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

        eos_token_id = self._tokenizer.eos_token_id

        for idx, seq in enumerate(sequences):
            seq_offsets = {}
            tokens = input_ids[idx].tolist()

            # Find </s> positions (EOS tokens)
            eos_positions = [i for i, t in enumerate(tokens) if t == eos_token_id]

            # Start after <s> (BOS)
            start = 1

            if seq.heavy_chain is not None:
                heavy_len = seq.length.get("heavy", 0)
                seq_offsets["heavy"] = (start, start + heavy_len)

                if seq.light_chain is not None and len(eos_positions) >= 2:
                    # Light chain starts after the </s></s> separator
                    light_start = eos_positions[1] + 1
                    light_len = seq.length.get("light", 0)
                    seq_offsets["light"] = (light_start, light_start + light_len)

            elif seq.light_chain is not None:
                light_len = seq.length.get("light", 0)
                seq_offsets["light"] = (start, start + light_len)

            offsets.append(seq_offsets)

        return offsets

    def _forward_embeddings(
        self,
        tokenized: Dict[str, torch.Tensor],
        layer: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get embeddings from a specific layer."""
        with torch.no_grad():
            outputs = self._model.roberta(
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
            outputs = self._model.roberta(
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
            outputs = self._model.roberta(
                **tokenized,
                output_attentions=True,
            )

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

        for i in range(1, len(tokens) - 1):
            # Skip separator tokens
            if tokens[i] == self._tokenizer.eos_token_id:
                continue

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

        # Parse the decoded string
        decoded = decoded.replace("<s>", "").replace("</s>", " | ").replace("<pad>", "")
        parts = [p.strip().replace(" ", "") for p in decoded.split("|")]
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
