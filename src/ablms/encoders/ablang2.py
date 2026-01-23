"""AbLang2 encoder model wrapper."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import ModelLoadError


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

        # Move to device
        if hasattr(self._ablang, "AbRep"):
            self._ablang.AbRep = self._ablang.AbRep.to(self._primary_device)
            self._model = self._ablang.AbRep
        else:
            self._model = self._ablang

        # Get tokenizer
        self._tokenizer = self._ablang.tokenizer

    def _format_for_model(
        self, sequences: List[AntibodySequence]
    ) -> List[str]:
        """
        Format sequences for AbLang2.

        AbLang2 uses "*" as the mask token and "|" as the chain separator.
        """
        formatted = []
        for seq in sequences:
            parts = []

            if seq.heavy_chain is not None:
                heavy = seq.heavy_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                parts.append(heavy)

            if seq.light_chain is not None:
                light = seq.light_chain.replace(
                    AntibodySequence.MASK_TOKEN, self.mask_token
                )
                parts.append(light)

            # AbLang2 uses "|" separator
            formatted.append(self.separator.join(parts))

        return formatted

    def _tokenize(
        self, formatted_sequences: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize formatted sequences using AbLang2 tokenizer."""
        # AbLang2 has its own tokenization approach
        encoded = self._tokenizer(
            formatted_sequences,
            pad=True,
            return_tensors="pt",
        )

        if isinstance(encoded, dict):
            return {k: v.to(self._primary_device) for k, v in encoded.items()}
        else:
            # Handle if tokenizer returns just input_ids
            return {"input_ids": encoded.to(self._primary_device)}

    def _compute_token_offsets(
        self,
        sequences: List[AntibodySequence],
        tokenized: Dict[str, torch.Tensor],
    ) -> List[Dict[str, Tuple[int, int]]]:
        """Compute token offsets for each chain."""
        offsets = []
        input_ids = tokenized["input_ids"]

        # Find separator token ID
        sep_token_id = self._tokenizer.convert_tokens_to_ids(self.separator)

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
        tokenized: Dict[str, torch.Tensor],
        layer: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get embeddings from a specific layer."""
        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            # AbLang2 model forward pass
            outputs = self._model(
                input_ids,
                output_hidden_states=True,
            )

        # Get hidden states
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            hidden_states = outputs[1]
        else:
            # Fallback: use last_hidden_state
            hidden_states = [outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]]

        embeddings = hidden_states[layer]

        # Create attention mask
        attention_mask = (input_ids != self._tokenizer.pad_token_id).long()

        return embeddings, attention_mask

    def _forward_all_hidden_states(
        self,
        tokenized: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor | None]:
        """Forward pass to get all hidden states."""
        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            outputs = self._model(
                input_ids,
                output_hidden_states=True,
            )

        if hasattr(outputs, "hidden_states"):
            hidden_states = list(outputs.hidden_states)
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            hidden_states = list(outputs[1])
        else:
            hidden_states = [outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]]

        attention_mask = (input_ids != self._tokenizer.pad_token_id).long()

        return hidden_states, attention_mask

    def _forward_attention(
        self,
        tokenized: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get attention weights."""
        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            outputs = self._model(
                input_ids,
                output_attentions=True,
            )

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            attentions = torch.stack(outputs.attentions, dim=1)
        else:
            # Return empty attention if not available
            batch_size, seq_len = input_ids.shape
            attentions = torch.zeros(batch_size, 1, 1, seq_len, seq_len, device=self._primary_device)

        attention_mask = (input_ids != self._tokenizer.pad_token_id).long()

        return attentions, attention_mask

    def _forward_logits(
        self,
        tokenized: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to get MLM logits."""
        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            # Use AbLang2's MLM head if available
            if hasattr(self._ablang, "AbLang"):
                outputs = self._ablang.AbLang(input_ids)
            else:
                outputs = self._model(input_ids)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        attention_mask = (input_ids != self._tokenizer.pad_token_id).long()

        return logits, attention_mask

    def _get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping."""
        if hasattr(self._tokenizer, "get_vocab"):
            return self._tokenizer.get_vocab()
        elif hasattr(self._tokenizer, "vocab"):
            return self._tokenizer.vocab
        else:
            # Build vocab from tokenizer attributes
            return {str(i): i for i in range(self._tokenizer.vocab_size)}

    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """Compute pseudo log-likelihood for a single sequence."""
        formatted = self._format_for_model([sequence])[0]
        tokenized = self._tokenize([formatted])
        input_ids = tokenized["input_ids"][0]

        mask_token_id = self._tokenizer.convert_tokens_to_ids(self.mask_token)
        total_ll = 0.0

        for i in range(len(input_ids)):
            if input_ids[i] == self._tokenizer.pad_token_id:
                continue
            if input_ids[i] == self._tokenizer.convert_tokens_to_ids(self.separator):
                continue

            masked_ids = input_ids.clone()
            original_token = input_ids[i].item()
            masked_ids[i] = mask_token_id

            inputs = {"input_ids": masked_ids.unsqueeze(0)}

            with torch.no_grad():
                logits, _ = self._forward_logits(inputs)
                log_probs = F.log_softmax(logits[0, i], dim=-1)
                total_ll += log_probs[original_token].item()

        return total_ll

    def _fill_mask_batch(
        self,
        sequences: List[AntibodySequence],
        top_k: int,
    ) -> List[List[AntibodySequence]]:
        """Fill masks for a batch of sequences."""
        results = []
        mask_token_id = self._tokenizer.convert_tokens_to_ids(self.mask_token)

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

        # Parse the decoded string
        parts = decoded.split(self.separator)
        parts = [p.strip() for p in parts if p.strip()]

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
