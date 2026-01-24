"""IgT5 encoder model wrapper."""

from __future__ import annotations


import torch
from transformers import T5EncoderModel, T5Tokenizer

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import UnsupportedOperationError
from ablms.outputs import MaskScanOutput


class IgT5(EncoderAbLM):
    """
    IgT5 encoder model for antibody sequences.

    IgT5 is a T5-based encoder model trained on paired antibody sequences.
    Note: IgT5 does NOT have an MLM head, so get_logits(), pseudo_log_likelihood(),
    and fill_mask() will raise UnsupportedOperationError.

    Model: Exscientia/IgT5
    Paper: https://arxiv.org/abs/2112.00306

    Attributes:
        model_name: "igt5"
        supports_paired: True
        max_length: 512
        embedding_dim: 1024
        separator: "</s>"
        has_mlm_head: False
    """

    model_name = "igt5"
    supports_paired = True
    max_length = 512
    embedding_dim = 1024
    mask_token = None  # T5 doesn't use mask tokens for embeddings
    separator = "</s>"
    has_mlm_head = False

    MODEL_ID = "Exscientia/IgT5"

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize IgT5 model.

        Args:
            device: (DEPRECATED) Device for inference. Use 'devices' instead.
            devices: Device(s) for inference. Auto-selects all GPUs if None.
        """
        super().__init__(device=device, devices=devices)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        self._tokenizer = T5Tokenizer.from_pretrained(self.MODEL_ID)
        self._model = T5EncoderModel.from_pretrained(self.MODEL_ID)
        self._model = self._model.to(self._primary_device)
        self._model.eval()

    def _format_for_model(
        self, sequences: list[AntibodySequence]
    ) -> list[str]:
        """
        Format sequences for IgT5 tokenization.

        IgT5 expects space-separated amino acids with </s> between chains.
        """
        formatted = []
        for seq in sequences:
            parts = []

            if seq.heavy_chain is not None:
                # Remove any mask tokens (IgT5 doesn't support MLM)
                heavy = seq.heavy_chain.replace(AntibodySequence.MASK_TOKEN, "X")
                # Add spaces between amino acids
                heavy = " ".join(list(heavy))
                parts.append(heavy)

            if seq.light_chain is not None:
                light = seq.light_chain.replace(AntibodySequence.MASK_TOKEN, "X")
                light = " ".join(list(light))
                parts.append(light)

            # Join with </s> separator
            formatted.append(f" {self.separator} ".join(parts))

        return formatted

    def _tokenize(
        self, formatted_sequences: list[str]
    ) -> dict[str, torch.Tensor]:
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

        eos_token_id = self._tokenizer.eos_token_id

        for idx, seq in enumerate(sequences):
            seq_offsets = {}
            tokens = input_ids[idx].tolist()

            start = 0

            # Find </s> positions
            eos_positions = [i for i, t in enumerate(tokens) if t == eos_token_id]

            if seq.heavy_chain is not None:
                heavy_len = seq.length.get("heavy", 0)
                # Account for space-separated tokens
                seq_offsets["heavy"] = (start, start + heavy_len)
                start += heavy_len

                if seq.light_chain is not None and eos_positions:
                    # Skip the </s> separator
                    light_start = eos_positions[0] + 1
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
        with torch.no_grad():
            outputs = self._model(
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
            outputs = self._model(
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
            outputs = self._model(
                **tokenized,
                output_attentions=True,
            )

        # Stack attention from all layers: [batch, layers, heads, seq, seq]
        attentions = torch.stack(outputs.attentions, dim=1)
        attention_mask = tokenized.get("attention_mask")

        return attentions, attention_mask

    def _forward_logits(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """IgT5 does not have an MLM head."""
        raise UnsupportedOperationError(
            "IgT5 does not have a masked language modeling head. "
            "Use get_embeddings() or get_attention() instead."
        )

    def _get_vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping."""
        return self._tokenizer.get_vocab()

    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """IgT5 does not support pseudo log-likelihood."""
        raise UnsupportedOperationError(
            "IgT5 does not have a masked language modeling head and "
            "cannot compute pseudo log-likelihood."
        )

    def _fill_mask_batch(
        self,
        sequences: list[AntibodySequence],
        top_k: int,
    ) -> list[list[AntibodySequence]]:
        """IgT5 does not support mask filling."""
        raise UnsupportedOperationError(
            "IgT5 does not have a masked language modeling head and "
            "cannot fill masks."
        )

    def _mask_scan_batch(
        self,
        sequences: list[AntibodySequence],
        batch_size: int = 32,
    ) -> list[MaskScanOutput]:
        """IgT5 does not support mask scanning."""
        raise UnsupportedOperationError(
            "IgT5 does not have a masked language modeling head and "
            "cannot perform mask scanning."
        )
