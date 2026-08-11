"""AntiBERTy encoder model wrapper."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import ModelLoadError
from ablms.outputs import MaskScanOutput


class AntiBERTy(EncoderAbLM):
    """
    AntiBERTy encoder model for antibody sequences.

    AntiBERTy is a BERT-based model trained on unpaired antibody sequences
    using the antiberty package. It does NOT support paired sequences.

    Package: antiberty
    Paper: https://www.biorxiv.org/content/10.1101/2021.07.09.451745

    Attributes:
        model_name: "antiberty"
        supports_paired: False
        max_length: 512
        embedding_dim: 512
        mask_token: "_"
    """

    model_name = "antiberty"
    supports_paired = False
    max_length = 512
    embedding_dim = 512
    mask_token = "_"
    separator = None
    has_mlm_head = True

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize AntiBERTy model.

        Args:
            device: (DEPRECATED) Device for inference. Use 'devices' instead.
            devices: Device(s) for inference. Auto-selects all GPUs if None.
        """
        super().__init__(device=device, devices=devices)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from the antiberty package."""
        try:
            from antiberty import AntiBERTyRunner
        except ImportError as e:
            raise ModelLoadError(
                "Failed to import antiberty package. "
                "Install it with: pip install antiberty"
            ) from e

        self._runner = AntiBERTyRunner()
        # Move model to device
        self._runner.model = self._runner.model.to(self._primary_device)
        self._runner.model.eval()
        self._model = self._runner.model
        self._tokenizer = self._runner.tokenizer

    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """
        Format sequences for AntiBERTy.

        AntiBERTy uses "_" as the mask token and expects raw sequences.
        """
        formatted = []
        for seq in sequences:
            sequence = seq.heavy_chain or seq.light_chain

            # Convert unified mask token to AntiBERTy mask token
            sequence = sequence.replace(AntibodySequence.MASK_TOKEN, self.mask_token)

            formatted.append(sequence)

        return formatted

    def _tokenize(self, formatted_sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize formatted sequences using AntiBERTy tokenizer."""
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

        for seq in sequences:
            seq_offsets = {}
            start = 1  # Skip [CLS]

            if seq.heavy_chain is not None:
                seq_len = seq.length.get("heavy", 0)
                seq_offsets["heavy"] = (start, start + seq_len)
            elif seq.light_chain is not None:
                seq_len = seq.length.get("light", 0)
                seq_offsets["light"] = (start, start + seq_len)

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

        # AntiBERTy returns logits directly
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
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
        mask_token_id = self._tokenizer.convert_tokens_to_ids(self.mask_token)

        for i in range(1, len(tokens) - 1):
            masked_tokens = tokens.copy()
            original_token = tokens[i]
            masked_tokens[i] = mask_token_id

            inputs = {
                "input_ids": torch.tensor([masked_tokens], device=self._primary_device),
                "attention_mask": torch.ones(
                    1, len(masked_tokens), device=self._primary_device
                ),
            }

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = (
                    outputs.logits[0, i]
                    if hasattr(outputs, "logits")
                    else outputs[0][0, i]
                )
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
        mask_token_id = self._tokenizer.convert_tokens_to_ids(self.mask_token)

        for seq in sequences:
            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize([formatted])

            with torch.no_grad():
                outputs = self._model(**tokenized)
                logits = (
                    outputs.logits[0] if hasattr(outputs, "logits") else outputs[0][0]
                )

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
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        sequence = decoded.replace(" ", "")

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
        mask_token_id = self._tokenizer.convert_tokens_to_ids(self.mask_token)

        for seq in sequences:
            formatted = self._format_for_model([seq])[0]
            tokens = self._tokenizer.encode(formatted, add_special_tokens=True)

            seq_len = len(tokens)
            vocab_size = self._model.config.vocab_size
            logits = torch.zeros(seq_len, vocab_size, device=self._primary_device)
            valid_mask = torch.zeros(
                seq_len, dtype=torch.bool, device=self._primary_device
            )

            # Build list of positions to mask (skip [CLS] and [SEP])
            positions_to_mask = list(range(1, seq_len - 1))

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
                    output_logits = (
                        outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    )

                # Extract logits for each masked position
                for batch_idx, pos in enumerate(batch_positions):
                    logits[pos] = output_logits[batch_idx, pos]
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
