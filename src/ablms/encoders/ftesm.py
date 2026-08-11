"""ft-ESM encoder model wrapper."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForMaskedLM

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.outputs import MaskScanOutput


class FtESM(EncoderAbLM):
    """
    ft-ESM encoder model for antibody sequences.

    ft-ESM is an ESM2-based model (finetuned from facebook/esm2_t33_650M_UR50D)
    trained on paired antibody sequences. It uses <cls><cls> (two consecutive
    CLS tokens) as the chain separator.

    Model: brineylab/ft-ESM

    Note: Unlike BERT-based models, ESM uses single-character amino acid tokens,
    so no spacing is needed between residues in _format_for_model.

    Attributes:
        model_name: "ftesm"
        supports_paired: True
        max_length: 1024
        embedding_dim: 1280
        mask_token: "<mask>"
        separator: "<cls><cls>"
    """

    model_name = "ftesm"
    supports_paired = True
    max_length = 1024
    embedding_dim = 1280
    mask_token = "<mask>"
    separator = "<cls><cls>"
    has_mlm_head = True

    MODEL_ID = "brineylab/ft-ESM"

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: str | int | list | torch.device | None = None,
    ) -> None:
        """
        Initialize ft-ESM model.

        Args:
            device: (DEPRECATED) Device for inference. Use 'devices' instead.
            devices: Device(s) for inference. Auto-selects all GPUs if None.
        """
        super().__init__(device=device, devices=devices)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = EsmForMaskedLM.from_pretrained(self.MODEL_ID)
        self._model = self._model.to(self._primary_device)
        self._model.eval()

    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """
        Format sequences for ft-ESM tokenization.

        ft-ESM (ESM2-based) uses single-character amino acid tokens,
        so no spacing is needed between residues. For paired sequences,
        chains are separated by <cls><cls> (two CLS tokens).
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

            # Join with <cls><cls> separator for paired sequences
            formatted.append(self.separator.join(parts))

        return formatted

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
        """
        Compute token offsets for each chain.

        For ft-ESM, the separator is <cls><cls> which appears as two
        consecutive CLS token IDs in the tokenized sequence.
        """
        offsets = []
        input_ids = tokenized["input_ids"]

        cls_token_id = self._tokenizer.cls_token_id

        for idx, seq in enumerate(sequences):
            seq_offsets = {}
            tokens = input_ids[idx].tolist()

            # Find positions of consecutive CLS tokens (the separator)
            # The first token is always CLS, so we look for pairs starting after position 0
            separator_start = None
            for i in range(1, len(tokens) - 1):
                if tokens[i] == cls_token_id and tokens[i + 1] == cls_token_id:
                    separator_start = i
                    break

            # Start after the initial [CLS] token
            start = 1

            if seq.heavy_chain is not None:
                heavy_len = seq.length.get("heavy", 0)
                if separator_start is not None:
                    # Heavy chain ends at the first CLS of the separator
                    seq_offsets["heavy"] = (
                        start,
                        min(separator_start, start + heavy_len),
                    )
                else:
                    # Single chain - ends at calculated length
                    seq_offsets["heavy"] = (start, start + heavy_len)

                if seq.light_chain is not None and separator_start is not None:
                    # Light chain starts after the <cls><cls> separator
                    light_start = separator_start + 2
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
            outputs = self._model.esm(
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
            outputs = self._model.esm(
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
            outputs = self._model.esm(
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
        cls_token_id = self._tokenizer.cls_token_id
        eos_token_id = self._tokenizer.eos_token_id

        for i in range(1, len(tokens) - 1):
            # Skip separator tokens (CLS used as separator) and EOS
            if tokens[i] == cls_token_id or tokens[i] == eos_token_id:
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
        # ESM uses <cls> at start and <eos> at end
        decoded = (
            decoded.replace("<cls>", " <cls> ")
            .replace("<eos>", "")
            .replace("<pad>", "")
        )
        decoded = decoded.strip()

        # Split by the <cls><cls> separator (which will appear as " <cls>  <cls> " after our replace)
        # or we can look for remaining <cls> markers
        parts = []
        current_part = []
        tokens = decoded.split()
        cls_count = 0

        for token in tokens:
            if token == "<cls>":
                cls_count += 1
                if cls_count >= 2:
                    # We've hit the separator, save current part and reset
                    if current_part:
                        parts.append("".join(current_part))
                        current_part = []
                    cls_count = 0
            else:
                if cls_count == 1:
                    # Single CLS at start, reset count
                    cls_count = 0
                current_part.append(token)

        if current_part:
            parts.append("".join(current_part))

        # Clean up parts (remove any remaining special tokens and spaces)
        parts = [p.replace(" ", "") for p in parts if p.strip()]

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
        cls_token_id = self._tokenizer.cls_token_id
        eos_token_id = self._tokenizer.eos_token_id
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
            for i in range(1, seq_len - 1):  # Skip [CLS] and [EOS]
                # Skip separator tokens (CLS used as separator) and EOS
                if tokens[i] != cls_token_id and tokens[i] != eos_token_id:
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
