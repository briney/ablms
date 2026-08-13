"""AntiBERTy encoder model wrapper."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ablms.core.encoder import EncoderAbLM
from ablms.core.sequence import AntibodySequence
from ablms.exceptions import ModelLoadError
from ablms.outputs import MaskScanOutput
from ablms.utils.compat import ensure_post_init_called, repair_bert_tokenizer


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

        # AntiBERTy has two transformers 5.x incompatibilities, both upstream.
        # The first must be repaired before construction, because it raises
        # inside `from_pretrained`: its model class calls the pre-4.6
        # `init_weights()` instead of `post_init()`, and only `post_init()`
        # populates `all_tied_weights_keys`. See ablms.utils.compat, issue #5.
        from antiberty.AntiBERTyRunner import VOCAB_FILE
        from antiberty.model.AntiBERTy import AntiBERTy

        ensure_post_init_called(AntiBERTy)

        self._runner = AntiBERTyRunner()

        # The second is silent: the runner builds its tokenizer with the
        # transformers 4 `vocab_file=` keyword, which 5.x ignores, collapsing
        # every residue to `[UNK]`. Left alone this still returns correctly
        # shaped embeddings, which is why it went unnoticed - the values are
        # simply meaningless.
        self._runner.tokenizer = repair_bert_tokenizer(
            self._runner.tokenizer,
            vocab_file=VOCAB_FILE,
            probe_tokens=["E", "V", "Q", "G"],
            do_lower_case=False,
        )

        # Move model to device
        self._runner.model = self._runner.model.to(self._primary_device)
        self._runner.model.eval()
        self._model = self._runner.model
        self._tokenizer = self._runner.tokenizer

    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """
        Format sequences for AntiBERTy as whitespace-separated residues.

        AntiBERTy's vocabulary is per-residue, and its tokenizer is a WordPiece
        model, so residues must be whitespace-delimited. An unspaced sequence is
        treated as a single unknown word and collapses to `[CLS] [UNK] [SEP]` -
        three tokens regardless of input, which is silent rather than fatal.
        This mirrors what `antiberty`'s own runner does before tokenizing.

        Masks are emitted as the vocabulary's `[MASK]`; the class-level
        `mask_token` ("_") is only the single-character stand-in used inside the
        unformatted string, and is itself absent from the vocabulary.

        Args:
            sequences: Sequences to format.

        Returns:
            One whitespace-separated residue string per input sequence.
        """
        formatted = []
        for seq in sequences:
            # Collapse the multi-character unified mask to the single-character
            # stand-in first, so one mask maps to exactly one token below.
            sequence = seq.primary_chain.replace(
                AntibodySequence.MASK_TOKEN, self.mask_token
            )
            residues = [
                "[MASK]" if residue == self.mask_token else residue
                for residue in sequence
            ]
            formatted.append(" ".join(residues))

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

        logits = self._mlm_logits(outputs)
        attention_mask = tokenized.get("attention_mask")

        return logits, attention_mask

    @staticmethod
    def _mlm_logits(outputs: object) -> torch.Tensor:
        """
        Extract MLM logits from an AntiBERTy forward pass.

        AntiBERTy returns an `AntiBERTyOutput`, which names its masked-LM head
        `prediction_logits` rather than the `logits` most HuggingFace heads use -
        it carries three further heads (species, chain, graft) alongside it.

        Indexing the output positionally is not a workaround: `AntiBERTyOutput`
        sets `loss` to the integer `0`, not `None`, when no labels are supplied,
        so `outputs[0]` returns that `0` and the failure surfaces later as
        `'int' object is not subscriptable`.

        Args:
            outputs: The model's return value.

        Returns:
            Logits of shape `(batch, tokens, vocab)`.

        Raises:
            AttributeError: If neither field is present, which would mean the
                upstream output type changed again.
        """
        for field in ("prediction_logits", "logits"):
            logits = getattr(outputs, field, None)
            if logits is not None:
                return logits
        raise AttributeError(
            "AntiBERTy output exposes neither `prediction_logits` nor `logits`; "
            f"got fields {sorted(vars(outputs))}. The upstream output type has "
            "changed."
        )

    def _get_vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping."""
        return self._tokenizer.get_vocab()

    def _compute_pseudo_ll(self, sequence: AntibodySequence) -> float:
        """Compute pseudo log-likelihood for a single sequence."""
        formatted = self._format_for_model([sequence])[0]
        tokens = self._tokenizer.encode(formatted, add_special_tokens=True)

        total_ll = 0.0
        # `self.mask_token` ("_") is not in AntiBERTy's vocabulary; the
        # tokenizer's own `[MASK]` id is the correct one.
        mask_token_id = self._tokenizer.mask_token_id

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
                logits = self._mlm_logits(outputs)[0, i]
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
        # `self.mask_token` ("_") is not in AntiBERTy's vocabulary; the
        # tokenizer's own `[MASK]` id is the correct one.
        mask_token_id = self._tokenizer.mask_token_id

        for seq in sequences:
            formatted = self._format_for_model([seq])[0]
            tokenized = self._tokenize([formatted])

            with torch.no_grad():
                outputs = self._model(**tokenized)
                logits = self._mlm_logits(outputs)[0]

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
        # `self.mask_token` ("_") is not in AntiBERTy's vocabulary; the
        # tokenizer's own `[MASK]` id is the correct one.
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
                    output_logits = self._mlm_logits(outputs)

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
