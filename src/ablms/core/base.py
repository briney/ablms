"""Base class for all antibody language models."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from ablms.core.sequence import AntibodySequence
from ablms.exceptions import (
    PairedSequenceError,
    SequenceTooLongError,
    ValidationError,
)
from ablms.parallel.utils import get_device_info, resolve_devices, resolve_single_device

if TYPE_CHECKING:
    from ablms.parallel.executor import MultiGPUExecutor


class BaseAbLM(ABC):
    """
    Abstract base class for all antibody language models.

    Provides common functionality for device management, input normalization,
    and validation. All model implementations should inherit from this class
    (via EncoderAbLM or GenerativeAbLM).

    Attributes:
        model_name: Name identifier for the model.
        supports_paired: Whether the model supports paired sequences.
        max_length: Maximum sequence length supported.
        mask_token: Model-specific mask token string.
        separator: Token used to separate chains (for paired models).
        device: Primary PyTorch device for model inference.
        devices: List of all devices used by the model.
        num_devices: Number of devices used by the model.
    """

    model_name: str = "base"
    supports_paired: bool = False
    max_length: int = 512
    mask_token: str | None = None
    separator: str | None = None
    embedding_dim: int = 768

    def __init__(
        self,
        device: str | torch.device | None = None,
        devices: (
            str | int | list[int | str] | torch.device | list[torch.device] | None
        ) = None,
    ) -> None:
        """
        Initialize the base model.

        Args:
            device: (DEPRECATED) Single device for model inference.
                Use 'devices' parameter instead.
            devices: Device(s) for model inference. Options:
                - None: Auto-detect all available GPUs
                - Single device: "cuda:0", 0, torch.device("cuda:0")
                - Multiple devices: [0, 1, 2], ["cuda:0", "cuda:1"]
                - "cpu": CPU only

        Note:
            If both device and devices are specified, devices takes precedence.
        """
        # Handle backward compatibility
        if devices is None and device is not None:
            warnings.warn(
                "The 'device' parameter is deprecated. Use 'devices' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            devices = device

        self._devices = resolve_devices(devices)
        self._primary_device = self._devices[0]
        self._is_multi_gpu = len(self._devices) > 1
        self._executor: MultiGPUExecutor | None = None
        # Deliberately `Any`: concrete model and tokenizer types come from
        # HuggingFace and are dynamically shaped. Subclasses reach for
        # attributes like `_model.bert` and `_tokenizer.sep_token_id` that no
        # static `PreTrainedModel` type declares.
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def device(self) -> torch.device:
        """Get the primary device (for backward compatibility)."""
        return self._primary_device

    @property
    def devices(self) -> list[torch.device]:
        """Get all devices used by this model."""
        return self._devices.copy()

    @property
    def num_devices(self) -> int:
        """Get the number of devices."""
        return len(self._devices)

    def to(self, device: str | torch.device) -> BaseAbLM:
        """
        Move model to a specific device.

        Note: When using multi-GPU, this reconfigures for single-device mode
        and shuts down any existing worker processes.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        # Shutdown existing executor if present
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        # Reset to single-device mode
        self._devices = [resolve_single_device(device)]
        self._primary_device = self._devices[0]
        self._is_multi_gpu = False

        if self._model is not None:
            self._model = self._model.to(self._primary_device)

        return self

    def cpu(self) -> BaseAbLM:
        """Move model to CPU."""
        return self.to("cpu")

    def cuda(self, device_id: int = 0) -> BaseAbLM:
        """Move model to CUDA device."""
        return self.to(f"cuda:{device_id}")

    def _get_executor(self) -> MultiGPUExecutor:
        """
        Get or create the multi-GPU executor.

        Returns:
            MultiGPUExecutor instance for this model.
        """
        if self._executor is None:
            from ablms.parallel.executor import MultiGPUExecutor

            self._executor = MultiGPUExecutor(
                model_class=self.__class__,
                model_init_kwargs=self._get_init_kwargs(),
                devices=self._devices,
            )
        return self._executor

    def _get_init_kwargs(self) -> dict[str, Any]:
        """
        Get the kwargs needed to reconstruct this model in a worker.

        Subclasses should override if they have additional init parameters
        beyond 'devices' that need to be passed to worker model instances.

        Returns:
            Dictionary of initialization kwargs (excluding 'devices').
        """
        return {}

    def _normalize_input(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
    ) -> list[AntibodySequence]:
        """
        Normalize input to a list of AntibodySequence objects.

        Args:
            sequences: Input sequences in various formats.

        Returns:
            List of AntibodySequence objects.
        """
        # Single string
        if isinstance(sequences, str):
            return [AntibodySequence(heavy=sequences)]

        # Single AntibodySequence
        if isinstance(sequences, AntibodySequence):
            return [sequences]

        # List of inputs
        if isinstance(sequences, list):
            if not sequences:
                return []

            # List of strings
            if isinstance(sequences[0], str):
                return [AntibodySequence(heavy=s) for s in sequences]

            # List of AntibodySequence
            if isinstance(sequences[0], AntibodySequence):
                return sequences

        raise ValidationError(
            f"Invalid input type: {type(sequences)}. Expected str, "
            "AntibodySequence, List[str], or List[AntibodySequence]."
        )

    def _validate_input(self, sequences: list[AntibodySequence]) -> None:
        """
        Validate input sequences for this model.

        Args:
            sequences: List of AntibodySequence objects to validate.

        Raises:
            PairedSequenceError: If paired sequences provided but not supported.
            SequenceTooLongError: If sequence exceeds max length.
        """
        for seq in sequences:
            # Check paired support
            if seq.is_paired and not self.supports_paired:
                raise PairedSequenceError(
                    f"Model '{self.model_name}' does not support paired sequences. "
                    "Provide heavy_chain or light_chain only, not both."
                )

            # Check max length
            total_len = seq.total_length
            if self.supports_paired and seq.is_paired:
                # Account for separator token(s)
                sep_len = len(self.separator) if self.separator else 1
                total_len += sep_len + 2  # +2 for special tokens

            if total_len > self.max_length:
                raise SequenceTooLongError(
                    f"Sequence length ({total_len}) exceeds maximum "
                    f"({self.max_length}) for model '{self.model_name}'"
                )

    @abstractmethod
    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """
        Format sequences for model-specific tokenization.

        Converts unified <MASK> tokens to model-specific mask tokens,
        adds separators for paired sequences, etc.

        Args:
            sequences: List of AntibodySequence objects.

        Returns:
            List of formatted strings ready for tokenization.
        """
        pass

    @abstractmethod
    def _tokenize(self, formatted_sequences: list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenize formatted sequences.

        Args:
            formatted_sequences: List of formatted sequence strings.

        Returns:
            Dictionary of tokenized tensors (input_ids, attention_mask, etc.).
        """
        pass

    def _compute_token_offsets(
        self,
        sequences: list[AntibodySequence],
        tokenized: dict[str, torch.Tensor],
    ) -> list[dict[str, tuple[int, int]]]:
        """
        Compute token offsets for each chain in the tokenized sequences.

        This method should be overridden by subclasses that need custom
        offset computation logic.

        Args:
            sequences: Original AntibodySequence objects.
            tokenized: Tokenized tensors from _tokenize().

        Returns:
            List of dictionaries mapping chain names to (start, end) positions.
        """
        # Default implementation - subclasses should override
        offsets = []
        for seq in sequences:
            seq_offsets = {}

            # Simple offset computation assuming 1 token per amino acid
            # plus special tokens at start/end
            start = 1  # Skip CLS/BOS token

            if seq.heavy_chain is not None:
                h_len = seq.length.get("heavy", 0)
                seq_offsets["heavy"] = (start, start + h_len)
                start += h_len

            if self.separator and seq.is_paired:
                start += 1  # Skip separator

            if seq.light_chain is not None:
                l_len = seq.length.get("light", 0)
                seq_offsets["light"] = (start, start + l_len)

            offsets.append(seq_offsets)

        return offsets

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    def __repr__(self) -> str:
        """Return a string representation."""
        device_str = get_device_info(self._devices)
        return f"{self.__class__.__name__}({device_str})"

    def __del__(self):
        """Cleanup on garbage collection."""
        if hasattr(self, "_executor") and self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
