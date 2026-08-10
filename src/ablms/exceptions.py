"""Custom exceptions for the ablms package."""


class AbLMsError(Exception):
    """Base exception for all ablms errors."""

    pass


class ValidationError(AbLMsError):
    """Raised when input validation fails."""

    pass


class InvalidSequenceError(ValidationError):
    """Raised when an antibody sequence is invalid."""

    pass


class InvalidAminoAcidError(InvalidSequenceError):
    """Raised when a sequence contains invalid amino acid characters."""

    pass


class SequenceTooLongError(ValidationError):
    """Raised when a sequence exceeds the model's maximum length."""

    pass


class PairedSequenceError(ValidationError):
    """Raised for errors related to paired sequence handling."""

    pass


class UnsupportedOperationError(AbLMsError):
    """Raised when an operation is not supported by a model."""

    pass


class ModelNotFoundError(AbLMsError):
    """Raised when a requested model cannot be found."""

    pass


class ModelLoadError(AbLMsError):
    """Raised when a model fails to load."""

    pass


class TokenizationError(AbLMsError):
    """Raised when tokenization fails."""

    pass


class MaskError(AbLMsError):
    """Raised for errors related to mask token handling."""

    pass


class DeviceError(AbLMsError):
    """Raised for errors related to device (CPU/GPU) handling."""

    pass


class MultiGPUError(AbLMsError):
    """Raised for errors in multi-GPU processing."""

    pass


class WorkerError(MultiGPUError):
    """Raised when a worker process fails."""

    def __init__(self, worker_id: int, original_error: Exception):
        self.worker_id = worker_id
        self.original_error = original_error
        super().__init__(
            f"Worker {worker_id} failed: {type(original_error).__name__}: {original_error}"
        )


class WorkerInitializationError(MultiGPUError):
    """Raised when workers fail to initialize."""

    pass


class SharedMemoryError(MultiGPUError):
    """Raised when workers appear unable to return results via shared memory."""

    pass
