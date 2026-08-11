"""
ablms - Unified Antibody Language Model API

A Python package providing a consistent, unified API for diverse antibody
language models with different architectures, tokenizers, and input/output formats.

Example usage:
    >>> from ablms import AntibodySequence, load_model
    >>>
    >>> # Create sequences
    >>> seq = AntibodySequence(
    ...     heavy="EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH...",
    ...     light="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN..."
    ... )
    >>>
    >>> # Load encoder model
    >>> model = load_model("igbert")
    >>>
    >>> # Get embeddings
    >>> embeddings = model.get_embeddings([seq])
    >>> print(embeddings.shape)  # [1, seq_len, 768]
"""

__version__ = "0.0.1"

# Core classes
# Base classes
from ablms.core.base import BaseAbLM

# Configuration and loading
from ablms.core.config import (
    MODEL_REGISTRY,
    ModelConfig,
    list_models,
    load_model,
)
from ablms.core.encoder import EncoderAbLM
from ablms.core.generative import GenerativeAbLM
from ablms.core.sequence import AntibodySequence, ChainType, Species

# Encoder models
from ablms.encoders import (
    BALM,
    ESM2,
    AbLang,
    AbLang2,
    AntiBERTa2,
    AntiBERTy,
    FtESM,
    IgBERT,
    IgT5,
)

# Exceptions
from ablms.exceptions import (
    AbLMsError,
    DeviceError,
    InvalidAminoAcidError,
    InvalidSequenceError,
    MaskError,
    ModelLoadError,
    ModelNotFoundError,
    PairedSequenceError,
    SequenceTooLongError,
    SharedMemoryError,
    TokenizationError,
    UnsupportedOperationError,
    ValidationError,
)

# Generative models
from ablms.generators import IgLM

# Output classes
from ablms.outputs import (
    AttentionOutput,
    EmbeddingOutput,
    GenerationOutput,
    LogitsOutput,
    MaskScanOutput,
)

__all__ = [
    # Version
    "__version__",
    # Core sequence
    "AntibodySequence",
    "ChainType",
    "Species",
    # Base classes
    "BaseAbLM",
    "EncoderAbLM",
    "GenerativeAbLM",
    # Output classes
    "EmbeddingOutput",
    "LogitsOutput",
    "AttentionOutput",
    "GenerationOutput",
    "MaskScanOutput",
    # Configuration
    "ModelConfig",
    "load_model",
    "list_models",
    "MODEL_REGISTRY",
    # Exceptions
    "AbLMsError",
    "ValidationError",
    "InvalidSequenceError",
    "InvalidAminoAcidError",
    "SequenceTooLongError",
    "PairedSequenceError",
    "UnsupportedOperationError",
    "ModelNotFoundError",
    "ModelLoadError",
    "TokenizationError",
    "MaskError",
    "DeviceError",
    "SharedMemoryError",
    # Encoder models
    "IgBERT",
    "IgT5",
    "AntiBERTa2",
    "BALM",
    "AntiBERTy",
    "AbLang2",
    "AbLang",
    "FtESM",
    "ESM2",
    # Generative models
    "IgLM",
]
