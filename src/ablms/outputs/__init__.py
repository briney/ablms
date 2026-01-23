"""Output dataclasses for ablms models."""

from ablms.outputs.embedding import EmbeddingOutput
from ablms.outputs.logits import LogitsOutput
from ablms.outputs.attention import AttentionOutput
from ablms.outputs.generation import GenerationOutput

__all__ = [
    "EmbeddingOutput",
    "LogitsOutput",
    "AttentionOutput",
    "GenerationOutput",
]
