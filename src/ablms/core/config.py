"""Model configuration and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ablms.exceptions import ModelNotFoundError


@dataclass
class ModelConfig:
    """
    Configuration for an antibody language model.

    Attributes:
        name: Unique identifier for the model.
        model_class: The model class to instantiate.
        model_id: HuggingFace model ID or package model name.
        supports_paired: Whether the model supports paired sequences.
        max_length: Maximum sequence length.
        embedding_dim: Embedding dimension.
        mask_token: Model-specific mask token.
        separator: Chain separator token (for paired models).
        has_mlm_head: Whether model has MLM head.
        model_type: Type of model ("encoder" or "generative").
        extra_kwargs: Additional model-specific configuration.
    """

    name: str
    model_class: type
    model_id: str
    supports_paired: bool = False
    max_length: int = 512
    embedding_dim: int = 768
    mask_token: str | None = None
    separator: str | None = None
    has_mlm_head: bool = True
    model_type: str = "encoder"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# Global model registry
MODEL_REGISTRY: dict[str, ModelConfig] = {}


def register_model(config: ModelConfig) -> None:
    """
    Register a model configuration.

    Args:
        config: ModelConfig to register.
    """
    MODEL_REGISTRY[config.name] = config


def get_model_config(name: str) -> ModelConfig:
    """
    Get configuration for a registered model.

    Args:
        name: Model name.

    Returns:
        ModelConfig for the model.

    Raises:
        ModelNotFoundError: If model is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ModelNotFoundError(
            f"Model '{name}' not found. Available models: {available}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> dict[str, str]:
    """
    List all registered models.

    Returns:
        Dictionary mapping model names to their types.
    """
    return {name: config.model_type for name, config in MODEL_REGISTRY.items()}


def load_model(
    name: str,
    device: str | torch.device | None = None,
    **kwargs,
):
    """
    Load a registered model by name.

    Args:
        name: Name of the model to load (e.g., "igbert", "antiberty").
        device: Device to load the model on. If None, auto-selects.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Instantiated model ready for use.

    Raises:
        ModelNotFoundError: If model is not registered.

    Example:
        >>> model = load_model("igbert")
        >>> embeddings = model.get_embeddings(["EVQLVESGGGLVQ..."])
    """
    config = get_model_config(name)

    # Merge config extra_kwargs with provided kwargs
    merged_kwargs = {**config.extra_kwargs, **kwargs}

    return config.model_class(device=device, **merged_kwargs)


def _register_all_models() -> None:
    """Register all available models. Called at module import."""
    # Import model classes here to avoid circular imports
    # These imports are deferred until actually needed

    try:
        from ablms.encoders.igbert import IgBERT

        register_model(
            ModelConfig(
                name="igbert",
                model_class=IgBERT,
                model_id="Exscientia/IgBert",
                supports_paired=True,
                max_length=512,
                embedding_dim=1024,
                mask_token="[MASK]",
                separator="[SEP]",
                has_mlm_head=True,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.encoders.igt5 import IgT5

        register_model(
            ModelConfig(
                name="igt5",
                model_class=IgT5,
                model_id="Exscientia/IgT5",
                supports_paired=True,
                max_length=512,
                embedding_dim=1024,
                mask_token=None,  # T5 doesn't use mask token the same way
                separator="</s>",
                has_mlm_head=False,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.encoders.antiberta2 import AntiBERTa2

        register_model(
            ModelConfig(
                name="antiberta2",
                model_class=AntiBERTa2,
                model_id="alchemab/antiberta2",
                supports_paired=True,
                max_length=512,
                embedding_dim=1024,
                mask_token="[MASK]",
                separator="[SEP]",
                has_mlm_head=True,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.encoders.balm import BALM

        register_model(
            ModelConfig(
                name="balm",
                model_class=BALM,
                model_id="brineylab/BALM-paired",
                supports_paired=True,
                max_length=512,
                embedding_dim=1024,
                mask_token="<mask>",
                separator="</s></s>",
                has_mlm_head=True,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.encoders.antiberty import AntiBERTy

        register_model(
            ModelConfig(
                name="antiberty",
                model_class=AntiBERTy,
                model_id="antiberty",
                supports_paired=False,
                max_length=512,
                embedding_dim=512,
                mask_token="_",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.encoders.ablang2 import AbLang2

        register_model(
            ModelConfig(
                name="ablang2",
                model_class=AbLang2,
                model_id="ablang2",
                supports_paired=True,
                max_length=512,
                embedding_dim=480,
                mask_token="*",
                separator="|",
                has_mlm_head=True,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.encoders.ablang import AbLang

        register_model(
            ModelConfig(
                name="ablang",
                model_class=AbLang,
                model_id="ablang",
                supports_paired=False,
                max_length=160,
                embedding_dim=768,
                mask_token="*",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.encoders.ftesm import FtESM

        register_model(
            ModelConfig(
                name="ftesm",
                model_class=FtESM,
                model_id="brineylab/ft-ESM",
                supports_paired=True,
                max_length=1024,
                embedding_dim=1280,
                mask_token="<mask>",
                separator="<cls><cls>",
                has_mlm_head=True,
                model_type="encoder",
            )
        )
    except ImportError:
        pass

    try:
        from ablms.generators.iglm import IgLM

        register_model(
            ModelConfig(
                name="iglm",
                model_class=IgLM,
                model_id="iglm",
                supports_paired=False,
                max_length=512,
                embedding_dim=None,
                mask_token=None,
                separator=None,
                has_mlm_head=False,
                model_type="generative",
            )
        )
    except ImportError:
        pass

    # ESM-2 family models
    try:
        from ablms.encoders.esm2 import ESM2

        register_model(
            ModelConfig(
                name="esm2-8m",
                model_class=ESM2,
                model_id="facebook/esm2_t6_8M_UR50D",
                supports_paired=False,
                max_length=1024,
                embedding_dim=320,
                mask_token="<mask>",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
                extra_kwargs={"model_id": "facebook/esm2_t6_8M_UR50D"},
            )
        )
        register_model(
            ModelConfig(
                name="esm2-35m",
                model_class=ESM2,
                model_id="facebook/esm2_t12_35M_UR50D",
                supports_paired=False,
                max_length=1024,
                embedding_dim=480,
                mask_token="<mask>",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
                extra_kwargs={"model_id": "facebook/esm2_t12_35M_UR50D"},
            )
        )
        register_model(
            ModelConfig(
                name="esm2-150m",
                model_class=ESM2,
                model_id="facebook/esm2_t30_150M_UR50D",
                supports_paired=False,
                max_length=1024,
                embedding_dim=640,
                mask_token="<mask>",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
                extra_kwargs={"model_id": "facebook/esm2_t30_150M_UR50D"},
            )
        )
        register_model(
            ModelConfig(
                name="esm2-650m",
                model_class=ESM2,
                model_id="facebook/esm2_t33_650M_UR50D",
                supports_paired=False,
                max_length=1024,
                embedding_dim=1280,
                mask_token="<mask>",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
                extra_kwargs={"model_id": "facebook/esm2_t33_650M_UR50D"},
            )
        )
        register_model(
            ModelConfig(
                name="esm2-3b",
                model_class=ESM2,
                model_id="facebook/esm2_t36_3B_UR50D",
                supports_paired=False,
                max_length=1024,
                embedding_dim=2560,
                mask_token="<mask>",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
                extra_kwargs={"model_id": "facebook/esm2_t36_3B_UR50D"},
            )
        )
        register_model(
            ModelConfig(
                name="esm2-15b",
                model_class=ESM2,
                model_id="facebook/esm2_t48_15B_UR50D",
                supports_paired=False,
                max_length=1024,
                embedding_dim=5120,
                mask_token="<mask>",
                separator=None,
                has_mlm_head=True,
                model_type="encoder",
                extra_kwargs={"model_id": "facebook/esm2_t48_15B_UR50D"},
            )
        )
    except ImportError:
        pass


# Register models when module is imported
_register_all_models()
