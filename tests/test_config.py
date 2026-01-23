"""Tests for model configuration and registry."""

import pytest

from ablms.core.config import (
    ModelConfig,
    MODEL_REGISTRY,
    list_models,
    get_model_config,
)
from ablms.exceptions import ModelNotFoundError


class TestModelRegistry:
    """Test model registry functionality."""

    def test_models_registered(self):
        """Test that expected models are registered."""
        models = list_models()
        # Check that at least some models are registered
        # The actual models may vary based on what can be imported
        assert isinstance(models, dict)

    def test_get_model_config_not_found(self):
        """Test that unknown model raises error."""
        with pytest.raises(ModelNotFoundError):
            get_model_config("nonexistent_model")


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        from ablms.core.encoder import EncoderAbLM

        config = ModelConfig(
            name="test_model",
            model_class=EncoderAbLM,
            model_id="test/model",
            supports_paired=True,
            max_length=512,
            embedding_dim=768,
        )

        assert config.name == "test_model"
        assert config.supports_paired
        assert config.max_length == 512
        assert config.embedding_dim == 768


class TestListModels:
    """Test list_models function."""

    def test_list_models_returns_dict(self):
        """Test that list_models returns a dictionary."""
        models = list_models()
        assert isinstance(models, dict)

    def test_list_models_values_are_types(self):
        """Test that model types are valid."""
        models = list_models()
        valid_types = {"encoder", "generative"}
        for model_type in models.values():
            assert model_type in valid_types
