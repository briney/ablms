"""Signature-conformance tests for `_process_*_batch` overrides.

`MultiGPUExecutor` looks its target up by name (`getattr(model, method_name)`)
and forwards every extra argument through `**method_kwargs`. Nothing checks that
a subclass override still accepts what the base class promised, so a parameter
added to the base method - such as `pooling` on
`EncoderAbLM._process_embeddings_batch` - silently breaks any model that
overrides it. The failure only appears at inference time, and only for that one
model.

These tests use `inspect.signature` alone, so they run without model weights or
any optional model package installed.
"""

from __future__ import annotations

import inspect

import pytest

from ablms.core.base import BaseAbLM
from ablms.core.encoder import EncoderAbLM
from ablms.core.generative import GenerativeAbLM
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
from ablms.generators import IgLM

BASE_CLASSES = (BaseAbLM, EncoderAbLM, GenerativeAbLM)

MODEL_CLASSES = (
    IgBERT,
    IgT5,
    AntiBERTa2,
    BALM,
    AntiBERTy,
    AbLang2,
    AbLang,
    FtESM,
    ESM2,
    IgLM,
)


def _is_batch_method(name: str) -> bool:
    return name.startswith("_process_") and name.endswith("_batch")


def _base_definition(model_class: type, name: str):
    """Return the base-class function `name` overrides, or None if it is new."""
    for base in model_class.__mro__[1:]:
        if base in BASE_CLASSES and name in vars(base):
            return vars(base)[name]
    return None


def _overrides(model_class: type) -> list[tuple[str, object, object]]:
    """List (name, base_function, override_function) for overridden batch methods."""
    found = []
    for name, override in vars(model_class).items():
        if not _is_batch_method(name) or not callable(override):
            continue
        base = _base_definition(model_class, name)
        if base is not None:
            found.append((name, base, override))
    return found


def assert_bind_compatible(base_func, override_func) -> None:
    """Assert `override_func` accepts every call the executor can make.

    The executor calls `method(batch, **method_kwargs)` on the bound method, so
    the base signature's first parameter after `self` arrives positionally and
    everything else arrives by keyword.
    """
    base_params = list(inspect.signature(base_func).parameters.values())[1:]
    keyword_args = {param.name: None for param in base_params[1:]}

    # `None` stands in for `self` and for the sequences batch; bind() only
    # inspects the signature, it never calls the function.
    inspect.signature(override_func).bind(None, None, **keyword_args)


class TestBatchMethodSignatures:
    """Every `_process_*_batch` override must match its base class's contract."""

    @pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
    def test_overrides_are_bind_compatible(self, model_class):
        for name, base_func, override_func in _overrides(model_class):
            try:
                assert_bind_compatible(base_func, override_func)
            except TypeError as exc:
                pytest.fail(
                    f"{model_class.__name__}.{name} does not accept the "
                    f"arguments the executor forwards for "
                    f"{base_func.__qualname__}: {exc}\n"
                    f"  base:     {inspect.signature(base_func)}\n"
                    f"  override: {inspect.signature(override_func)}"
                )

    def test_at_least_one_override_is_checked(self):
        """Guards the loop above against silently finding nothing to check."""
        checked = [
            f"{cls.__name__}.{name}"
            for cls in MODEL_CLASSES
            for name, _, _ in _overrides(cls)
        ]
        assert "AbLang._process_embeddings_batch" in checked

    def test_ablang_embeddings_batch_accepts_pooling(self):
        """AbLang overrides the one batch method that gained a parameter.

        `EncoderAbLM.get_embeddings` and `iter_embeddings` both pass
        `pooling=` through the executor, including when it is None, so an
        override without the parameter breaks AbLang embeddings entirely.
        """
        inspect.signature(AbLang._process_embeddings_batch).bind(
            None, [], layer=-1, pooling="mean"
        )

    def test_a_dropped_parameter_is_detected(self):
        """The checker must actually reject a narrowed signature."""

        def base(self, sequences, layer=-1, pooling=None):
            pass

        def narrowed(self, sequences, layer=-1):
            pass

        assert_bind_compatible(base, base)
        with pytest.raises(TypeError):
            assert_bind_compatible(base, narrowed)


class TestLayerCountDeclarations:
    """Every encoder must be able to report its layer count."""

    ENCODER_CLASSES = (
        IgBERT,
        IgT5,
        AntiBERTa2,
        BALM,
        AntiBERTy,
        AbLang2,
        AbLang,
        FtESM,
        ESM2,
    )

    @pytest.mark.parametrize("model_class", ENCODER_CLASSES, ids=lambda c: c.__name__)
    def test_num_layers_is_a_property(self, model_class):
        """num_layers must be a property, not an int, so it tracks the checkpoint."""
        resolved = next(
            vars(klass)["num_layers"]
            for klass in model_class.__mro__
            if "num_layers" in vars(klass)
        )
        assert isinstance(resolved, property)

    @pytest.mark.parametrize("model_class", ENCODER_CLASSES, ids=lambda c: c.__name__)
    def test_declares_intermediate_layer_support(self, model_class):
        assert isinstance(model_class.supports_intermediate_layers, bool)

    def test_ablang_declares_no_intermediate_layers(self):
        """AbLang exposes only its final layer; everything else exposes all of them."""
        assert AbLang.supports_intermediate_layers is False
        assert IgBERT.supports_intermediate_layers is True
        assert ESM2.supports_intermediate_layers is True
