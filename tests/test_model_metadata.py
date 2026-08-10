"""Checks that declared model metadata matches the real checkpoints.

Most models declare their identity twice - once as class attributes on the
encoder and once in `MODEL_REGISTRY` - and neither declaration is derived from
the weights. Nothing forced the two to agree with each other, or with the
checkpoint, and both had drifted:

- `IgBERT` declared `embedding_dim = 768`, but `Exscientia/IgBert` is 1024-wide
  with 30 layers. Real forward passes returned 1024-dim tensors regardless,
  which is why it stayed invisible: only the declared attribute and the
  empty-input tensor shapes were ever wrong.
- The registry pointed `balm` at `BALM/BALM-paired`, which does not exist. The
  class loads `brineylab/BALM-paired`, so loading worked and only the metadata
  was wrong.

These tests close the gap from both sides. The consistency tests are free and
always run; they alone would have caught the `balm` model id. The checkpoint
test needs only each model's `config.json`, never its weights.

A repository that does not exist is a **failure**, not a skip - a skip is what
let the `balm` model id hide in the first place. Only genuine connectivity
problems skip.
"""

from __future__ import annotations

import json

import pytest

from ablms.core.config import MODEL_REGISTRY

# Models distributed as pip packages rather than HuggingFace repos. Their
# weights ship inside the package, so there is no config.json to read.
NON_HUGGINGFACE_IDS = {"ablang", "ablang2", "antiberty", "iglm"}


def _registered_with_dim() -> list[tuple[str, object]]:
    """Registry entries that declare an embedding dimension."""
    return [
        (name, config)
        for name, config in sorted(MODEL_REGISTRY.items())
        if config.embedding_dim is not None
    ]


def _hidden_size(config: dict) -> int | None:
    """Read the hidden size from a HuggingFace config across architectures."""
    # T5 spells it d_model; ESM, BERT, and RoFormer use hidden_size.
    for key in ("hidden_size", "d_model"):
        if config.get(key) is not None:
            return config[key]
    return None


def _class_model_id(model_class: type) -> str | None:
    """The model id an encoder class actually loads, if it declares one."""
    return getattr(model_class, "MODEL_ID", None)


class TestRegistryMatchesModelClass:
    """The registry and the encoder class must not disagree with each other.

    `load_model()` instantiates the class, which uses its own attributes, so the
    registry is documentation. Documentation that disagrees with the code is
    how someone ends up debugging the wrong model.
    """

    @pytest.mark.parametrize(
        "name,config",
        _registered_with_dim(),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_embedding_dim_agrees(self, name, config):
        """A model's two declarations of embedding_dim must be the same number.

        Skipped for models that set embedding_dim per instance rather than as a
        class attribute - ESM2 picks its dimension from `model_id`, so the class
        attribute is only an inherited default.
        """
        if "embedding_dim" not in vars(config.model_class):
            pytest.skip(
                f"{config.model_class.__name__} sets embedding_dim per instance"
            )

        assert config.embedding_dim == config.model_class.embedding_dim, (
            f"{name}: MODEL_REGISTRY says {config.embedding_dim} but "
            f"{config.model_class.__name__}.embedding_dim is "
            f"{config.model_class.embedding_dim}"
        )

    @pytest.mark.parametrize(
        "name,config",
        sorted(MODEL_REGISTRY.items()),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_model_id_agrees(self, name, config):
        """The registry must name the same checkpoint the class loads."""
        class_id = _class_model_id(config.model_class)
        if class_id is None:
            pytest.skip(
                f"{config.model_class.__name__} declares no MODEL_ID "
                f"(package-distributed, or selects one per instance)"
            )

        assert config.model_id == class_id, (
            f"{name}: MODEL_REGISTRY points at {config.model_id!r} but "
            f"{config.model_class.__name__}.MODEL_ID is {class_id!r}"
        )


class TestDeclaredDimMatchesCheckpoint:
    """The declared dimension must match the weights users actually load."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "name,config",
        _registered_with_dim(),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_embedding_dim_matches_hub_config(self, name, config):
        """Compare embedding_dim against the checkpoint's own config.json.

        Downloads a few kilobytes of JSON per model, never the weights.
        """
        if config.model_id in NON_HUGGINGFACE_IDS:
            pytest.skip(f"{name} is not a HuggingFace repo")

        hub = pytest.importorskip("huggingface_hub")

        try:
            path = hub.hf_hub_download(config.model_id, "config.json")
        except (
            hub.errors.RepositoryNotFoundError,
            hub.errors.EntryNotFoundError,
            hub.errors.HFValidationError,
        ) as exc:
            # The declared id is wrong. Skipping here is what let the `balm`
            # model id go unnoticed, so this must fail loudly.
            pytest.fail(
                f"{name} declares model_id={config.model_id!r}, which the Hub "
                f"cannot resolve: {type(exc).__name__}"
            )
        except Exception as exc:  # offline, timeout, auth
            pytest.skip(f"could not reach the Hub for {config.model_id}: {exc}")

        actual = _hidden_size(json.load(open(path)))
        if actual is None:
            pytest.skip(f"{config.model_id} config exposes no hidden size")

        assert config.embedding_dim == actual, (
            f"{name} declares embedding_dim={config.embedding_dim} but "
            f"{config.model_id} has hidden size {actual}"
        )


class TestUnverifiableModelsAreAccountedFor:
    """Guard the skip list against silently swallowing a new model."""

    def test_skip_list_matches_the_registry(self):
        """Every package-distributed id in the registry must be a known one.

        Without this, adding a model whose id happens to lack a slash would
        quietly opt it out of the checkpoint test above.
        """
        package_ids = {
            config.model_id
            for config in MODEL_REGISTRY.values()
            if "/" not in config.model_id
        }
        assert package_ids == NON_HUGGINGFACE_IDS
