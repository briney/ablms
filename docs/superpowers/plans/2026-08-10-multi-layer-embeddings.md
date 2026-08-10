# Multi-Layer Embedding Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `get_embeddings(layer=...)` accept a single index (as today), an explicit list of indices, or `"all"`, returning a stacked layer axis so callers can concatenate per-layer pooled vectors for UMAP/t-SNE.

**Architecture:** Every HuggingFace-backed encoder already runs its forward pass with `output_hidden_states=True` and discards all but one layer, and every encoder already implements `_forward_all_hidden_states()`. The multi-layer path therefore reuses that existing method and selects from its output — no encoder's forward pass changes. Argument validation resolves the layer spec in the parent process before dispatch; pooling is applied per layer *before* the layers are stacked, so the large `[batch, layers, seq_len, hidden_dim]` tensor is never materialised on pooled runs.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-multi-layer-embeddings-design.md`

## Global Constraints

- Python 3.10+ union syntax. Write `x | None`, never `Optional[x]`; never import `Optional` or `Union` from `typing`.
- Type hints on all function signatures. Google-style docstrings on all public classes and functions.
- Max line length 88 — `pyproject.toml:71,75` configures both `black` and `ruff` at 88, which overrides the 100 in the global style guide. Format with `black src/ tests/` and lint with `ruff check src/ tests/`. Leave pre-existing violations in untouched code alone.
- Run `mypy src/` as well. It is not currently clean (5 pre-existing errors, mostly the repo-wide `self._model` typed as `None` pattern), so the bar is *adding no new errors*, not reaching zero. Narrow `X | None` values explicitly rather than relying on a property check, which mypy cannot use as a type guard.
- Run tests with `python -m pytest`, never bare `pytest` (bare `pytest` resolves to the wrong interpreter in this environment and reports a fake collection failure).
- Tests requiring model weights are marked `@pytest.mark.slow` and use real sequences from `tests/test_embedding_memory.py`, never synthetic data.
- `layer=-1` (the default) must remain byte-identical in value, shape, and `EmbeddingOutput.layer` to the pre-change behaviour. This is the acceptance bar for every task.
- Layer indices address the `hidden_states` tuple: index `0` is the embedding-layer output, index `i` is the output of transformer block `i`. A model with `num_layers = 12` exposes 13 selectable indices.
- The `ablms` test environment does **not** have the `ablang`, `ablang2`, `antiberty`, or `iglm` packages installed. Tests for those models must be signature-level only, or `@pytest.mark.slow` with a skip guard. Pre-existing AbLang test failures are unrelated to this work.

---

### Task 1: Generalize executor padding to any differing axis

`MultiGPUExecutor._pad_tensors_to_max_length` hardcodes "dimension 1 is the sequence axis" and falls through to a 2D padding tensor for any other rank. A token-level multi-layer batch is 4D `[batch, layers, seq_len, hidden_dim]`, so it would build a `[batch, pad]` tensor and `torch.cat` would raise a dimension-mismatch `RuntimeError` — but only when batches pad to different lengths, which is most real datasets. This task fixes that before anything produces 4D results.

**Files:**
- Modify: `src/ablms/parallel/executor.py:516-547`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `MultiGPUExecutor._pad_tensors_to_max_length(tensors: list[torch.Tensor]) -> list[torch.Tensor]` — pads every axis except dim 0 to the maximum across the list. Task 5 depends on this accepting 4D input.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_executor.py`. Note the existing helper at `tests/test_executor.py:105` builds an executor as `MultiGPUExecutor(FakeWorkerModel, {}, devices)` — constructing one does not spawn workers, so these tests are fast and need no `slow` marker.

```python
class TestPadTensorsToMaxLength:
    """Padding must not assume which axis carries sequence length.

    It is dim 1 for [batch, seq, hidden] but dim 2 for the multi-layer
    [batch, layers, seq, hidden]. The rule that covers both: every tensor in a
    group has the same shape except along the axis that varies, so pad whichever
    non-batch axes actually differ. Dim 0 is the concatenation axis.
    """

    @pytest.fixture
    def executor(self):
        return MultiGPUExecutor(FakeWorkerModel, {}, [torch.device("cpu")])

    def test_pads_three_dimensional_on_sequence_axis(self, executor):
        """The pre-existing [batch, seq, hidden] behaviour is unchanged."""
        tensors = [torch.ones(2, 5, 4), torch.ones(3, 7, 4)]
        padded = executor._pad_tensors_to_max_length(tensors)

        assert [tuple(t.shape) for t in padded] == [(2, 7, 4), (3, 7, 4)]
        assert torch.cat(padded, dim=0).shape == (5, 7, 4)
        assert padded[0][:, 5:, :].eq(0).all()

    def test_pads_two_dimensional_mask(self, executor):
        """[batch, seq] attention masks are padded the same way."""
        padded = executor._pad_tensors_to_max_length(
            [torch.ones(2, 5), torch.ones(3, 7)]
        )
        assert [tuple(t.shape) for t in padded] == [(2, 7), (3, 7)]

    def test_pads_four_dimensional_on_sequence_axis(self, executor):
        """[batch, layers, seq, hidden] pads dim 2, leaving the layer axis alone."""
        tensors = [torch.ones(2, 3, 5, 4), torch.ones(1, 3, 7, 4)]
        padded = executor._pad_tensors_to_max_length(tensors)

        assert [tuple(t.shape) for t in padded] == [(2, 3, 7, 4), (1, 3, 7, 4)]
        assert torch.cat(padded, dim=0).shape == (3, 3, 7, 4)
        assert padded[0][:, :, 5:, :].eq(0).all()

    def test_leaves_batch_axis_alone(self, executor):
        """Dim 0 is the concatenation axis and must never be padded."""
        padded = executor._pad_tensors_to_max_length(
            [torch.ones(2, 5, 4), torch.ones(3, 5, 4)]
        )
        assert [tuple(t.shape) for t in padded] == [(2, 5, 4), (3, 5, 4)]

    def test_returns_unchanged_when_nothing_differs(self, executor):
        tensors = [torch.ones(2, 5, 4), torch.ones(2, 5, 4)]
        padded = executor._pad_tensors_to_max_length(tensors)
        assert [tuple(t.shape) for t in padded] == [(2, 5, 4), (2, 5, 4)]

    def test_empty_and_one_dimensional_are_passed_through(self, executor):
        assert executor._pad_tensors_to_max_length([]) == []
        singles = [torch.ones(3), torch.ones(5)]
        assert executor._pad_tensors_to_max_length(singles) is singles
```

Confirm `tests/test_executor.py` already imports `torch`, `pytest`, `MultiGPUExecutor`, and `FakeWorkerModel`; add whichever are missing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_executor.py::TestPadTensorsToMaxLength -v`
Expected: `test_pads_four_dimensional_on_sequence_axis` FAILS — the 4D input hits the `else` branch, which builds a `[batch, pad]` tensor, and `torch.cat` raises `RuntimeError: Tensors must have same number of dimensions`. The 2D and 3D tests should already PASS.

- [ ] **Step 3: Replace the method**

In `src/ablms/parallel/executor.py`, replace `_pad_tensors_to_max_length` (lines 516-547) with:

```python
    def _pad_tensors_to_max_length(
        self, tensors: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Pad tensors so they agree on every axis except the concatenation axis.

        Batches are tokenized independently, so they pad to different sequence
        lengths and cannot be concatenated as-is. Rather than assume which axis
        carries sequence length - dim 1 for [batch, seq, hidden], dim 2 for the
        multi-layer [batch, layers, seq, hidden] - pad whichever non-batch axes
        actually differ across the group. Dimension 0 is the concatenation axis
        and is left alone.

        Args:
            tensors: List of tensors to pad. All must have the same rank.

        Returns:
            List of tensors that differ only along dimension 0.

        Raises:
            ValueError: If the tensors do not all have the same rank.
        """
        if not tensors or tensors[0].dim() < 2:
            return tensors

        ndim = tensors[0].dim()
        if any(t.dim() != ndim for t in tensors):
            ranks = sorted({t.dim() for t in tensors})
            raise ValueError(
                f"Cannot pad tensors of differing rank for concatenation: got "
                f"ranks {ranks}. This usually means a worker returned an "
                f"unexpected shape."
            )

        max_sizes = [max(t.shape[d] for t in tensors) for d in range(ndim)]

        padded = []
        for tensor in tensors:
            # F.pad reads dimensions last-to-first, two entries (before, after)
            # per dimension. Stopping at 1 leaves the batch axis unpadded.
            pad_spec: list[int] = []
            for dim in range(ndim - 1, 0, -1):
                pad_spec.extend([0, max_sizes[dim] - tensor.shape[dim]])

            if any(pad_spec):
                tensor = torch.nn.functional.pad(tensor, pad_spec)
            padded.append(tensor)

        return padded
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_executor.py -v`
Expected: all PASS, including the pre-existing executor tests.

- [ ] **Step 5: Run the regression suite for the shared path**

Every executor caller goes through this method, so check the callers that already exercise ragged batches.

Run: `python -m pytest tests/test_embedding_memory.py tests/test_mask_scan.py -v -m slow`
Expected: PASS (`TestRaggedBatchTokenLevel` is the direct regression guard).

- [ ] **Step 6: Commit**

```bash
git add src/ablms/parallel/executor.py tests/test_executor.py
git commit -m "Pad any differing axis, not just dimension 1"
```

---

### Task 2: Expose the model's layer count

Validation runs in the parent process before dispatch, so the parent needs the layer count. It has one: every encoder loads its model eagerly in `__init__` (`igbert.py:57`, `esm2.py:85`, and the same line in all seven others), so `self._model` is always available. Reading the live checkpoint beats a declared constant because `ESM2` spans six variants from 6 to 48 layers and `FtESM` follows whatever base it was fine-tuned from.

**Files:**
- Modify: `src/ablms/core/encoder.py` (add property + class attribute near `has_mlm_head`, line 30)
- Modify: `src/ablms/encoders/igt5.py` (override)
- Modify: `src/ablms/encoders/ablang2.py` (override)
- Modify: `src/ablms/encoders/ablang.py` (override + flag)
- Test: `tests/test_encoder_contract.py`, `tests/test_embedding_memory.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `EncoderAbLM.num_layers -> int` (property): count of transformer blocks. Selectable indices are `0..num_layers`.
  - `EncoderAbLM.supports_intermediate_layers: bool = True` (class attribute).
  - Task 3 consumes both; Task 5 consumes `num_layers` for its cross-check.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_encoder_contract.py` (signature-level, runs with no weights or optional packages):

```python
class TestLayerCountDeclarations:
    """Every encoder must be able to report its layer count."""

    ENCODER_CLASSES = (IgBERT, IgT5, AntiBERTa2, BALM, AntiBERTy, AbLang2, AbLang, FtESM, ESM2)

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
```

Add to `tests/test_embedding_memory.py` (real forward pass — this is what catches a missing or wrong override):

```python
class TestLayerCount:
    """num_layers must agree with the model's actual hidden_states tuple."""

    @pytest.mark.slow
    def test_matches_forward_pass(self, esm2_cpu, sequences):
        formatted = esm2_cpu._format_for_model(sequences[:1])
        tokenized = esm2_cpu._tokenize(formatted)
        hidden_states, _ = esm2_cpu._forward_all_hidden_states(tokenized)

        assert esm2_cpu.num_layers + 1 == len(hidden_states)

    @pytest.mark.slow
    def test_matches_the_checkpoint_variant(self, esm2_cpu):
        """The t6 checkpoint has 6 blocks; a hardcoded constant would not track this."""
        assert esm2_cpu.num_layers == 6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_encoder_contract.py::TestLayerCountDeclarations -v`
Expected: FAIL with `StopIteration` (no class in the MRO defines `num_layers`) and `AttributeError: type object 'IgBERT' has no attribute 'supports_intermediate_layers'`.

- [ ] **Step 3: Add the base property and flag**

In `src/ablms/core/encoder.py`, directly below `has_mlm_head: bool = True` (line 30):

```python
    # Whether the model can return layers other than its final one
    supports_intermediate_layers: bool = True

    @property
    def num_layers(self) -> int:
        """
        Number of transformer blocks in the loaded model.

        Selectable layer indices run from 0 to num_layers inclusive: index 0 is
        the embedding-layer output and index i is the output of block i. This
        matches HuggingFace, where len(hidden_states) == num_hidden_layers + 1.

        Subclasses must override this if their model object does not expose a
        HuggingFace config with `num_hidden_layers`.

        Returns:
            Count of transformer blocks.
        """
        return self._model.config.num_hidden_layers
```

- [ ] **Step 4: Add the three overrides**

In `src/ablms/encoders/igt5.py`, after the class attribute block (line 41). IgT5 loads `T5EncoderModel` (`igt5.py:63`), and `T5Config` spells the encoder depth `num_layers`:

```python
    @property
    def num_layers(self) -> int:
        """T5Config spells the encoder depth `num_layers`, not `num_hidden_layers`."""
        return self._model.config.num_layers
```

In `src/ablms/encoders/ablang2.py`, after the class attribute block (line 40). `self._model` is AbRep (`ablang2.py:76`), which has no HF config; this is the arithmetic already inlined at `ablang2.py:177`:

```python
    @property
    def num_layers(self) -> int:
        """AbLang2 has no HuggingFace config; count the encoder blocks directly."""
        return len(self._model.encoder_blocks)
```

In `src/ablms/encoders/ablang.py`, after the class attribute block (line 42):

```python
    supports_intermediate_layers = False

    @property
    def num_layers(self) -> int:
        """
        AbRep's depth, per the AbLang paper.

        Hardcoded because AbLang's model object exposes no config. Only the
        final layer is reachable (`_forward_embeddings_with_model` can return
        nothing else), so this value affects only which explicit positive index
        is accepted as "final" and the wording of the resulting error. Confirm
        it against a real forward pass if the `ablang` package is ever installed
        in the test environment.
        """
        return 12
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_encoder_contract.py -v`
Expected: all PASS.

Run: `python -m pytest tests/test_embedding_memory.py::TestLayerCount -v -m slow`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/ablms/core/encoder.py src/ablms/encoders/igt5.py \
        src/ablms/encoders/ablang2.py src/ablms/encoders/ablang.py \
        tests/test_encoder_contract.py tests/test_embedding_memory.py
git commit -m "Expose num_layers and intermediate-layer support on encoders"
```

---

### Task 3: Layer-selection validation helper

A pure function, testable with no model weights. It resolves `int | list[int] | "all"` into either an unchanged `int` (single-layer, preserving `layer=-1` in the output exactly as today) or a list of resolved non-negative indices.

**Files:**
- Create: `src/ablms/utils/layers.py`
- Modify: `src/ablms/utils/__init__.py`
- Create: `tests/test_layer_selection.py`

**Interfaces:**
- Consumes: `EncoderAbLM.num_layers`, `EncoderAbLM.supports_intermediate_layers` (Task 2).
- Produces: `resolve_layer_selection(layer, num_layers, *, model_name, supports_intermediate_layers=True) -> int | list[int]`. Tasks 6 and 7 call it; an `int` return means single-layer output, a `list` means a layer axis.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_selection.py`:

```python
"""Tests for layer-selection resolution."""

from __future__ import annotations

import pytest

from ablms.exceptions import UnsupportedOperationError
from ablms.utils.layers import resolve_layer_selection

# A 12-block model: 13 selectable indices, 0..12.
NUM_LAYERS = 12


def resolve(layer, **kwargs):
    return resolve_layer_selection(
        layer, NUM_LAYERS, model_name="testmodel", **kwargs
    )


class TestSingleLayer:
    """A single int passes through unchanged, preserving today's behaviour."""

    @pytest.mark.parametrize("layer", [-1, 0, 6, 12, -13])
    def test_valid_index_is_returned_unchanged(self, layer):
        assert resolve(layer) == layer

    def test_negative_index_is_not_normalized(self):
        """layer=-1 must stay -1 so EmbeddingOutput.layer reads as it does today."""
        assert resolve(-1) == -1

    @pytest.mark.parametrize("layer", [13, 99, -14])
    def test_out_of_range_raises(self, layer):
        with pytest.raises(ValueError, match="out of range"):
            resolve(layer)

    def test_error_names_the_model_and_the_valid_range(self):
        with pytest.raises(ValueError, match="testmodel"):
            resolve(13)
        with pytest.raises(ValueError, match=r"0\.\.12"):
            resolve(13)


class TestAllLayers:
    def test_all_returns_every_index_ascending(self):
        assert resolve("all") == list(range(13))

    def test_all_is_case_insensitive(self):
        assert resolve("ALL") == list(range(13))

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Invalid layer selection"):
            resolve("last")


class TestLayerList:
    def test_order_is_preserved(self):
        assert resolve([12, 0, 6]) == [12, 0, 6]

    def test_negatives_are_resolved(self):
        assert resolve([0, -1]) == [0, 12]

    def test_single_element_list_keeps_list_form(self):
        """The argument's type decides the output shape, not its length."""
        assert resolve([-1]) == [12]

    def test_tuple_and_range_are_accepted(self):
        assert resolve((0, 6)) == [0, 6]
        assert resolve(range(0, 13, 6)) == [0, 6, 12]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            resolve([])

    def test_duplicates_after_resolution_raise(self):
        """[12, -1] is the form a caller writes by accident."""
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            resolve([12, -1])

    def test_literal_duplicates_raise(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            resolve([6, 6])

    def test_out_of_range_member_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            resolve([0, 13])


class TestRejectedTypes:
    def test_bool_is_rejected(self):
        """bool subclasses int; True must not silently mean layer 1."""
        with pytest.raises(ValueError):
            resolve(True)

    def test_float_is_rejected(self):
        with pytest.raises(ValueError):
            resolve(1.0)

    def test_none_is_rejected(self):
        with pytest.raises(ValueError):
            resolve(None)


class TestFinalLayerOnlyModels:
    """Models like AbLang expose nothing but their final layer."""

    def restricted(self, layer):
        return resolve(layer, supports_intermediate_layers=False)

    @pytest.mark.parametrize("layer", [-1, 12])
    def test_final_layer_is_allowed(self, layer):
        assert self.restricted(layer) == layer

    def test_default_is_allowed(self):
        assert self.restricted(-1) == -1

    @pytest.mark.parametrize("layer", [0, 3, -2])
    def test_other_single_layers_raise(self, layer):
        with pytest.raises(UnsupportedOperationError, match="final layer"):
            self.restricted(layer)

    def test_all_raises(self):
        with pytest.raises(UnsupportedOperationError):
            self.restricted("all")

    def test_multi_element_list_raises(self):
        with pytest.raises(UnsupportedOperationError):
            self.restricted([0, 12])

    def test_final_layer_as_a_list_is_allowed(self):
        assert self.restricted([-1]) == [12]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_layer_selection.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'ablms.utils.layers'`.

- [ ] **Step 3: Write the implementation**

Create `src/ablms/utils/layers.py`:

```python
"""Resolution and validation of layer selections for embedding extraction."""

from __future__ import annotations

from ablms.exceptions import UnsupportedOperationError

ALL_LAYERS = "all"


def resolve_layer_selection(
    layer: int | list[int] | str,
    num_layers: int,
    *,
    model_name: str,
    supports_intermediate_layers: bool = True,
) -> int | list[int]:
    """
    Validate a layer selection and resolve it to concrete indices.

    Indices address the model's hidden_states tuple: index 0 is the
    embedding-layer output and index i is the output of transformer block i, so
    a model with `num_layers` blocks has `num_layers + 1` selectable indices.

    A single int is returned unchanged rather than normalized, so a default
    `layer=-1` call still reports `layer=-1` on its output, exactly as before
    multi-layer support existed. Lists are always resolved to non-negative
    indices, because they are used to index hidden_states directly and are
    reported on `EmbeddingOutput.layers`.

    Args:
        layer: A single index, a list/tuple/range of indices, or "all"
            (case-insensitive) for every layer in ascending order.
        num_layers: Count of transformer blocks in the model.
        model_name: Model name, used in error messages.
        supports_intermediate_layers: False for models that can only produce
            their final layer.

    Returns:
        The int unchanged for a single-layer selection, or a list of resolved
        non-negative indices in the order requested.

    Raises:
        ValueError: If the selection is malformed, empty, out of range, or
            contains duplicates after negative indices are resolved.
        UnsupportedOperationError: If a non-final layer is requested from a
            model with supports_intermediate_layers=False.
    """
    n_selectable = num_layers + 1

    if isinstance(layer, str):
        if layer.lower() != ALL_LAYERS:
            raise ValueError(
                f"Invalid layer selection {layer!r}. Pass an int, a list of "
                f"ints, or {ALL_LAYERS!r}."
            )
        resolved = list(range(n_selectable))
        _check_supported(resolved, n_selectable, model_name, supports_intermediate_layers)
        return resolved

    if isinstance(layer, (list, tuple, range)):
        indices = list(layer)
        if not indices:
            raise ValueError(
                "Layer selection list is empty. Pass at least one index, or "
                f"{ALL_LAYERS!r} for every layer."
            )
        for index in indices:
            _validate_index(index, n_selectable, model_name)

        resolved = [_normalize(index, n_selectable) for index in indices]
        _reject_duplicates(resolved, indices, model_name)
        _check_supported(resolved, n_selectable, model_name, supports_intermediate_layers)
        return resolved

    _validate_index(layer, n_selectable, model_name)
    _check_supported(
        [_normalize(layer, n_selectable)],
        n_selectable,
        model_name,
        supports_intermediate_layers,
    )
    return layer


def _validate_index(index: object, n_selectable: int, model_name: str) -> None:
    """Reject non-ints and out-of-range indices."""
    # bool is a subclass of int; True must not silently mean layer 1.
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(
            f"Layer indices must be ints, got {index!r} "
            f"({type(index).__name__})."
        )
    if not -n_selectable <= index < n_selectable:
        raise ValueError(
            f"Layer {index} is out of range for {model_name}: valid indices are "
            f"0..{n_selectable - 1} or -1..-{n_selectable}. Index 0 is the "
            f"embedding layer and {n_selectable - 1} is the final block."
        )


def _normalize(index: int, n_selectable: int) -> int:
    """Convert a negative index to its non-negative equivalent."""
    return index + n_selectable if index < 0 else index


def _reject_duplicates(
    resolved: list[int], requested: list[int], model_name: str
) -> None:
    """Reject repeated layers, including via mixed negative and positive forms."""
    seen: set[int] = set()
    for position, index in enumerate(resolved):
        if index in seen:
            raise ValueError(
                f"Duplicate layer {index} in selection {requested} for "
                f"{model_name} (entry {requested[position]!r} repeats an "
                f"earlier layer). Each layer may be selected only once."
            )
        seen.add(index)


def _check_supported(
    resolved: list[int],
    n_selectable: int,
    model_name: str,
    supports_intermediate_layers: bool,
) -> None:
    """Reject non-final selections for models that expose only their last layer."""
    if supports_intermediate_layers:
        return

    final = n_selectable - 1
    if resolved != [final]:
        raise UnsupportedOperationError(
            f"{model_name} exposes only its final layer ({final}); requested "
            f"{resolved}. Use layer=-1, or omit the argument."
        )
```

- [ ] **Step 4: Export from the utils package**

In `src/ablms/utils/__init__.py`, add the import and the `__all__` entries:

```python
from ablms.utils.layers import ALL_LAYERS, resolve_layer_selection
```

and add `"ALL_LAYERS"` and `"resolve_layer_selection"` to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_layer_selection.py -v`
Expected: all PASS (about 35 tests).

- [ ] **Step 6: Commit**

```bash
git add src/ablms/utils/layers.py src/ablms/utils/__init__.py tests/test_layer_selection.py
git commit -m "Add layer-selection resolution and validation"
```

---

### Task 4: Multi-layer support in EmbeddingOutput

Pure dataclass work, no model weights needed. The layer axis sits immediately after batch, so a pooled multi-layer result is `[batch, layers, hidden]` and a token-level one is `[batch, layers, seq, hidden]`.

**Files:**
- Modify: `src/ablms/outputs/embedding.py`
- Test: `tests/test_outputs.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, on `EmbeddingOutput`:
  - field `layers: list[int] | None = None` (None means single-layer)
  - field `layer: int | None = -1` (None when `layers` is set)
  - `is_multi_layer -> bool`, `num_layers -> int`, `is_pooled -> bool`
  - `get_layer(layer: int) -> torch.Tensor`, `concat_layers() -> torch.Tensor`
  - Tasks 6 and 7 construct these outputs.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_outputs.py`, inside `class TestEmbeddingOutput`:

```python
    def test_single_layer_output_is_not_multi_layer(self):
        """Defaults are unchanged: no layers field means the old behaviour."""
        output = EmbeddingOutput(embeddings=torch.randn(2, 10, 768))

        assert not output.is_multi_layer
        assert output.num_layers == 1
        assert output.layers is None
        assert output.layer == -1

    def test_multi_layer_token_level_properties(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 10, 768), layer=None, layers=[0, 6, 12]
        )

        assert output.is_multi_layer
        assert output.num_layers == 3
        assert output.batch_size == 2
        assert output.hidden_dim == 768
        assert not output.is_pooled

    def test_multi_layer_pooled_is_detected(self):
        """[batch, layers, hidden] is pooled even though it has three dims."""
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768),
            pooled=torch.randn(2, 3, 768),
            layer=None,
            layers=[0, 6, 12],
        )

        assert output.is_pooled
        assert output.hidden_dim == 768

    def test_get_layer_returns_the_requested_layer(self):
        embeddings = torch.randn(2, 3, 768)
        output = EmbeddingOutput(embeddings=embeddings, layer=None, layers=[0, 6, 12])

        assert torch.equal(output.get_layer(6), embeddings[:, 1])
        assert torch.equal(output.get_layer(0), embeddings[:, 0])

    def test_get_layer_rejects_an_unselected_layer(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768), layer=None, layers=[0, 6, 12]
        )
        with pytest.raises(ValueError, match="not selected"):
            output.get_layer(7)

    def test_get_layer_rejects_single_layer_output(self):
        output = EmbeddingOutput(embeddings=torch.randn(2, 10, 768))
        with pytest.raises(ValueError, match="single layer"):
            output.get_layer(0)

    def test_concat_layers_on_pooled_output(self):
        embeddings = torch.randn(2, 3, 768)
        output = EmbeddingOutput(
            embeddings=embeddings, pooled=embeddings, layer=None, layers=[0, 6, 12]
        )

        concatenated = output.concat_layers()

        assert concatenated.shape == (2, 3 * 768)
        expected = torch.cat([embeddings[:, i] for i in range(3)], dim=-1)
        assert torch.equal(concatenated, expected)

    def test_concat_layers_on_token_level_output(self):
        embeddings = torch.randn(2, 3, 10, 768)
        output = EmbeddingOutput(embeddings=embeddings, layer=None, layers=[0, 6, 12])

        concatenated = output.concat_layers()

        assert concatenated.shape == (2, 10, 3 * 768)
        expected = torch.cat([embeddings[:, i] for i in range(3)], dim=-1)
        assert torch.equal(concatenated, expected)

    def test_concat_layers_rejects_single_layer_output(self):
        output = EmbeddingOutput(embeddings=torch.randn(2, 10, 768))
        with pytest.raises(ValueError, match="single layer"):
            output.concat_layers()

    def test_multi_layer_chain_embeddings_keep_the_layer_axis(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(1, 3, 20, 768),
            token_offsets=[{"heavy": (1, 10), "light": (11, 19)}],
            layer=None,
            layers=[0, 6, 12],
        )

        heavy = output.get_chain_embeddings(0, "heavy")

        assert heavy.shape == (3, 9, 768)

    def test_multi_layer_sequence_tokens_strip_padding(self):
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        output = EmbeddingOutput(
            embeddings=torch.randn(1, 3, 5, 768),
            attention_mask=mask,
            layer=None,
            layers=[0, 6, 12],
        )

        tokens = output.get_sequence_tokens(0)

        assert tokens.shape == (3, 3, 768)

    def test_layers_survive_a_device_move(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768), layer=None, layers=[0, 6, 12]
        )
        assert output.cpu().layers == [0, 6, 12]
        assert output.numpy().layers == [0, 6, 12]

    def test_repr_reports_the_selected_layers(self):
        output = EmbeddingOutput(
            embeddings=torch.randn(2, 3, 768), layer=None, layers=[0, 6, 12]
        )
        assert "layers=[0, 6, 12]" in repr(output)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_outputs.py::TestEmbeddingOutput -v`
Expected: FAIL with `TypeError: EmbeddingOutput.__init__() got an unexpected keyword argument 'layers'`. The four pre-existing tests (`test_basic_properties`, `test_pooled_embeddings`, `test_get_chain_embeddings`, and the rest) must still PASS.

- [ ] **Step 3: Update the dataclass**

In `src/ablms/outputs/embedding.py`:

Add to the class docstring's Attributes section, after the `layer` entry:

```
        layers: Resolved indices of every selected layer when the output carries
            a layer axis, or None for a single-layer output.
```

and change the `layer` entry to:

```
        layer: The layer index from which embeddings were extracted, or None
            when multiple layers were selected (see `layers`).
```

Change the two fields (line 36) to:

```python
    layer: int | None = -1
    layers: list[int] | None = None
```

Add the new properties after `is_pooled` and replace `is_pooled` itself:

```python
    @property
    def is_multi_layer(self) -> bool:
        """Whether the embeddings carry a layer axis at dimension 1."""
        return self.layers is not None

    @property
    def num_layers(self) -> int:
        """Number of layers represented in this output."""
        return len(self.layers) if self.layers is not None else 1

    @property
    def is_pooled(self) -> bool:
        """Check if embeddings are pooled (sequence-level).

        Pooling removes the sequence axis, so a pooled output has one fewer
        dimension than a token-level one at the same layer arity: [batch,
        hidden] against [batch, seq, hidden], and [batch, layers, hidden]
        against [batch, layers, seq, hidden].
        """
        return self.embeddings.ndim == (3 if self.is_multi_layer else 2)

    def get_layer(self, layer: int) -> torch.Tensor:
        """
        Extract a single layer from a multi-layer output.

        Args:
            layer: The model's layer index, as reported in `layers` - not a
                position within the stack.

        Returns:
            The embeddings for that layer, with the layer axis removed:
            [batch, hidden_dim] if pooled, else [batch, seq_len, hidden_dim].

        Raises:
            ValueError: If this output holds a single layer, or if the
                requested layer was not among those selected.
        """
        if self.layers is None:
            raise ValueError(
                "This output holds a single layer; use .embeddings directly."
            )
        if layer not in self.layers:
            raise ValueError(
                f"Layer {layer} was not selected. Available layers: {self.layers}"
            )
        return self.embeddings[:, self.layers.index(layer)]

    def concat_layers(self) -> torch.Tensor:
        """
        Fold the layer axis into the hidden dimension.

        This is the form used for dimensionality reduction over every layer at
        once, where each sequence becomes one long feature vector.

        Returns:
            [batch, num_layers * hidden_dim] if pooled, else
            [batch, seq_len, num_layers * hidden_dim]. Layers appear in the
            order given by `layers`.

        Raises:
            ValueError: If this output holds a single layer.
        """
        if self.layers is None:
            raise ValueError(
                "This output holds a single layer; there is nothing to concatenate."
            )
        if self.is_pooled:
            # [batch, layers, hidden] -> [batch, layers * hidden]
            return self.embeddings.flatten(start_dim=1)
        # [batch, layers, seq, hidden] -> [batch, seq, layers * hidden]
        return self.embeddings.permute(0, 2, 1, 3).flatten(start_dim=2)
```

In `get_chain_embeddings`, replace the final `return` (line 89) with:

```python
        start, end = offsets[chain]
        if self.is_multi_layer:
            return self.embeddings[idx, :, start:end, :]
        return self.embeddings[idx, start:end, :]
```

and extend its Returns docstring with: `For multi-layer output the layer axis leads: [num_layers, chain_len, hidden_dim].`

In `get_sequence_tokens`, replace the masked return (lines 154-155) with:

```python
        mask = self.attention_mask[idx].bool()
        if self.is_multi_layer:
            return self.embeddings[idx][:, mask]
        return self.embeddings[idx][mask]
```

and extend its Returns docstring with: `For multi-layer output: [num_layers, actual_seq_len, hidden_dim].`

Add `layers=self.layers,` to the `EmbeddingOutput(...)` constructions inside both `to()` (line 101) and `numpy()` (line 127).

Replace `__repr__`:

```python
    def __repr__(self) -> str:
        """Return a string representation."""
        shape_str = "x".join(str(d) for d in self.embeddings.shape)
        layer_str = (
            f", layers={self.layers}" if self.is_multi_layer else f", layer={self.layer}"
        )
        pooled_str = f", pooled={self.pooled.shape}" if self.pooled is not None else ""
        return f"EmbeddingOutput(shape=[{shape_str}]{layer_str}{pooled_str})"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_outputs.py -v`
Expected: all PASS, including the pre-existing tests at lines 26 and 33 — they construct bare 2D and 3D tensors with no `layers=`, so `is_pooled` keeps its old meaning for them.

- [ ] **Step 5: Commit**

```bash
git add src/ablms/outputs/embedding.py tests/test_outputs.py
git commit -m "Add multi-layer support to EmbeddingOutput"
```

---

### Task 5: Multi-layer path in `_process_embeddings_batch`

Route multi-layer requests through the `_forward_all_hidden_states()` that every encoder already implements, and pool each layer *before* stacking so the large token-level tensor is never built on pooled runs.

**Files:**
- Modify: `src/ablms/core/encoder.py:110-154`
- Test: `tests/test_embedding_memory.py`

**Interfaces:**
- Consumes: `EncoderAbLM.num_layers` (Task 2).
- Produces:
  - `EncoderAbLM._process_embeddings_batch(sequences, layer: int | list[int] = -1, pooling: str | None = None)` — a list `layer` yields a layer axis at dimension 1.
  - `EncoderAbLM._forward_selected_layers(tokenized, layers: list[int], pooling: str | None) -> tuple[torch.Tensor, torch.Tensor | None]`.
  - Task 6 dispatches to `_process_embeddings_batch` through the executor.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_embedding_memory.py`:

```python
class TestMultiLayerBatchProcessing:
    """The layer axis sits at dimension 1, and pooling still reduces per batch."""

    @pytest.mark.slow
    def test_token_level_stacks_layers(self, esm2_cpu, sequences):
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=[0, 3, 6], pooling=None
        )

        assert embeddings.dim() == 4
        assert embeddings.shape[0] == len(sequences)
        assert embeddings.shape[1] == 3
        assert embeddings.shape[3] == esm2_cpu.embedding_dim
        assert mask is not None

    @pytest.mark.slow
    def test_pooling_reduces_before_transfer(self, esm2_cpu, sequences):
        """The [batch, layers, seq, hidden] tensor must never reach the queue."""
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=[0, 3, 6], pooling="mean"
        )

        assert embeddings.shape == (len(sequences), 3, esm2_cpu.embedding_dim)
        assert mask is None

    @pytest.mark.slow
    def test_each_layer_matches_a_single_layer_call(self, esm2_cpu, sequences):
        stacked, _, _ = esm2_cpu._process_embeddings_batch(
            sequences, layer=[0, 3, 6], pooling="mean"
        )

        for position, layer in enumerate([0, 3, 6]):
            single, _, _ = esm2_cpu._process_embeddings_batch(
                sequences, layer=layer, pooling="mean"
            )
            assert torch.allclose(stacked[:, position], single, atol=1e-6)

    @pytest.mark.slow
    def test_single_int_path_is_unchanged(self, esm2_cpu, sequences):
        """An int layer must not gain a layer axis."""
        embeddings, _, _ = esm2_cpu._process_embeddings_batch(
            sequences, layer=-1, pooling=None
        )
        assert embeddings.dim() == 3

    @pytest.mark.slow
    def test_a_wrong_layer_count_is_caught(self, esm2_cpu, sequences, monkeypatch):
        """A missing num_layers override must fail loudly, not mis-index.

        This is the guard for a newly added encoder whose config does not spell
        its depth `num_hidden_layers`.
        """
        monkeypatch.setattr(type(esm2_cpu), "num_layers", property(lambda self: 99))

        with pytest.raises(RuntimeError, match="num_layers"):
            esm2_cpu._process_embeddings_batch(sequences, layer=[0, 1], pooling="mean")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_embedding_memory.py::TestMultiLayerBatchProcessing -v -m slow`
Expected: FAIL — `_forward_embeddings` does `hidden_states[layer]` on the HuggingFace `hidden_states` tuple with a list argument, raising `TypeError: tuple indices must be integers or slices, not list`.

- [ ] **Step 3: Rewrite the batch method and add the helper**

In `src/ablms/core/encoder.py`, replace the body of `_process_embeddings_batch` (lines 138-154) and update its signature and docstring:

```python
    def _process_embeddings_batch(
        self,
        sequences: list[AntibodySequence],
        layer: int | list[int] = -1,
        pooling: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, tuple[int, int]]]]:
        """
        Process a single batch of sequences for embeddings.

        This method is called by workers and should NOT be parallelized further.

        Pooling is applied on the model's device, before the result is moved to
        the host. This keeps the large [batch, seq_len, hidden_dim] tensor off
        the host entirely and out of the inter-process queue, which is what
        makes large multi-GPU runs viable.

        Args:
            sequences: Batch of sequences (already batched by executor).
            layer: A single layer index, or a list of resolved non-negative
                indices. A list adds a layer axis at dimension 1.
            pooling: Optional pooling strategy applied within this batch.
                One of "mean", "max", "cls", "first", "last", or None for
                token-level output.

        Returns:
            Tuple of (embeddings, attention_mask, token_offsets). The mask is
            None whenever pooling was applied. Embeddings are
            [batch, seq_len, hidden_dim], or [batch, hidden_dim] when pooled;
            a list `layer` inserts a layer axis at dimension 1.
        """
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize(formatted)
        offsets = self._compute_token_offsets(sequences, tokenized)

        if isinstance(layer, int):
            embeddings, mask = self._forward_embeddings(tokenized, layer)
            if pooling is not None:
                embeddings = apply_pooling(
                    embeddings, strategy=pooling, attention_mask=mask
                )
                mask = None
        else:
            embeddings, mask = self._forward_selected_layers(tokenized, layer, pooling)

        # Move results to CPU for cross-process transfer
        embeddings = embeddings.cpu()
        if mask is not None:
            mask = mask.cpu()

        return embeddings, mask, offsets

    def _forward_selected_layers(
        self,
        tokenized: dict[str, torch.Tensor],
        layers: list[int],
        pooling: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Stack several layers, pooling each one before it is stacked.

        Every encoder already implements `_forward_all_hidden_states`, and the
        HuggingFace-backed ones compute every layer regardless, so selecting
        from its output costs nothing over a single-layer forward pass.

        Pooling per layer rather than after stacking means the
        [batch, layers, seq_len, hidden_dim] tensor is never allocated on
        pooled runs - only [batch, layers, hidden_dim] survives to cross the
        result queue. See the "reduce before transfer" note in CLAUDE.md.

        Args:
            tokenized: Tokenized batch.
            layers: Resolved non-negative indices, in the order requested.
            pooling: Optional pooling strategy, applied to each layer.

        Returns:
            Tuple of (embeddings, attention_mask). Embeddings are
            [batch, len(layers), hidden_dim] when pooled, else
            [batch, len(layers), seq_len, hidden_dim]. The mask is None
            whenever pooling was applied.

        Raises:
            RuntimeError: If the model's reported num_layers disagrees with the
                number of hidden states its forward pass returned.
        """
        hidden_states, mask = self._forward_all_hidden_states(tokenized)

        expected = self.num_layers + 1
        if len(hidden_states) != expected:
            raise RuntimeError(
                f"{self.model_name} reports num_layers={self.num_layers} "
                f"({expected} selectable layers), but its forward pass returned "
                f"{len(hidden_states)} hidden states. The num_layers property "
                f"needs an override for this model."
            )

        if pooling is not None:
            pooled = [
                apply_pooling(hidden_states[i], strategy=pooling, attention_mask=mask)
                for i in layers
            ]
            return torch.stack(pooled, dim=1), None

        return torch.stack([hidden_states[i] for i in layers], dim=1), mask
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_embedding_memory.py -v -m slow`
Expected: all PASS, including the pre-existing `TestBatchLevelPooling` tests that assert the single-int path is unchanged.

- [ ] **Step 5: Check the AbLang override is still bind-compatible**

`AbLang` overrides `_process_embeddings_batch`; its parameter names are unchanged, so it should still pass. Verify rather than assume.

Run: `python -m pytest tests/test_encoder_contract.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ablms/core/encoder.py tests/test_embedding_memory.py
git commit -m "Select and stack multiple layers in the embeddings batch path"
```

---

### Task 6: Wire `get_embeddings()` and `iter_embeddings()`

Resolve the layer argument in the parent before dispatch, so bad input fails at the call site rather than inside a worker.

**Files:**
- Modify: `src/ablms/core/encoder.py:32-108` (`get_embeddings`), `156-237` (`iter_embeddings`, `_iter_embeddings`)
- Test: `tests/test_embedding_memory.py`

**Interfaces:**
- Consumes: `resolve_layer_selection` (Task 3), `EmbeddingOutput.layers` (Task 4), `_process_embeddings_batch` with a list `layer` (Task 5), 4D-capable padding (Task 1).
- Produces: the public API — `get_embeddings(sequences, layer: int | list[int] | str = -1, ...)` and `iter_embeddings(...)` with the same widened parameter.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_embedding_memory.py`:

```python
class TestMultiLayerGetEmbeddings:
    """The public API: layer accepts an int, a list, or "all"."""

    @pytest.mark.slow
    def test_default_call_is_unchanged(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(sequences, show_progress=False)

        assert output.embeddings.dim() == 3
        assert not output.is_multi_layer
        assert output.layer == -1
        assert output.layers is None

    @pytest.mark.slow
    def test_all_layers_pooled(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(
            sequences, layer="all", pooling="cls", batch_size=2, show_progress=False
        )

        assert output.is_multi_layer
        assert output.layers == list(range(esm2_cpu.num_layers + 1))
        assert output.layer is None
        assert output.is_pooled
        assert output.embeddings.shape == (
            len(sequences),
            esm2_cpu.num_layers + 1,
            esm2_cpu.embedding_dim,
        )

    @pytest.mark.slow
    def test_concat_layers_gives_one_vector_per_sequence(self, esm2_cpu, sequences):
        """The dimensionality-reduction use case."""
        output = esm2_cpu.get_embeddings(
            sequences, layer="all", pooling="cls", show_progress=False
        )
        features = output.concat_layers()

        assert features.shape == (
            len(sequences),
            (esm2_cpu.num_layers + 1) * esm2_cpu.embedding_dim,
        )

    @pytest.mark.slow
    def test_explicit_list_preserves_order(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(
            sequences, layer=[6, 0], pooling="mean", show_progress=False
        )
        assert output.layers == [6, 0]

    @pytest.mark.slow
    def test_get_layer_matches_a_single_layer_call(self, esm2_cpu, sequences):
        multi = esm2_cpu.get_embeddings(
            sequences, layer=[0, 3, 6], pooling="mean", show_progress=False
        )
        single = esm2_cpu.get_embeddings(
            sequences, layer=3, pooling="mean", show_progress=False
        )

        assert torch.allclose(multi.get_layer(3), single.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_single_element_list_keeps_the_layer_axis(self, esm2_cpu, sequences):
        """The argument's type decides the shape, not its length."""
        listed = esm2_cpu.get_embeddings(
            sequences, layer=[-1], pooling="mean", show_progress=False
        )
        scalar = esm2_cpu.get_embeddings(
            sequences, layer=-1, pooling="mean", show_progress=False
        )

        assert listed.embeddings.shape == (len(sequences), 1, esm2_cpu.embedding_dim)
        assert torch.allclose(listed.embeddings[:, 0], scalar.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_ragged_multi_batch_token_level_concatenates(
        self, esm2_cpu, ragged_sequences
    ):
        """Batches padded to different lengths must concatenate across the layer axis.

        This is the case that fails if _pad_tensors_to_max_length assumes
        dimension 1 is the sequence axis.
        """
        output = esm2_cpu.get_embeddings(
            ragged_sequences, layer=[0, 3], pooling=None, batch_size=2,
            show_progress=False,
        )
        n = len(ragged_sequences)
        max_len = output.embeddings.shape[2]

        assert output.embeddings.shape == (n, 2, max_len, esm2_cpu.embedding_dim)
        assert output.attention_mask.shape == (n, max_len)

    @pytest.mark.slow
    def test_invalid_layer_fails_at_the_call_site(self, esm2_cpu, sequences):
        with pytest.raises(ValueError, match="out of range"):
            esm2_cpu.get_embeddings(sequences, layer=999, show_progress=False)

    @pytest.mark.slow
    def test_empty_input_reports_the_layer_axis(self, esm2_cpu):
        output = esm2_cpu.get_embeddings([], layer="all", pooling="mean")

        assert output.layers == list(range(esm2_cpu.num_layers + 1))
        assert output.embeddings.shape == (
            0, esm2_cpu.num_layers + 1, esm2_cpu.embedding_dim
        )


class TestMultiLayerIterEmbeddings:
    @pytest.mark.slow
    def test_each_batch_carries_its_layers(self, esm2_cpu, sequences):
        outputs = list(
            esm2_cpu.iter_embeddings(
                sequences, layer=[0, 3], pooling="mean", batch_size=2,
                show_progress=False,
            )
        )

        assert len(outputs) == 3
        assert all(o.layers == [0, 3] for o in outputs)
        assert outputs[0].embeddings.shape == (2, 2, esm2_cpu.embedding_dim)

    @pytest.mark.slow
    def test_stream_matches_get_embeddings(self, esm2_cpu, sequences):
        streamed = torch.cat(
            [
                o.embeddings
                for o in esm2_cpu.iter_embeddings(
                    sequences, layer="all", pooling="mean", batch_size=2,
                    show_progress=False,
                )
            ]
        )
        combined = esm2_cpu.get_embeddings(
            sequences, layer="all", pooling="mean", batch_size=2, show_progress=False
        )

        assert torch.allclose(streamed, combined.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_invalid_layer_fails_eagerly(self, esm2_cpu, sequences):
        """Like sequence validation, this must raise on call, not on first next()."""
        with pytest.raises(ValueError, match="out of range"):
            esm2_cpu.iter_embeddings(sequences, layer=999, show_progress=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_embedding_memory.py::TestMultiLayerGetEmbeddings -v -m slow`
Expected: FAIL — `test_all_layers_pooled` raises `TypeError: tuple indices must be integers or slices, not str`, because `get_embeddings` passes the raw `"all"` straight through to the worker's `hidden_states` lookup.

- [ ] **Step 3: Add the import**

In `src/ablms/core/encoder.py`, next to the existing pooling import (line 14):

```python
from ablms.utils.layers import resolve_layer_selection
```

- [ ] **Step 4: Rewrite `get_embeddings`**

Replace lines 32-108 with:

```python
    def get_embeddings(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        layer: int | list[int] | str = -1,
        pooling: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingOutput:
        """
        Get embeddings for sequences.

        Args:
            sequences: Input sequences in various formats.
            layer: Which layer(s) to extract. One of:
                - an int (default -1, the final layer). Index 0 is the
                  embedding layer and index i is the output of block i.
                - a list of ints, which adds a layer axis at dimension 1.
                - "all", for every layer in ascending order.
                A list of length one still adds the layer axis, so a
                programmatically built selection has a stable shape.
            pooling: Optional pooling strategy for sequence-level embeddings.
                If None (default), returns token-level embeddings. Pooling is
                applied within each batch on the model's device, and per layer
                before layers are stacked, so pooled runs never materialize the
                full token-level tensor.
                Valid options: "mean", "max", "cls", "first", "last".
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            EmbeddingOutput containing embeddings. Shape is
            [batch, seq_len, hidden_dim] for token-level (pooling=None) or
            [batch, hidden_dim] for sequence-level (pooling specified), with a
            layer axis inserted at dimension 1 when several layers are selected.

        Raises:
            ValueError: If the layer selection is malformed or out of range.
            UnsupportedOperationError: If a non-final layer is requested from a
                model that exposes only its final layer.

        Note:
            Token-level output for many layers is large: "all" on a 12-block
            model with hidden_dim 1024 is roughly 13x the single-layer payload.
            Use iter_embeddings() for anything that will not fit in memory.

        Example:
            >>> # Every layer, CLS-pooled, as one feature vector per sequence
            >>> out = model.get_embeddings(seqs, layer="all", pooling="cls")
            >>> features = out.concat_layers()  # [batch, n_layers * hidden_dim]
        """
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        selection = resolve_layer_selection(
            layer,
            self.num_layers,
            model_name=self.model_name,
            supports_intermediate_layers=self.supports_intermediate_layers,
        )
        layers = None if isinstance(selection, int) else selection
        single_layer = selection if isinstance(selection, int) else None

        if len(sequences) == 0:
            return self._empty_embedding_output(layers, single_layer, pooling)

        executor = self._get_executor()
        all_embeddings, all_masks, all_offsets = executor.execute(
            method_name="_process_embeddings_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing embeddings",
            layer=selection,
            pooling=pooling,
        )

        return EmbeddingOutput(
            embeddings=all_embeddings,
            # Pooling already reduced each batch, so there is no mask to carry.
            attention_mask=None if pooling is not None else all_masks,
            token_offsets=all_offsets,
            pooled=all_embeddings if pooling is not None else None,
            sequences=sequences,
            layer=single_layer,
            layers=layers,
        )

    def _empty_embedding_output(
        self,
        layers: list[int] | None,
        single_layer: int | None,
        pooling: str | None,
    ) -> EmbeddingOutput:
        """Build the zero-sequence result with the shape a real run would produce."""
        shape: tuple[int, ...]
        if pooling is None:
            shape = (0, 0, self.embedding_dim)
        else:
            shape = (0, self.embedding_dim)

        if layers is not None:
            shape = (0, len(layers), *shape[1:])

        return EmbeddingOutput(
            embeddings=torch.empty(*shape),
            attention_mask=None,
            token_offsets=[],
            sequences=[],
            layer=single_layer,
            layers=layers,
        )
```

- [ ] **Step 5: Update `iter_embeddings` and `_iter_embeddings`**

Change `iter_embeddings`'s signature to `layer: int | list[int] | str = -1`, copy the `layer` argument documentation from `get_embeddings` into its Args section, add `ValueError` and `UnsupportedOperationError` to its Raises section, and replace its body after `self._validate_input(sequences)` with:

```python
        selection = resolve_layer_selection(
            layer,
            self.num_layers,
            model_name=self.model_name,
            supports_intermediate_layers=self.supports_intermediate_layers,
        )

        return self._iter_embeddings(
            sequences=sequences,
            layer=selection,
            pooling=pooling,
            batch_size=batch_size,
            show_progress=show_progress,
        )
```

In `_iter_embeddings`, widen the parameter to `layer: int | list[int]`, and inside the generator set the layer fields on each yielded output:

```python
        layers = None if isinstance(layer, int) else layer
        single_layer = layer if isinstance(layer, int) else None

        executor = self._get_executor()
        for batch_idx, (embeddings, mask, offsets) in executor.execute_iter(
            method_name="_process_embeddings_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing embeddings",
            layer=layer,
            pooling=pooling,
        ):
            start = batch_idx * batch_size
            yield EmbeddingOutput(
                embeddings=embeddings,
                attention_mask=mask,
                token_offsets=offsets,
                pooled=embeddings if pooling is not None else None,
                sequences=sequences[start : start + batch_size],
                layer=single_layer,
                layers=layers,
            )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_embedding_memory.py -v -m slow`
Expected: all PASS, including every pre-existing test in the file.

- [ ] **Step 7: Commit**

```bash
git add src/ablms/core/encoder.py tests/test_embedding_memory.py
git commit -m "Accept a layer list or 'all' in the public embedding API"
```

---

### Task 7: Reimplement `get_hidden_states()` over `layer="all"`

Keep the method and its `list[EmbeddingOutput]` return type, but stop maintaining a second all-layer code path.

**Files:**
- Modify: `src/ablms/core/encoder.py:239-284`
- Test: `tests/test_embedding_memory.py`

**Interfaces:**
- Consumes: `get_embeddings(layer="all")` (Task 6), `EmbeddingOutput.layers` (Task 4).
- Produces: `get_hidden_states(sequences, batch_size=32, show_progress=True) -> list[EmbeddingOutput]`, unchanged in signature and return type.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_embedding_memory.py`:

```python
class TestGetHiddenStates:
    """The wrapper must return exactly what the standalone path used to."""

    @pytest.mark.slow
    def test_returns_one_output_per_layer(self, esm2_cpu, sequences):
        outputs = esm2_cpu.get_hidden_states(sequences, show_progress=False)

        assert len(outputs) == esm2_cpu.num_layers + 1
        assert [o.layer for o in outputs] == list(range(esm2_cpu.num_layers + 1))
        assert all(not o.is_multi_layer for o in outputs)

    @pytest.mark.slow
    def test_each_output_is_token_level(self, esm2_cpu, sequences):
        outputs = esm2_cpu.get_hidden_states(sequences, show_progress=False)

        for output in outputs:
            assert output.embeddings.dim() == 3
            assert output.embeddings.shape[0] == len(sequences)
            assert output.embeddings.shape[2] == esm2_cpu.embedding_dim
            assert output.attention_mask is not None

    @pytest.mark.slow
    def test_matches_get_embeddings_for_the_same_layer(self, esm2_cpu, sequences):
        outputs = esm2_cpu.get_hidden_states(sequences, show_progress=False)
        direct = esm2_cpu.get_embeddings(sequences, layer=3, show_progress=False)

        assert torch.allclose(outputs[3].embeddings, direct.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_empty_input_returns_empty_list(self, esm2_cpu):
        assert esm2_cpu.get_hidden_states([]) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_embedding_memory.py::TestGetHiddenStates -v -m slow`
Expected: `test_returns_one_output_per_layer` FAILS on the `layer` assertion — the current implementation is a separate path, so run it and record the actual failure before changing anything. If all four pass against the old implementation, that is the desired baseline: the rewrite must keep them passing.

- [ ] **Step 3: Replace the method**

Replace `get_hidden_states` (lines 239-284) with:

```python
    def get_hidden_states(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[EmbeddingOutput]:
        """
        Get embeddings from all layers.

        A thin wrapper over `get_embeddings(layer="all")`, kept for backwards
        compatibility. Prefer `layer="all"` directly: it returns one output with
        a layer axis, supports pooling, and streams through `iter_embeddings()`.
        This method materializes the full token-level output for every layer.

        Args:
            sequences: Input sequences in various formats.
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Returns:
            List of EmbeddingOutput objects, one per layer, in ascending layer
            order. Models that expose only their final layer return a
            single-element list.
        """
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        if len(sequences) == 0:
            return []

        # A final-layer-only model (AbLang) would raise on "all".
        selection = "all" if self.supports_intermediate_layers else -1
        stacked = self.get_embeddings(
            sequences,
            layer=selection,
            pooling=None,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        if not stacked.is_multi_layer:
            return [stacked]

        return [
            EmbeddingOutput(
                embeddings=stacked.embeddings[:, position],
                attention_mask=stacked.attention_mask,
                token_offsets=stacked.token_offsets,
                sequences=stacked.sequences,
                layer=layer_index,
            )
            for position, layer_index in enumerate(stacked.layers)
        ]
```

Leave `_process_hidden_states_batch` in place: `_forward_all_hidden_states` is still the engine of the multi-layer path, and other callers may rely on the batch method.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_embedding_memory.py -v -m slow`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ablms/core/encoder.py tests/test_embedding_memory.py
git commit -m "Reimplement get_hidden_states over layer='all'"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md` (embeddings section around line 96-108; output classes list around line 319)
- Modify: `CLAUDE.md` (Output Classes; Important Patterns; Adding a New Encoder Model)

**Interfaces:**
- Consumes: the finished public API from Tasks 4, 6, and 7.
- Produces: no code.

- [ ] **Step 1: Document the API in the README**

In `README.md`, after the pooled example (line 103) and before the streaming example (line 105), insert:

````markdown
# Select several layers, or every layer, by passing a list or "all"
# A layer axis is inserted at dimension 1
multi = model.get_embeddings(sequences, layer=[0, 6, 12], pooling="cls")
print(multi.embeddings.shape)  # [2, 3, 768]
print(multi.get_layer(6).shape)  # [2, 768]

# Concatenate every layer into one feature vector per sequence,
# the usual input for a UMAP or t-SNE projection
every = model.get_embeddings(sequences, layer="all", pooling="cls")
print(every.concat_layers().shape)  # [2, 13 * 768]
````

Then add a note below the code block:

````markdown
Token-level output for many layers is large — `layer="all"` on a 12-block model
is roughly 13x the single-layer payload — so pair it with `iter_embeddings()`
rather than `get_embeddings()` for anything sizeable. Pooled multi-layer runs
stay small: pooling is applied per layer before the layers are stacked.

`AbLang` exposes only its final layer and raises `UnsupportedOperationError` for
any other selection.
````

In the output classes list (line 319), update the `EmbeddingOutput` entry to mention `get_layer()`, `concat_layers()`, and `layers`.

- [ ] **Step 2: Update CLAUDE.md**

In the "Output Classes" section, change the `EmbeddingOutput` line to:

```markdown
- `EmbeddingOutput`: Token/sequence embeddings with `get_chain_embeddings()`. Multi-layer
  results carry a layer axis at dimension 1, `layers` listing the resolved indices, and
  `get_layer()` / `concat_layers()` for extracting or flattening it.
```

In "Important Patterns", after the "Reduce before transfer" bullet, add:

```markdown
- **Layer selection**: `get_embeddings(layer=...)` accepts an int, a list of ints, or
  `"all"`. `utils/layers.py::resolve_layer_selection` validates and resolves it in the
  parent before dispatch; a list return means the batch method stacks a layer axis at
  dimension 1. Multi-layer requests route through `_forward_all_hidden_states()`, which
  every encoder already implements, so no encoder needs a multi-layer forward pass. Pooling
  is applied per layer *before* stacking, so the token-level tensor is never built on
  pooled runs.
```

In "Adding a New Encoder Model", extend step 2 (class attributes):

```markdown
2. Set class attributes: `model_name`, `supports_paired`, `max_length`, `embedding_dim`,
   `mask_token`, `separator`, `has_mlm_head`. Override the `num_layers` property if the
   model object does not expose a HuggingFace config with `num_hidden_layers` (IgT5 and
   AbLang2 do), and set `supports_intermediate_layers = False` if only the final layer is
   reachable (AbLang).
```

- [ ] **Step 3: Verify the README examples actually run**

Do not trust the shapes in the documentation. Run them.

```bash
python -c "
from ablms.encoders import ESM2
from ablms import AntibodySequence
m = ESM2(devices='cpu', model_id='facebook/esm2_t6_8M_UR50D')
seqs = [AntibodySequence(heavy='EVQLVESGGGLVQPGRSLRLSCAASGFTFS'),
        AntibodySequence(heavy='QVQLVQSGAEVKKPGASVKVSCKASGYTFT')]
multi = m.get_embeddings(seqs, layer=[0, 3, 6], pooling='cls', show_progress=False)
print('multi', tuple(multi.embeddings.shape), 'layer6', tuple(multi.get_layer(6).shape))
every = m.get_embeddings(seqs, layer='all', pooling='cls', show_progress=False)
print('concat', tuple(every.concat_layers().shape))
"
```

Expected: `multi (2, 3, 320) layer6 (2, 320)` and `concat (2, 2240)` for the t6/320-dim model. Adjust the README's commented shapes if the model used there (IgBERT, 768-dim, 12 blocks) implies different numbers — `[2, 3, 768]` and `[2, 13 * 768]` are correct for a 12-block 768-dim model.

- [ ] **Step 4: Run the full suite and the linters**

Run: `python -m pytest -v`
Expected: PASS, except the pre-existing AbLang failures caused by the missing `ablang` package. Confirm the count of those failures matches `git stash && python -m pytest -q; git stash pop` if there is any doubt about which are pre-existing.

Run: `black src/ tests/ && ruff check src/ tests/`
Expected: reformatting only, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md src/ tests/
git commit -m "Document multi-layer embedding selection"
```

---

## Verification

After Task 8, confirm the acceptance bar from Global Constraints holds:

- [ ] `python -m pytest -m slow -v` passes for `tests/test_embedding_memory.py`, `tests/test_outputs.py`, `tests/test_executor.py`, `tests/test_layer_selection.py`, `tests/test_encoder_contract.py`.
- [ ] A default `get_embeddings(seqs)` call returns a 3D tensor with `layer == -1` and `layers is None`.
- [ ] `get_embeddings(seqs, layer="all", pooling="cls").concat_layers()` returns `[n_sequences, (num_layers + 1) * embedding_dim]`.
- [ ] Multi-GPU is exercised at least once if hardware allows: `get_embeddings(seqs, layer=[0, 3], batch_size=2)` on a model constructed with two devices, using sequences of unequal length. This is the path Task 1 protects and single-device runs do not cover.
