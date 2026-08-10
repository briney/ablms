# Bounding `get_embeddings()` Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop multi-GPU embedding runs from exhausting `/dev/shm` by pooling inside each batch, bounding the number of in-flight results, and adding a streaming `iter_embeddings()`.

**Architecture:** `EncoderAbLM.get_embeddings` currently ships a full `[batch, seq_len, hidden_dim]` tensor per batch through a `torch.multiprocessing` queue (which backs each one with a `/dev/shm` segment), holds every batch alive until the run finishes, and only then applies pooling. This plan pushes pooling down into `_process_embeddings_batch` so the reduction happens on the accelerator before the tensor ever crosses the queue; rebuilds `MultiGPUExecutor` around a generator with a bounded submission window that copies received tensors out of shared memory; and exposes that generator publicly as `iter_embeddings()`.

**Tech Stack:** Python 3.10+, PyTorch, `torch.multiprocessing` (spawn context), tqdm, pytest, HuggingFace Transformers (ESM-2 for end-to-end tests).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-embedding-memory-design.md`. Read it before starting.
- Python 3.10+ union syntax: write `x | None`, never `Optional[x]`. Never import `Optional` or `Union` from `typing`.
- `from __future__ import annotations` at the top of every module touched (already present in all of them).
- Google-style docstrings on all public classes and functions.
- Line length 88 (`[tool.black]` and `[tool.ruff]` in `pyproject.toml`).
- Formatter `black`, linter `ruff check`, both scoped to `src/ tests/`.
- No change to the numerical output of `get_embeddings` for any pooling strategy.
- `MultiGPUExecutor.execute()` keeps its exact current signature — it has seven callers in `src/ablms/core/encoder.py` (lines 79, 165, 243, 322, 399, 464, 536) that are out of scope for this work.
- Tests use real models, never mocks of model behavior: `facebook/esm2_t6_8M_UR50D` on CPU, marked `@pytest.mark.slow`. The one permitted fake is a non-model stand-in used to exercise executor plumbing (Task 4).

---

### Task 1: Register the `slow` pytest marker

`@pytest.mark.slow` is used throughout `tests/` but never declared, so every run emits `PytestUnknownMarkWarning` for each use. Register it first so later tasks get clean output.

**Files:**
- Modify: `pyproject.toml:62-65`

**Interfaces:**
- Consumes: nothing.
- Produces: a registered `slow` marker, so `pytest -m "not slow"` works and warnings stop.

- [ ] **Step 1: Confirm the warning exists**

Run: `pytest tests/test_esm2.py -m "not slow" -q 2>&1 | grep -c PytestUnknownMarkWarning`
Expected: a non-zero count.

- [ ] **Step 2: Register the marker**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests that load real model weights (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 3: Verify the warning is gone**

Run: `pytest tests/test_esm2.py -m "not slow" -q 2>&1 | grep -c PytestUnknownMarkWarning`
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Register slow pytest marker"
```

---

### Task 2: Pool inside the batch, on the accelerator

The core fix. `_process_embeddings_batch` gains a `pooling` parameter and applies it *before* the existing `.cpu()` call, so the large `[B, L, D]` tensor is never copied to host memory and never crosses the queue.

**Files:**
- Modify: `src/ablms/core/encoder.py:32-139` (`get_embeddings` and `_process_embeddings_batch`)
- Modify: `src/ablms/parallel/executor.py:378-410` (`_pad_tensors_to_max_length`)
- Test: `tests/test_pooling.py` (append a new class)
- Test: `tests/test_embedding_memory.py` (create)

**Interfaces:**
- Consumes: `apply_pooling(embeddings, strategy, attention_mask)` from `ablms.utils.pooling`, already imported at `encoder.py:14`.
- Produces: `_process_embeddings_batch(sequences, layer=-1, pooling=None)` returning `(embeddings, mask, offsets)` where `embeddings` is `[B, D]` and `mask` is `None` when `pooling` is not `None`, and `[B, L, D]` with a `[B, L]` mask otherwise. Task 5 depends on this signature.

- [ ] **Step 1: Write the failing padding-invariance test**

This is the guarantee that makes per-batch pooling safe. Append to `tests/test_pooling.py`:

```python
class TestPoolingIsPaddingInvariant:
    """Pooling a batch alone must equal pooling it inside a longer padded stack.

    This is what makes it safe to pool per batch rather than after all batches
    have been concatenated and padded to a global maximum length.
    """

    @pytest.mark.parametrize("strategy", ["mean", "max", "cls", "first", "last"])
    def test_extra_right_padding_does_not_change_result(self, strategy):
        torch.manual_seed(0)
        batch, real_len, hidden, extra = 3, 7, 16, 11

        embeddings = torch.randn(batch, real_len, hidden)
        mask = torch.ones(batch, real_len)
        mask[1, 5:] = 0  # a shorter sequence in the batch

        padded = torch.cat([embeddings, torch.zeros(batch, extra, hidden)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(batch, extra)], dim=1)

        tight = apply_pooling(embeddings, strategy, mask)
        loose = apply_pooling(padded, strategy, padded_mask)

        assert torch.allclose(tight, loose, atol=1e-6)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_pooling.py::TestPoolingIsPaddingInvariant -v`
Expected: PASS for all five strategies. This test characterizes existing behavior in `src/ablms/utils/pooling.py` rather than driving new code — it is the safety net the rest of the task leans on. If any strategy fails here, stop and report it, because per-batch pooling would then change results.

- [ ] **Step 3: Write the failing test for the new parameter**

Create `tests/test_embedding_memory.py`:

```python
"""Tests for bounded-memory embedding extraction."""

import pytest
import torch

from ablms import AntibodySequence
from ablms.encoders import ESM2

MODEL_ID = "facebook/esm2_t6_8M_UR50D"

HEAVY_CHAINS = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQG",
    "EVQLVESGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKG",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKS",
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKG",
]


@pytest.fixture(scope="session")
def esm2_cpu():
    """Session-scoped real model; loading weights is the expensive part."""
    return ESM2(devices="cpu", model_id=MODEL_ID)


@pytest.fixture(scope="session")
def sequences():
    return [AntibodySequence(heavy=h) for h in HEAVY_CHAINS]


class TestBatchLevelPooling:
    """_process_embeddings_batch reduces before returning."""

    @pytest.mark.slow
    def test_pooling_returns_two_dimensional_result(self, esm2_cpu, sequences):
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=-1, pooling="mean"
        )
        assert embeddings.dim() == 2
        assert embeddings.shape == (len(sequences), esm2_cpu.embedding_dim)
        assert mask is None
        assert len(offsets) == len(sequences)

    @pytest.mark.slow
    def test_no_pooling_still_returns_token_level(self, esm2_cpu, sequences):
        embeddings, mask, offsets = esm2_cpu._process_embeddings_batch(
            sequences, layer=-1, pooling=None
        )
        assert embeddings.dim() == 3
        assert embeddings.shape[0] == len(sequences)
        assert embeddings.shape[2] == esm2_cpu.embedding_dim
        assert mask is not None
```

- [ ] **Step 4: Run it to verify it fails**

Run: `pytest tests/test_embedding_memory.py::TestBatchLevelPooling -v`
Expected: FAIL with `TypeError: _process_embeddings_batch() got an unexpected keyword argument 'pooling'`.

- [ ] **Step 5: Add the `pooling` parameter to `_process_embeddings_batch`**

Replace `src/ablms/core/encoder.py:112-139` with:

```python
    def _process_embeddings_batch(
        self,
        sequences: list[AntibodySequence],
        layer: int = -1,
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
            layer: Layer index to extract embeddings from.
            pooling: Optional pooling strategy applied within this batch.
                One of "mean", "max", "cls", "first", "last", or None for
                token-level output.

        Returns:
            Tuple of (embeddings, attention_mask, token_offsets). When pooling
            is applied, embeddings has shape [batch, hidden_dim] and the mask is
            None; otherwise embeddings has shape [batch, seq_len, hidden_dim].
        """
        formatted = self._format_for_model(sequences)
        tokenized = self._tokenize(formatted)
        offsets = self._compute_token_offsets(sequences, tokenized)
        embeddings, mask = self._forward_embeddings(tokenized, layer)

        if pooling is not None:
            embeddings = apply_pooling(
                embeddings, strategy=pooling, attention_mask=mask
            )
            mask = None

        # Move results to CPU for cross-process transfer
        embeddings = embeddings.cpu()
        if mask is not None:
            mask = mask.cpu()

        return embeddings, mask, offsets
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest tests/test_embedding_memory.py::TestBatchLevelPooling -v`
Expected: PASS.

- [ ] **Step 7: Write the failing end-to-end equivalence test**

Append to `tests/test_embedding_memory.py`:

```python
class TestPooledEmbeddingsUnchanged:
    """get_embeddings must produce identical values after the refactor."""

    @pytest.mark.slow
    @pytest.mark.parametrize("strategy", ["mean", "max", "cls", "first", "last"])
    def test_multi_batch_matches_single_batch(self, esm2_cpu, sequences, strategy):
        """Splitting into batches must not change pooled values.

        Before this change, pooling ran once over the globally padded stack.
        Now it runs per batch. Processing the same input as one batch and as
        three batches must agree, which is exactly the property that would
        break if per-batch pooling were not padding-invariant.
        """
        one_batch = esm2_cpu.get_embeddings(
            sequences, pooling=strategy, batch_size=len(sequences), show_progress=False
        )
        many_batches = esm2_cpu.get_embeddings(
            sequences, pooling=strategy, batch_size=2, show_progress=False
        )

        assert one_batch.embeddings.shape == (len(sequences), esm2_cpu.embedding_dim)
        assert torch.allclose(
            one_batch.embeddings, many_batches.embeddings, atol=1e-5
        )

    @pytest.mark.slow
    def test_pooled_output_fields(self, esm2_cpu, sequences):
        output = esm2_cpu.get_embeddings(
            sequences, pooling="mean", batch_size=2, show_progress=False
        )
        assert output.is_pooled
        assert output.pooled is not None
        assert output.attention_mask is None
        assert len(output.token_offsets) == len(sequences)
```

- [ ] **Step 8: Run it to verify it fails**

Run: `pytest tests/test_embedding_memory.py::TestPooledEmbeddingsUnchanged -v`
Expected: FAIL — `get_embeddings` does not yet forward `pooling` to the batch method, so it still pools post-hoc and the executor receives 3-D tensors while `_process_embeddings_batch` ignores the new kwarg. The precise failure is a `TypeError` from the executor passing `pooling` through `**method_kwargs` only after Step 9; before that, the shapes disagree.

- [ ] **Step 9: Forward `pooling` from `get_embeddings`**

Replace `src/ablms/core/encoder.py:78-110` (from `executor = self._get_executor()` through the final `return`) with:

```python
        executor = self._get_executor()
        all_embeddings, all_masks, all_offsets = executor.execute(
            method_name="_process_embeddings_batch",
            sequences=sequences,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc="Computing embeddings",
            layer=layer,
            pooling=pooling,
        )

        if pooling is not None:
            # Already reduced per batch; all_embeddings is [total, hidden_dim].
            return EmbeddingOutput(
                embeddings=all_embeddings,
                attention_mask=None,
                token_offsets=all_offsets,
                pooled=all_embeddings,
                sequences=sequences,
                layer=layer,
            )

        return EmbeddingOutput(
            embeddings=all_embeddings,
            attention_mask=all_masks,
            token_offsets=all_offsets,
            sequences=sequences,
            layer=layer,
        )
```

Also update the `pooling` argument's docstring in `get_embeddings` to record the new behavior. Replace the `pooling:` entry in the Args block with:

```
            pooling: Optional pooling strategy for sequence-level embeddings.
                If None (default), returns token-level embeddings. Pooling is
                applied within each batch on the model's device, so pooled runs
                never materialize the full token-level tensor.
                Valid options: "mean", "max", "cls", "first", "last".
```

- [ ] **Step 10: Fix `_pad_tensors_to_max_length` to skip reduced results**

Pooled batches are `[B, D]`. Today they survive `_pad_tensors_to_max_length` only because `hidden_dim` is constant across batches, so the computed max equals `D` and no padding happens. Make that explicit. In `src/ablms/parallel/executor.py`, replace the guard at line 394-395:

```python
        if not tensors or tensors[0].dim() < 3:
            # Nothing to pad: 1-D and 2-D results (e.g. pooled [batch, hidden])
            # have no sequence axis. Only [batch, seq, ...] tensors are padded.
            return tensors
```

and update that method's docstring `Args`/`Returns` to say tensors with fewer than three dimensions are returned unchanged.

- [ ] **Step 11: Run the full fast suite plus the new tests**

Run: `pytest tests/test_pooling.py tests/test_outputs.py -q && pytest tests/test_embedding_memory.py -v`
Expected: all PASS.

- [ ] **Step 12: Format, lint, commit**

```bash
black src/ tests/ && ruff check src/ tests/
git add src/ablms/core/encoder.py src/ablms/parallel/executor.py tests/test_pooling.py tests/test_embedding_memory.py
git commit -m "Pool embeddings within each batch on-device

Applying pooling in _process_embeddings_batch before the .cpu() call
keeps the full [batch, seq_len, hidden_dim] tensor off the host and out
of the multiprocessing queue. Payload per batch drops from
B*L*D*4 bytes to B*D*4 - roughly 256x at default settings.

Pooling is padding-invariant given the attention mask, so results are
unchanged; tests/test_pooling.py now asserts that property directly."
```

---

### Task 3: Copy queue-received tensors out of shared memory

A tensor pulled off a `torch.multiprocessing` queue is still backed by its `/dev/shm` segment for as long as it is referenced. Cloning it moves the data to ordinary heap memory and lets the segment be freed.

**Files:**
- Modify: `src/ablms/parallel/executor.py` (add a module-level helper)
- Test: `tests/test_executor.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `_detach_from_shm(obj: Any) -> Any`, a module-level function in `ablms.parallel.executor`. Task 4 calls it on every result received from the queue.

- [ ] **Step 1: Write the failing test**

Create `tests/test_executor.py`:

```python
"""Tests for the multi-GPU executor's memory and ordering behavior."""

import pytest
import torch

from ablms.parallel.executor import _detach_from_shm


class TestDetachFromShm:
    """Results received from a worker queue must be copied out of /dev/shm."""

    def test_tensor_leaves_shared_memory(self):
        shared = torch.randn(4, 8).share_memory_()
        assert shared.is_shared()

        detached = _detach_from_shm(shared)

        assert not detached.is_shared()
        assert torch.equal(detached, shared)

    def test_tuple_of_tensors_and_none(self):
        shared = torch.randn(2, 3).share_memory_()
        offsets = [{"heavy": (1, 10)}]

        detached = _detach_from_shm((shared, None, offsets))

        assert isinstance(detached, tuple)
        assert not detached[0].is_shared()
        assert detached[1] is None
        assert detached[2] == offsets

    def test_list_of_tensors(self):
        """Covers _process_hidden_states_batch, which returns a list per layer."""
        shared = [torch.randn(2, 3).share_memory_() for _ in range(3)]

        detached = _detach_from_shm(shared)

        assert len(detached) == 3
        assert all(not t.is_shared() for t in detached)

    def test_non_tensor_passes_through(self):
        assert _detach_from_shm([1.0, 2.0]) == [1.0, 2.0]
        assert _detach_from_shm("scores") == "scores"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL with `ImportError: cannot import name '_detach_from_shm'`.

- [ ] **Step 3: Implement the helper**

In `src/ablms/parallel/executor.py`, add after the `if TYPE_CHECKING:` block and before `class MultiGPUExecutor:`:

```python
def _detach_from_shm(obj: Any) -> Any:
    """
    Copy queue-received tensors out of shared memory.

    torch.multiprocessing transfers CPU tensors by moving their storage into a
    POSIX shared memory segment and passing a file descriptor, rather than by
    copying bytes through the pipe. The segment stays live for as long as any
    process holds the tensor, so a parent that retains results keeps consuming
    /dev/shm - which is only 64 MB by default inside a container. Cloning moves
    the data into ordinary heap memory so the segment can be released.

    Args:
        obj: A value received from a worker: a tensor, or a tuple/list that may
            contain tensors, or anything else.

    Returns:
        The same structure with every tensor replaced by a heap-backed clone.
        Non-tensor values are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_detach_from_shm(item) for item in obj)
    if isinstance(obj, list):
        return [_detach_from_shm(item) for item in obj]
    return obj
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_executor.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
black src/ tests/ && ruff check src/ tests/
git add src/ablms/parallel/executor.py tests/test_executor.py
git commit -m "Add _detach_from_shm to copy results out of shared memory"
```

---

### Task 4: Rebuild the executor around a bounded generator

Replace up-front submission of every task with a sliding window, and expose results as a generator that yields in input order. This is what stops `/dev/shm` usage growing with dataset size for token-level runs.

**Files:**
- Modify: `src/ablms/parallel/utils.py:12-15` (add a window constant)
- Modify: `src/ablms/parallel/executor.py:145-337` (replace `execute`, `_execute_single`, `_execute_multi`, `_distribute_work`)
- Test: `tests/test_executor.py` (append)
- Test: `tests/fake_worker_model.py` (create)

**Interfaces:**
- Consumes: `_detach_from_shm` from Task 3.
- Produces:
  - `DEFAULT_SUBMISSION_WINDOW: int` in `ablms.parallel.utils`.
  - `MultiGPUExecutor.execute_iter(method_name, sequences, batch_size, show_progress=True, progress_desc=None, **method_kwargs) -> Iterator[tuple[int, Any]]`, yielding `(batch_index, result)` in ascending `batch_index` order. Task 5 consumes this.
  - `MultiGPUExecutor.execute(...)` keeps its existing signature and return type.

- [ ] **Step 1: Create the stand-in worker model**

`spawn` pickles the model class by reference, so it must live at module level in an importable module. `tests/__init__.py` already exists, so `tests.fake_worker_model` is importable. Create `tests/fake_worker_model.py`:

```python
"""A minimal stand-in for a model, used to exercise executor plumbing.

This is deliberately not a BaseAbLM. It exists to test submission windowing
and result ordering without paying for real model weights in a subprocess.
Model behavior itself is always tested against real models.
"""

from __future__ import annotations

import os

import torch


class FakeWorkerModel:
    """Echoes its input batch back as a tensor, tagged with the worker's PID."""

    def __init__(self, devices=None, **kwargs):
        self.devices = devices
        self.pid = os.getpid()

    def _process_echo_batch(self, sequences, scale: int = 1):
        """Return a [len(sequences), 2] tensor of (value * scale, pid)."""
        values = torch.tensor(
            [[float(s) * scale, float(self.pid)] for s in sequences],
            dtype=torch.float32,
        )
        return values, None, [{"item": (0, 1)} for _ in sequences]
```

- [ ] **Step 2: Write the failing ordering and completeness test**

Append to `tests/test_executor.py`:

```python
from ablms.parallel.executor import MultiGPUExecutor
from tests.fake_worker_model import FakeWorkerModel


def _cpu_executor(num_workers: int) -> MultiGPUExecutor:
    """Build an executor backed by real worker processes, without needing GPUs.

    resolve_devices(["cpu", "cpu"]) returns two devices, so the executor takes
    its multi-device branch and spawns genuine subprocesses.
    """
    devices = [torch.device("cpu") for _ in range(num_workers)]
    return MultiGPUExecutor(FakeWorkerModel, {}, devices)


class TestExecuteIterOrdering:
    """execute_iter yields batches in input order across multiple workers."""

    @pytest.mark.slow
    def test_yields_in_input_order(self):
        executor = _cpu_executor(2)
        try:
            emitted = list(
                executor.execute_iter(
                    method_name="_process_echo_batch",
                    sequences=list(range(20)),
                    batch_size=3,
                    show_progress=False,
                )
            )
        finally:
            executor.shutdown()

        indices = [batch_idx for batch_idx, _ in emitted]
        assert indices == list(range(len(emitted)))

        values = torch.cat([result[0][:, 0] for _, result in emitted])
        assert torch.equal(values, torch.arange(20, dtype=torch.float32))

    @pytest.mark.slow
    def test_results_are_not_in_shared_memory(self):
        executor = _cpu_executor(2)
        try:
            for _, (tensor, _, _) in executor.execute_iter(
                method_name="_process_echo_batch",
                sequences=list(range(12)),
                batch_size=3,
                show_progress=False,
            ):
                assert not tensor.is_shared()
        finally:
            executor.shutdown()

    @pytest.mark.slow
    def test_execute_matches_execute_iter(self):
        executor = _cpu_executor(2)
        try:
            combined, _, offsets = executor.execute(
                method_name="_process_echo_batch",
                sequences=list(range(15)),
                batch_size=4,
                show_progress=False,
                scale=2,
            )
        finally:
            executor.shutdown()

        assert combined.shape == (15, 2)
        assert torch.equal(
            combined[:, 0], torch.arange(15, dtype=torch.float32) * 2
        )
        assert len(offsets) == 15

    @pytest.mark.slow
    def test_work_is_spread_across_workers(self):
        """Both workers should receive tasks, confirming the window releases."""
        executor = _cpu_executor(2)
        try:
            combined, _, _ = executor.execute(
                method_name="_process_echo_batch",
                sequences=list(range(40)),
                batch_size=2,
                show_progress=False,
            )
        finally:
            executor.shutdown()

        distinct_pids = set(combined[:, 1].tolist())
        assert len(distinct_pids) == 2
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pytest tests/test_executor.py::TestExecuteIterOrdering -v`
Expected: FAIL with `AttributeError: 'MultiGPUExecutor' object has no attribute 'execute_iter'`.

- [ ] **Step 4: Add the window constant**

In `src/ablms/parallel/utils.py`, after the `WORKER_TIMEOUT` line (line 15), add:

```python
DEFAULT_SUBMISSION_WINDOW = int(os.environ.get("ABLMS_SUBMISSION_WINDOW", 2))
```

Then add `"DEFAULT_SUBMISSION_WINDOW"` to both the import list and `__all__` in `src/ablms/parallel/__init__.py`.

- [ ] **Step 5: Store the window on the executor**

In `src/ablms/parallel/executor.py`, change the import at line 11 to:

```python
from ablms.parallel.utils import DEFAULT_SUBMISSION_WINDOW, WORKER_TIMEOUT
```

Add `import queue` to the stdlib imports at the top of the file. Then in `MultiGPUExecutor.__init__`, after `self._is_single_device = len(devices) == 1`, add:

```python
        # Maximum batches outstanding per worker. Bounds how many results can
        # sit in shared memory at once, independent of dataset size.
        self._submission_window = DEFAULT_SUBMISSION_WINDOW
```

- [ ] **Step 6: Replace `execute` and the two execution paths**

Delete `_distribute_work` entirely (`executor.py:303-337`) — it has no callers outside this file, confirmed by grep. Replace `execute`, `_execute_single`, and `_execute_multi` (`executor.py:145-301`) with:

```python
    def execute(
        self,
        method_name: str,
        sequences: list[AntibodySequence],
        batch_size: int,
        show_progress: bool = True,
        progress_desc: str | None = None,
        **method_kwargs: Any,
    ) -> Any:
        """
        Execute a model method across all devices and combine the results.

        Args:
            method_name: Name of the method to call (e.g. "_process_embeddings_batch").
            sequences: Input sequences to process.
            batch_size: Per-device batch size.
            show_progress: Whether to show a tqdm progress bar.
            progress_desc: Optional custom description for the progress bar.
            **method_kwargs: Additional arguments for the method.

        Returns:
            Combined results from all workers (type depends on the method).
        """
        results = [
            result
            for _, result in self.execute_iter(
                method_name,
                sequences,
                batch_size,
                show_progress=show_progress,
                progress_desc=progress_desc,
                **method_kwargs,
            )
        ]
        return self._combine_results(results)

    def execute_iter(
        self,
        method_name: str,
        sequences: list[AntibodySequence],
        batch_size: int,
        show_progress: bool = True,
        progress_desc: str | None = None,
        **method_kwargs: Any,
    ) -> Iterator[tuple[int, Any]]:
        """
        Execute a model method, yielding each batch's result as it completes.

        Results are yielded in input order. Nothing is retained between yields,
        so a caller that consumes and discards each batch holds memory
        proportional to one batch rather than to the whole dataset.

        Args:
            method_name: Name of the method to call.
            sequences: Input sequences to process.
            batch_size: Per-device batch size.
            show_progress: Whether to show a tqdm progress bar.
            progress_desc: Optional custom description for the progress bar.
            **method_kwargs: Additional arguments for the method.

        Yields:
            Tuples of (batch_index, result), in ascending batch_index order.
        """
        self._ensure_initialized()

        batches = [
            sequences[i : i + batch_size]
            for i in range(0, len(sequences), batch_size)
        ]
        if not batches:
            return

        if self._is_single_device:
            yield from self._iter_single(
                method_name, batches, show_progress, progress_desc, **method_kwargs
            )
        else:
            yield from self._iter_multi(
                method_name, batches, show_progress, progress_desc, **method_kwargs
            )

    def _iter_single(
        self,
        method_name: str,
        batches: list[list[AntibodySequence]],
        show_progress: bool,
        progress_desc: str | None,
        **method_kwargs: Any,
    ) -> Iterator[tuple[int, Any]]:
        """Run batches in-process (no subprocess, no shared memory)."""
        from tqdm.auto import tqdm

        method = getattr(self._local_model, method_name)
        pbar = tqdm(
            total=len(batches),
            desc=progress_desc or "Processing",
            disable=not show_progress,
        )
        try:
            for batch_idx, batch in enumerate(batches):
                result = method(batch, **method_kwargs)
                pbar.update(1)
                yield batch_idx, result
        finally:
            pbar.close()

    def _iter_multi(
        self,
        method_name: str,
        batches: list[list[AntibodySequence]],
        show_progress: bool,
        progress_desc: str | None,
        **method_kwargs: Any,
    ) -> Iterator[tuple[int, Any]]:
        """
        Run batches across worker processes with a bounded submission window.

        At most `num_workers * submission_window` batches are outstanding at any
        moment. Each completed batch releases exactly one new submission, sent
        back to the worker that just reported in - which both bounds shared
        memory use and balances load toward whichever device is keeping up.
        """
        from tqdm.auto import tqdm

        from ablms.exceptions import WorkerError

        total = len(batches)
        num_workers = len(self._workers)

        task_worker: dict[int, int] = {}
        pending: dict[int, Any] = {}
        next_to_submit = 0
        next_to_yield = 0

        def submit(batch_idx: int, worker_idx: int) -> None:
            task_worker[batch_idx] = worker_idx
            self._workers[worker_idx].submit_task(
                task_id=batch_idx,
                method_name=method_name,
                sequences=batches[batch_idx],
                kwargs=method_kwargs,
            )

        pbar = tqdm(
            total=total,
            desc=progress_desc or f"Processing ({num_workers} devices)",
            disable=not show_progress,
        )
        try:
            # Prime the window.
            for worker_idx in range(num_workers):
                for _ in range(self._submission_window):
                    if next_to_submit >= total:
                        break
                    submit(next_to_submit, worker_idx)
                    next_to_submit += 1

            while next_to_yield < total:
                try:
                    msg_type, task_id, data = self._result_queue.get(
                        timeout=WORKER_TIMEOUT
                    )
                except queue.Empty:
                    raise self._stalled_error() from None

                if msg_type in ("error", "fatal"):
                    self._shutdown_workers_fast()
                    raise WorkerError(
                        worker_id=data["worker_id"],
                        original_error=data["exception"],
                    )

                # Copy out of shared memory so the worker's segment is freed
                # as soon as `data` goes out of scope.
                pending[task_id] = _detach_from_shm(data)
                del data
                pbar.update(1)

                worker_idx = task_worker.pop(task_id)
                if next_to_submit < total:
                    submit(next_to_submit, worker_idx)
                    next_to_submit += 1

                while next_to_yield in pending:
                    yield next_to_yield, pending.pop(next_to_yield)
                    next_to_yield += 1
        finally:
            pbar.close()
```

Add `Iterator` to the imports at the top of the file:

```python
from collections.abc import Iterator
```

- [ ] **Step 7: Add a temporary stall handler so the tests can run**

Task 6 replaces this with a real diagnostic. For now, add this method to `MultiGPUExecutor` so `_iter_multi` resolves:

```python
    def _stalled_error(self) -> Exception:
        """Build the error raised when no worker reports within the timeout."""
        self._shutdown_workers_fast()
        return TimeoutError(
            f"No worker returned a result within {WORKER_TIMEOUT}s."
        )
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest tests/test_executor.py -v`
Expected: PASS (all tests, including the four new slow ones).

- [ ] **Step 9: Verify the existing suite still passes**

Run: `pytest tests/ -q -m "not slow"` then `pytest tests/test_embedding_memory.py -v`
Expected: all PASS. The second command re-checks Task 2's work against the rewritten executor.

- [ ] **Step 10: Format, lint, commit**

```bash
black src/ tests/ && ruff check src/ tests/
git add src/ablms/parallel/ tests/test_executor.py tests/fake_worker_model.py
git commit -m "Bound in-flight results with a submission window

The executor previously submitted every task up front and held all
results until the last one landed, so live /dev/shm grew with the
dataset. It now keeps at most num_workers * ABLMS_SUBMISSION_WINDOW
batches outstanding, clones each result out of shared memory on
receipt, and yields in input order via execute_iter(). execute() is a
thin wrapper over that generator, so there is one code path.

Completed batches are replaced on the worker that reported them, which
also balances load toward whichever device is keeping up."
```

---

### Task 5: Add the public `iter_embeddings()` API

**Files:**
- Modify: `src/ablms/core/encoder.py` (add after `_process_embeddings_batch`)
- Test: `tests/test_embedding_memory.py` (append)

**Interfaces:**
- Consumes: `MultiGPUExecutor.execute_iter` from Task 4; `_process_embeddings_batch(sequences, layer, pooling)` from Task 2.
- Produces: `EncoderAbLM.iter_embeddings(sequences, layer=-1, pooling=None, batch_size=32, show_progress=True) -> Iterator[EmbeddingOutput]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embedding_memory.py`:

```python
class TestIterEmbeddings:
    """Streaming embeddings yield per batch and agree with get_embeddings."""

    @pytest.mark.slow
    def test_yields_one_output_per_batch(self, esm2_cpu, sequences):
        outputs = list(
            esm2_cpu.iter_embeddings(sequences, batch_size=2, show_progress=False)
        )
        assert len(outputs) == 3  # 5 sequences at batch_size 2
        assert [len(o) for o in outputs] == [2, 2, 1]
        assert all(o.sequences is not None for o in outputs)
        assert outputs[0].sequences[0] is sequences[0]
        assert outputs[2].sequences[0] is sequences[4]

    @pytest.mark.slow
    def test_pooled_stream_matches_get_embeddings(self, esm2_cpu, sequences):
        streamed = torch.cat(
            [
                o.embeddings
                for o in esm2_cpu.iter_embeddings(
                    sequences, pooling="mean", batch_size=2, show_progress=False
                )
            ]
        )
        combined = esm2_cpu.get_embeddings(
            sequences, pooling="mean", batch_size=2, show_progress=False
        )
        assert torch.allclose(streamed, combined.embeddings, atol=1e-6)

    @pytest.mark.slow
    def test_token_level_stream_matches_get_embeddings(self, esm2_cpu, sequences):
        """Compare per sequence, not per tensor.

        get_embeddings pads every batch to a single global maximum length,
        while each streamed batch is padded only to its own maximum. Both
        describe the same embeddings, so compare with get_sequence_tokens(),
        which strips padding via the attention mask.
        """
        streamed = [
            tokens
            for output in esm2_cpu.iter_embeddings(
                sequences, batch_size=2, show_progress=False
            )
            for tokens in output
        ]
        combined = esm2_cpu.get_embeddings(
            sequences, batch_size=2, show_progress=False
        )

        assert len(streamed) == len(sequences)
        for i, tokens in enumerate(streamed):
            assert torch.allclose(tokens, combined.get_sequence_tokens(i), atol=1e-6)

    @pytest.mark.slow
    def test_validation_is_eager(self, esm2_cpu):
        """Invalid input must raise on call, not on first next()."""
        from ablms import PairedSequenceError

        paired = AntibodySequence(heavy=HEAVY_CHAINS[0], light="DIQMTQSPSSLSASVGDRV")
        with pytest.raises(PairedSequenceError):
            esm2_cpu.iter_embeddings([paired], show_progress=False)

    @pytest.mark.slow
    def test_empty_input_yields_nothing(self, esm2_cpu):
        assert list(esm2_cpu.iter_embeddings([], show_progress=False)) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_embedding_memory.py::TestIterEmbeddings -v`
Expected: FAIL with `AttributeError: 'ESM2' object has no attribute 'iter_embeddings'`.

- [ ] **Step 3: Implement `iter_embeddings`**

Add `from collections.abc import Iterator` to the imports in `src/ablms/core/encoder.py`. Insert these two methods immediately after `_process_embeddings_batch`:

```python
    def iter_embeddings(
        self,
        sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
        layer: int = -1,
        pooling: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> Iterator[EmbeddingOutput]:
        """
        Stream embeddings one batch at a time.

        Unlike get_embeddings(), nothing is accumulated: each batch is yielded
        as soon as it is ready and released once the caller is done with it.
        Use this when the full token-level output for the dataset would not fit
        in memory, writing each batch to HDF5, zarr, or npy as it arrives.

        Args:
            sequences: Input sequences in various formats.
            layer: Layer index to extract embeddings from (-1 for last layer).
            pooling: Optional pooling strategy applied within each batch.
                Valid options: "mean", "max", "cls", "first", "last".
            batch_size: Batch size for processing (per GPU when using multi-GPU).
            show_progress: Whether to show a progress bar.

        Yields:
            One EmbeddingOutput per batch, in input order. Each carries its own
            slice of the input sequences and its own token offsets, so batches
            are self-describing.

        Raises:
            PairedSequenceError: If paired sequences are provided but the model
                does not support them.
            SequenceTooLongError: If a sequence exceeds the model's max length.

        Example:
            >>> for batch in model.iter_embeddings(sequences, pooling="mean"):
            ...     writer.append(batch.embeddings.numpy())
        """
        # Validate eagerly rather than on first next(), so bad input fails at
        # the call site.
        sequences = self._normalize_input(sequences)
        self._validate_input(sequences)

        return self._iter_embeddings(
            sequences=sequences,
            layer=layer,
            pooling=pooling,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def _iter_embeddings(
        self,
        sequences: list[AntibodySequence],
        layer: int,
        pooling: str | None,
        batch_size: int,
        show_progress: bool,
    ) -> Iterator[EmbeddingOutput]:
        """Generator backing iter_embeddings(); assumes validated input."""
        if not sequences:
            return

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
                layer=layer,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_embedding_memory.py::TestIterEmbeddings -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the whole embedding suite**

Run: `pytest tests/test_embedding_memory.py tests/test_executor.py -v`
Expected: all PASS.

- [ ] **Step 6: Format, lint, commit**

```bash
black src/ tests/ && ruff check src/ tests/
git add src/ablms/core/encoder.py tests/test_embedding_memory.py
git commit -m "Add iter_embeddings() for streaming per-batch output

Yields one EmbeddingOutput per batch in input order without
accumulating, so token-level runs larger than memory can be written
incrementally. Input is validated eagerly rather than at first next()."
```

---

### Task 6: Report shared-memory exhaustion instead of hanging

The `RuntimeError: unable to allocate shared memory(shm)` is raised in the queue's feeder thread, outside the `try/except` in `worker_main` that wraps `_execute_task` (`worker.py:63-73`), so it cannot be caught at its origin. Detect the symptom instead: a timeout while every worker is still alive.

**Files:**
- Modify: `src/ablms/exceptions.py` (append)
- Modify: `src/ablms/__init__.py` (export)
- Modify: `src/ablms/parallel/executor.py` (replace `_stalled_error` from Task 4 Step 7)
- Test: `tests/test_executor.py` (append)

**Interfaces:**
- Consumes: `MultiGPUError` from `ablms.exceptions`.
- Produces: `SharedMemoryError(MultiGPUError)`, exported from `ablms`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_executor.py`:

```python
class TestStalledWorkerDiagnostics:
    """A result-queue timeout must name shared memory as the likely cause."""

    def test_shared_memory_error_is_exported(self):
        from ablms import SharedMemoryError
        from ablms.exceptions import MultiGPUError

        assert issubclass(SharedMemoryError, MultiGPUError)

    @pytest.mark.slow
    def test_stall_message_is_actionable(self):
        from ablms import SharedMemoryError

        executor = _cpu_executor(1)
        # Force the multi-device path with a single worker, then make the
        # timeout immediate so no result can arrive in time.
        executor._is_single_device = False
        try:
            executor._ensure_initialized()
            error = executor._stalled_error()
        finally:
            executor.shutdown()

        assert isinstance(error, SharedMemoryError)
        message = str(error)
        assert "shm-size" in message
        assert "batch_size" in message
        assert "iter_embeddings" in message
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_executor.py::TestStalledWorkerDiagnostics -v`
Expected: FAIL with `ImportError: cannot import name 'SharedMemoryError' from 'ablms'`.

- [ ] **Step 3: Define the exception**

Append to `src/ablms/exceptions.py`:

```python
class SharedMemoryError(MultiGPUError):
    """Raised when workers appear unable to return results via shared memory."""

    pass
```

`MultiGPUError` is the right parent rather than `WorkerError`, which takes a
`(worker_id, original_error)` constructor and presumes a single identifiable
failing worker — there is none here.

- [ ] **Step 4: Export it**

In `src/ablms/__init__.py`, add `SharedMemoryError` to the `from ablms.exceptions import (...)` block (after `DeviceError` on line 64) and add `"SharedMemoryError"` to `__all__` (after `"DeviceError"` on line 117).

- [ ] **Step 5: Replace the placeholder stall handler**

In `src/ablms/parallel/executor.py`, replace the `_stalled_error` stub added in Task 4 with:

```python
    def _stalled_error(self) -> Exception:
        """
        Build the error raised when no worker reports within the timeout.

        The shared-memory failure that usually causes this is raised inside the
        result queue's feeder thread, which is outside the worker's own
        exception handling, so it never arrives as a WorkerError. What the
        parent observes is simply a result that never comes while every worker
        process is still healthy.

        Returns:
            SharedMemoryError when all workers are alive, MultiGPUError when
            one or more has died silently.
        """
        import shutil

        from ablms.exceptions import MultiGPUError, SharedMemoryError

        dead = [w.worker_id for w in self._workers if not w.is_alive]
        self._shutdown_workers_fast()

        if dead:
            return MultiGPUError(
                f"No result received within {WORKER_TIMEOUT}s and worker(s) "
                f"{dead} are no longer running. The worker process died without "
                f"reporting an error, which usually means it was killed by the "
                f"OS (out of memory) or crashed in a native extension."
            )

        try:
            free_gb = shutil.disk_usage("/dev/shm").free / 1e9
            shm_note = f"{free_gb:.2f} GB free on /dev/shm"
        except OSError:
            shm_note = "/dev/shm could not be inspected"

        return SharedMemoryError(
            f"No result received from any worker within {WORKER_TIMEOUT}s, but "
            f"every worker is still alive ({shm_note}).\n\n"
            f"This almost always means a worker could not allocate shared "
            f"memory to hand its result back. torch.multiprocessing transfers "
            f"CPU tensors through /dev/shm, and that failure is raised in the "
            f"queue's feeder thread, so it cannot be reported as a worker "
            f"error - the result is simply lost.\n\n"
            f"Remedies, roughly in order of effectiveness:\n"
            f"  - Pass pooling= to get_embeddings() to reduce each batch before "
            f"it is transferred.\n"
            f"  - Stream with iter_embeddings() instead of accumulating.\n"
            f"  - Reduce batch_size.\n"
            f"  - If running in a container, raise --shm-size (the Docker "
            f"default is only 64 MB).\n"
            f"  - Set ABLMS_DISABLE_MULTI_GPU=true to avoid the queue entirely."
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_executor.py::TestStalledWorkerDiagnostics -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Run the full suite**

Run: `pytest tests/ -v`
Expected: all PASS, including slow tests.

- [ ] **Step 8: Format, lint, commit**

```bash
black src/ tests/ && ruff check src/ tests/
git add src/ablms/exceptions.py src/ablms/__init__.py src/ablms/parallel/executor.py tests/test_executor.py
git commit -m "Raise SharedMemoryError instead of hanging on a lost result

A worker that cannot allocate shared memory fails in the queue feeder
thread, outside its own exception handling, so the result vanishes and
the parent blocked for WORKER_TIMEOUT before surfacing a bare
queue.Empty. It now reports free space on /dev/shm and lists remedies."
```

---

### Task 7: Document the new behavior

**Files:**
- Modify: `README.md:344-373` (Multi-GPU Parallelism section)
- Modify: `CLAUDE.md` (Important Patterns section)

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Document memory behavior in the README**

In `README.md`, replace the "Key features" list in the "Multi-GPU Parallelism" section (lines 363-367) with:

```markdown
Key features:
- **Automatic detection**: Uses all available GPUs by default
- **Lazy initialization**: Worker processes spawn on first inference call
- **Single-GPU optimization**: No subprocess overhead when using one device
- **Bounded memory**: At most a few batches per GPU are in flight at once, so
  memory use does not grow with dataset size
- **Progress tracking**: Built-in tqdm progress bar for all inference methods
```

Then insert a new subsection immediately after that list, before the "Disable the progress bar" paragraph:

````markdown
#### Large datasets

Results travel from worker processes to the parent through shared memory
(`/dev/shm`), so what matters for very large runs is how much each batch
carries. Two things keep that bounded.

**Pool inside the batch.** When you pass `pooling=`, the reduction happens on
the GPU before the batch is transferred, so the full token-level tensor is
never materialized:

```python
# Each batch transfers [batch_size, hidden_dim], not
# [batch_size, seq_len, hidden_dim] - roughly 250x smaller at typical lengths.
embeddings = model.get_embeddings(sequences, pooling="mean", batch_size=64)
```

**Stream token-level output.** When you need per-residue embeddings for more
sequences than fit in memory, `iter_embeddings()` yields one batch at a time,
in input order, and retains nothing:

```python
import h5py

with h5py.File("embeddings.h5", "w") as f:
    for i, batch in enumerate(model.iter_embeddings(sequences, batch_size=64)):
        for j, tokens in enumerate(batch):  # iterating strips padding
            f.create_dataset(f"seq_{i * 64 + j}", data=tokens.numpy())
```

If a run still exhausts shared memory, `ablms` raises `SharedMemoryError` with
the current `/dev/shm` free space and suggested remedies. Inside a container the
usual cause is Docker's 64 MB default; raise it with `--shm-size=8g`.

Two environment variables tune this:

| Variable | Default | Effect |
| --- | --- | --- |
| `ABLMS_SUBMISSION_WINDOW` | `2` | Batches in flight per GPU. Lower to reduce shared memory use, raise to hide scheduling latency. |
| `ABLMS_WORKER_TIMEOUT` | `300` | Seconds to wait for a batch before raising `SharedMemoryError`. |
````

- [ ] **Step 2: Add the streaming example to the embeddings quickstart**

In `README.md`, inside the "Getting Embeddings" fenced block, after the `print(pooled.embeddings.shape)  # [2, 768]` line (line 103) and before the closing fence (line 104), add:

```markdown
# Stream batches instead of accumulating, for datasets larger than memory
for batch in model.iter_embeddings(sequences, pooling="mean", batch_size=64):
    ...  # batch is an EmbeddingOutput covering just this batch
```

- [ ] **Step 3: Record the pattern in CLAUDE.md**

In `CLAUDE.md`, under "Important Patterns", add these two bullets after the "Batch processing methods" bullet:

```markdown
- **Reduce before transfer**: `_process_*_batch()` methods run in worker processes and
  return through a queue backed by `/dev/shm`. Any reduction that shrinks the result
  (pooling, scoring) belongs inside the batch method, before `.cpu()`, not after the
  executor concatenates. See `EncoderAbLM._process_embeddings_batch`.
- **Streaming variants**: `MultiGPUExecutor.execute_iter()` yields `(batch_index, result)`
  in input order with a bounded submission window; `execute()` is a thin wrapper that
  combines it. Public streaming APIs (e.g. `iter_embeddings()`) build on `execute_iter`.
```

- [ ] **Step 4: Verify the README examples are accurate**

Run: `python -c "
from ablms.encoders import ESM2
m = ESM2(devices='cpu', model_id='facebook/esm2_t6_8M_UR50D')
from ablms import AntibodySequence
seqs = [AntibodySequence(heavy='EVQLVESGGGLVQPGGSLRLSCAAS') for _ in range(5)]
for b in m.iter_embeddings(seqs, pooling='mean', batch_size=2, show_progress=False):
    print(b)
"`
Expected: three `EmbeddingOutput(shape=[2x320], ...)`-style lines (the last with batch size 1), no errors.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Document bounded-memory embedding extraction"
```

---

## Self-Review

**Spec coverage.** Change 1 (per-batch pooling on device, plus the
`_pad_tensors_to_max_length` guard) is Task 2. Change 2 (submission window,
clone on receipt) is Tasks 3 and 4. Change 3 (`execute_iter` refactor and
`iter_embeddings`) is Tasks 4 and 5. Change 4 (`SharedMemoryError` and the
timeout diagnosis) is Task 6. The spec's testing section maps to: pooling
equivalence in Task 2 Step 1, executor windowing and ordering in Task 4 Step 2,
end-to-end in Task 2 Step 7 and Task 5 Step 1, and marker registration in
Task 1. The spec's non-goals (disk offload, `float16`, changes to
`get_hidden_states`/`get_attention`/`get_logits`) have no tasks, as intended —
those three methods gain the Task 4 windowing fix for free because they share
the executor, without any API change.

**Type consistency.** `_process_embeddings_batch(sequences, layer, pooling)`
is defined in Task 2 and consumed unchanged in Task 5. `execute_iter` yields
`tuple[int, Any]` in Task 4 and is unpacked as `batch_idx, (embeddings, mask,
offsets)` in Task 5, matching the three-tuple that Task 2 returns.
`_detach_from_shm` is defined in Task 3 and called in Task 4.
`_stalled_error()` is introduced as a stub in Task 4 Step 7 and replaced in
Task 6 Step 5 with the same name and return type. `SharedMemoryError` is
defined in Task 6 Step 3 and referenced only afterward.

**Ordering note.** Task 4 temporarily introduces a `TimeoutError` stub so its
tests can run standalone; Task 6 replaces it. If executing out of order, Task 6
must follow Task 4.
