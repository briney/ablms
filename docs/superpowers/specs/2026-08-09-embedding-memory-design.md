# Bounding memory in `get_embeddings()`

**Date:** 2026-08-09
**Status:** Approved, ready for implementation planning

## Problem

Large multi-GPU embedding runs fail with repeated tracebacks of the form:

```
RuntimeError: unable to allocate shared memory(shm) for file </torch_3018_2243491522_369>:
Resource temporarily unavailable (11)
  File ".../multiprocessing/queues.py", line 264, in _feed
  File ".../torch/multiprocessing/reductions.py", line 615, in reduce_storage
    fd, size = storage._share_fd_cpu_()
```

This is neither CUDA out-of-memory nor an ordinary host-RAM `MemoryError`. It is exhaustion
of POSIX shared memory in the worker-to-parent result path.

### Mechanism

`EncoderAbLM._process_embeddings_batch` (`src/ablms/core/encoder.py:135`) already moves each
batch to the host with `.cpu()`, and every `_forward_embeddings` implementation runs under
`torch.no_grad()`. GPU memory therefore does *not* accumulate across batches, in either
single-device or multi-device mode. Adding a "offload to CPU" flag would be a no-op.

What does accumulate is shared memory. Three properties of the current implementation
combine to make it unbounded:

1. **Tensors cross the queue via shared memory.** `torch.multiprocessing`'s default sharing
   strategy is `file_descriptor`. When a worker returns a CPU tensor through the result
   queue, torch does not copy the bytes into the pipe; it moves the storage into a
   `/dev/shm` segment and passes a file descriptor. The segment stays live for as long as
   any process holds the tensor.

2. **The parent holds every result until the last one lands.**
   `MultiGPUExecutor._execute_multi` submits all tasks up front (`executor.py:250-259`) and
   accumulates results into a dict (`executor.py:277-279`) that is not consumed until every
   task has completed. Live shared memory therefore grows to the size of the entire output.

3. **Pooling is applied far too late.** `get_embeddings` calls `apply_pooling` only after the
   executor has concatenated everything (`encoder.py:89-94`). A run that ultimately wants one
   vector per sequence still ships, and retains, the full token-level tensor for the whole
   dataset.

### Scale

Per-batch payload is `batch_size x padded_len x hidden_dim x 4` bytes. At the default
`batch_size=32`, a padded length of ~256 tokens, and `hidden_dim=1024`, that is **33.5 MB per
batch**. Docker's default `--shm-size` is **64 MB**, so a containerised run exhausts its
entire shared memory after roughly two batches. This matches the observed failure mode: many
errors, immediately, rather than a slow leak.

### Secondary defect: the failure is silent and untimed

The exception is raised in the queue's background feeder thread, which is outside the
`try/except` in `worker_main` that wraps `_execute_task` (`worker.py:63-73`). It cannot be
caught where it occurs, so it never becomes a `WorkerError`. The result is simply lost; the
parent continues blocking in `result_queue.get(timeout=WORKER_TIMEOUT)` and, after
`WORKER_TIMEOUT` (300 s by default, `parallel/utils.py:15`), propagates a bare `queue.Empty`
with nothing connecting it to the real cause. The raw tracebacks on stderr are visible only
because nothing is catching them.

## Goals

- Pooled runs (`pooling="mean"` and friends) must succeed on a large dataset inside a
  container with a 64 MB `/dev/shm`.
- Token-level runs must be expressible without materialising the whole dataset anywhere.
- Shared-memory failures must surface as an attributable error, not a 300-second hang.
- No change to the numerical output of `get_embeddings`.

## Non-goals

- Built-in disk offload (an `offload_dir=` that memory-maps batches and returns a lazy
  handle). Rejected in favour of a streaming API: it would make `ablms` own a storage format
  and its lifecycle, and the streaming API lets callers use whatever their pipeline already
  uses.
- A lower output dtype (`output_dtype=torch.float16`). Only a 2x win, and it composes with
  everything here, so it can be added later if wanted.
- Changes to `get_hidden_states`, `get_attention`, or `get_logits`. They share the same
  executor and so inherit the Change 2 fix, but they get no new API surface here.

## Design

### Change 1 — pool inside the batch, on the accelerator

`get_embeddings` passes `pooling` through to `_process_embeddings_batch`, which applies it
before the existing `.cpu()` call. The large `[B, L, D]` tensor is then never copied to host
memory and never crosses the queue.

```python
def _process_embeddings_batch(self, sequences, layer=-1, pooling=None):
    formatted = self._format_for_model(sequences)
    tokenized = self._tokenize(formatted)
    offsets = self._compute_token_offsets(sequences, tokenized)
    embeddings, mask = self._forward_embeddings(tokenized, layer)

    if pooling is not None:
        embeddings = apply_pooling(embeddings, strategy=pooling, attention_mask=mask)
        mask = None  # no longer meaningful for a [B, D] result

    embeddings = embeddings.cpu()
    if mask is not None:
        mask = mask.cpu()
    return embeddings, mask, offsets
```

`get_embeddings` correspondingly stops calling `apply_pooling` on the combined result and
instead forwards `pooling` as a method kwarg. When `pooling` is not `None` it builds the
`EmbeddingOutput` with `embeddings=pooled`, `pooled=pooled` (the same tensor object, so no
duplicated memory), and `attention_mask=None`, exactly as today.

**This does not change results.** Batches are currently zero-padded to a global maximum
length before pooling, and every strategy is invariant to right-padding given the mask:
`mean_pooling` and `max_pooling` consume the mask directly, `cls_pooling` reads index 0, and
`last_pooling` derives its index from `attention_mask.sum(dim=1) - 1`. In the case where a
model returns `mask=None`, per-batch pooling is strictly *more* correct than the status quo,
because today the global zero-padding is folded into the mean or max.

Payload per batch drops from `B x L x D x 4` to `B x D x 4` — 33.5 MB to 131 KB at the
numbers above, a 256x reduction. This alone brings pooled runs inside a 64 MB `/dev/shm`.

**Related fix.** `_pad_tensors_to_max_length` (`executor.py:378`) pads any tensor with
`dim >= 2` along dimension 1. A pooled `[B, D]` tensor survives it only because `hidden_dim`
is constant across batches, so the computed max equals `D` and no padding occurs. That is
accidental correctness. Make it explicit: skip padding for results with `dim < 3`.

### Change 2 — bound the in-flight set, and clone out of shared memory

Two changes inside `MultiGPUExecutor`, both of which help the token-level runs that pooling
cannot shrink.

**Submission window.** Replace up-front submission of all tasks with a sliding window:
submit `window` batches per worker (default 2), then release one more each time a result
arrives. This caps live shared memory at `num_workers x window` payloads regardless of
dataset size, and stops workers racing arbitrarily far ahead of the consumer.

**Clone on receipt.** A tensor pulled off the result queue is still backed by its `/dev/shm`
segment and remains so for as long as it is referenced. Cloning it copies the data into
ordinary heap memory and lets the segment be released when the received tensor is dropped:

```python
def _detach_from_shm(obj):
    """Copy queue-received tensors out of shared memory so segments can be freed."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_detach_from_shm(o) for o in obj)
    if isinstance(obj, list):
        return [_detach_from_shm(o) for o in obj]
    return obj
```

This converts shared-memory pressure into ordinary RAM pressure, which is the whole point:
`/dev/shm` is 64 MB in the failing container, while RAM is not. The cost is one memcpy per
batch, negligible against a forward pass. It matters most for `iter_embeddings`, where a
consumer writing to disk may hold a batch long enough to pin its segment.

### Change 3 — `iter_embeddings()`

Refactor `MultiGPUExecutor.execute()` onto a new generator, `execute_iter()`, which yields
`(batch_index, result)` pairs as they become available. `execute()` becomes a thin wrapper
that collects and combines them, so there is one implementation of the window and ordering
logic rather than two. Both `_execute_single` and `_execute_multi` are expressed as
generators; the single-device path involves no subprocess and therefore no shared memory at
all, but shares the same interface.

Public API on `EncoderAbLM`:

```python
def iter_embeddings(
    self,
    sequences: str | AntibodySequence | list[str] | list[AntibodySequence],
    layer: int = -1,
    pooling: str | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Iterator[EmbeddingOutput]:
    """Yield one EmbeddingOutput per batch, in input order, without accumulating."""
```

Each yielded `EmbeddingOutput` carries that batch's slice of `sequences` and its own
`token_offsets`, so it is self-describing. Callers write to HDF5, zarr, npy, or anything
else; `ablms` retains nothing.

**Ordering.** Results are yielded in input order using a reorder buffer. This is safe
specifically because the submission window bounds how far ahead any worker can run, so the
buffer is bounded by `num_workers x window` and cannot silently grow back into the problem
being fixed.

`get_embeddings` is then definitionally the concatenation of what `iter_embeddings` yields,
which gives a direct equivalence test.

### Change 4 — fail loudly on shared-memory exhaustion

The feeder-thread exception cannot be caught at its origin, so detect the symptom instead.
When `result_queue.get(timeout=WORKER_TIMEOUT)` raises `queue.Empty` while worker processes
are still alive, raise a new `SharedMemoryError(MultiGPUError)` in `ablms/exceptions.py`.
`MultiGPUError` is the right parent rather than `WorkerError`, because `WorkerError` has a
custom `__init__(worker_id, original_error)` signature and there is no single failing worker
to attribute here. Its message should:

- names shared-memory exhaustion as the likely cause,
- reports free bytes on `/dev/shm` (via `shutil.disk_usage("/dev/shm")`),
- and lists the remedies: increase the container's `--shm-size`, reduce `batch_size`, pass
  `pooling=`, or switch to `iter_embeddings`.

If the workers are *not* alive, the existing worker-death reporting applies and should be
preferred.

## Testing

Following the project convention of real models and real data over synthetic fixtures.

1. **Pooling equivalence** (`tests/test_pooling.py`, fast, no model). Pool a globally-padded
   stack post-hoc; pool the same underlying data split into ragged per-batch tensors; assert
   the two agree for `mean`, `max`, `cls`, `first`, and `last`. This is the guarantee that
   Change 1 does not alter results.

2. **Executor windowing and ordering** (`tests/test_executor.py`, new). The multi-process
   path is testable without a second GPU: `resolve_devices(["cpu", "cpu"])` returns two
   devices, so `MultiGPUExecutor` takes the `_execute_multi` branch and spawns two real
   worker processes. Assert that results are yielded in input order, that the number of
   outstanding submitted-but-unreturned tasks never exceeds `num_workers x window`, and that
   `execute()` and `execute_iter()` produce identical combined output.

3. **End-to-end** (`@pytest.mark.slow`). Using `facebook/esm2_t6_8M_UR50D` on CPU, the
   pattern already established in `tests/test_esm2.py`: assert that
   `get_embeddings(..., pooling="mean")` returns the same values as the pre-change
   implementation on a multi-batch input (capture a reference tensor before the change and
   compare with `torch.allclose`), and that `iter_embeddings` agrees with `get_embeddings`.

   For the pooled case that second comparison is a direct `torch.cat` of the yielded `[B, D]`
   tensors. For the token-level case it is *not*, because `get_embeddings` pads all batches to
   a single global maximum length while each yielded batch is padded only to its own maximum.
   Compare per sequence instead, using `EmbeddingOutput.get_sequence_tokens(i)`, which strips
   padding via the attention mask and is therefore invariant to which padding scheme produced
   it.

4. **Marker registration.** `slow` is used throughout the suite but never declared in
   `[tool.pytest.ini_options]`, so every run emits `PytestUnknownMarkWarning`. Register it.

## Interim workaround

Until this lands, the only effective levers avoid the queue entirely: run single-device
(`ABLMS_DISABLE_MULTI_GPU=true`, or pass one device explicitly), or restart the container
with a larger `--shm-size`. Passing `pooling=` does not help today, because the reduction
happens after concatenation rather than before the queue.
