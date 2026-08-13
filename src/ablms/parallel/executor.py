"""Multi-GPU executor for parallel inference."""

from __future__ import annotations

import dataclasses
import queue
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.multiprocessing as mp

from ablms.parallel.utils import DEFAULT_SUBMISSION_WINDOW, WORKER_TIMEOUT
from ablms.parallel.worker import WorkerHandle

if TYPE_CHECKING:
    from multiprocessing.context import SpawnContext

    from ablms.core.base import BaseAbLM
    from ablms.core.sequence import AntibodySequence


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
        obj: A value received from a worker: a tensor, a tuple/list/dataclass
            that may contain tensors, or anything else.

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
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Output containers such as MaskScanOutput hold their tensors as
        # attributes, so recursing into fields is the only way to reach them.
        return dataclasses.replace(
            obj,
            **{
                f.name: _detach_from_shm(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    return obj


class _TensorColumn:
    """
    One tuple position whose per-batch value is a [batch, ...] tensor.

    Batches are written straight into a buffer sized for the whole run, so the
    result is never held twice. Allocating that buffer up front - before any
    batch exists - is what makes the memory bounded: a freed batch goes back to
    the allocator's free list rather than to the OS, so it can only be reused by
    a *later* allocation. A buffer allocated afterwards, as `torch.cat` does,
    cannot reuse it and the peak is the sum of both.

    Batches whose trailing shape does not match the first one (token-level
    output, which pads per batch) cannot stream into a fixed buffer. Those fall
    back to collecting chunks and padding at the end - the original behaviour,
    and acceptable there because token-level batches are large enough to be
    mapped individually and so do return to the OS when freed.
    """

    def __init__(self, total_rows: int, first: torch.Tensor) -> None:
        self._buffer: torch.Tensor | None = torch.empty(
            (total_rows, *first.shape[1:]), dtype=first.dtype
        )
        self._rows = 0
        self._chunks: list[torch.Tensor] | None = None

    def add(self, tensor: torch.Tensor) -> None:
        if self._chunks is None:
            assert self._buffer is not None
            rows = tensor.shape[0]
            fits = (
                tensor.shape[1:] == self._buffer.shape[1:]
                and self._rows + rows <= self._buffer.shape[0]
            )
            if fits:
                self._buffer[self._rows : self._rows + rows] = tensor
                self._rows += rows
                return

            # Ragged batch, or more rows than there were input sequences. Keep
            # what is written so far as one chunk and collect the rest, so the
            # final pad-and-concatenate runs over a few chunks rather than one
            # per batch. Cloning releases the oversized buffer; the mismatch
            # almost always shows up on the second batch, so little is copied.
            self._chunks = [self._buffer[: self._rows].clone()] if self._rows else []
            self._buffer = None

        self._chunks.append(tensor)

    def finish(self, executor: MultiGPUExecutor) -> torch.Tensor:
        if self._chunks is None:
            assert self._buffer is not None
            return self._buffer[: self._rows]
        return executor._concat_batches(self._chunks)


class _ListColumn:
    """One tuple position holding a per-batch list of non-tensors (offsets)."""

    def __init__(self) -> None:
        self._items: list[Any] = []

    def add(self, value: list[Any]) -> None:
        self._items.extend(value)

    def finish(self, executor: MultiGPUExecutor) -> list[Any]:
        return self._items


class _NoneColumn:
    """One tuple position that is None in every batch (an absent mask)."""

    def add(self, value: Any) -> None:
        pass

    def finish(self, executor: MultiGPUExecutor) -> None:
        return None


class _OpaqueColumn:
    """A position with no streaming form; collected and combined at the end."""

    def __init__(self) -> None:
        self._values: list[Any] = []

    def add(self, value: Any) -> None:
        self._values.append(value)

    def finish(self, executor: MultiGPUExecutor) -> Any:
        return executor._combine_column(self._values)


class _ResultAccumulator:
    """
    Combines batch results into preallocated storage as they arrive.

    The structure of a result is not known until the first batch lands, so the
    columns are chosen then and reused for every subsequent batch.
    """

    def __init__(self, total_rows: int) -> None:
        self._total_rows = total_rows
        self._columns: list[Any] | None = None
        self._is_tuple = False

    def add(self, result: Any) -> None:
        values = result if isinstance(result, tuple) else (result,)
        if self._columns is None:
            self._is_tuple = isinstance(result, tuple)
            self._columns = [self._make_column(value) for value in values]
        for column, value in zip(self._columns, values):
            column.add(value)

    def _make_column(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            return _TensorColumn(self._total_rows, value)
        if value is None:
            return _NoneColumn()
        if isinstance(value, list) and not (
            value and isinstance(value[0], torch.Tensor)
        ):
            return _ListColumn()
        return _OpaqueColumn()

    def finish(self, executor: MultiGPUExecutor) -> Any:
        if self._columns is None:
            return None
        combined = [column.finish(executor) for column in self._columns]
        return tuple(combined) if self._is_tuple else combined[0]


class MultiGPUExecutor:
    """
    Manages a pool of worker processes for multi-GPU inference.

    Features:
    - Lazy initialization (workers spawn on first inference call)
    - Graceful single-GPU fallback (no subprocess overhead)
    - Automatic work distribution with order preservation
    - Progress tracking with tqdm

    The executor is owned by the model instance and handles:
    - Worker lifecycle management
    - Work distribution and load balancing
    - Result collection and ordering
    - Error propagation

    Example:
        >>> executor = MultiGPUExecutor(IgBERT, {}, [device0, device1])
        >>> results = executor.execute(
        ...     method_name="_process_embeddings_batch",
        ...     sequences=sequences,
        ...     batch_size=32,
        ...     show_progress=True,
        ...     layer=-1,
        ... )
    """

    def __init__(
        self,
        model_class: type[BaseAbLM],
        model_init_kwargs: dict[str, Any],
        devices: list[torch.device],
    ):
        """
        Initialize executor (does NOT spawn workers yet).

        Args:
            model_class: The model class to instantiate in workers.
            model_init_kwargs: Arguments for model instantiation (minus device).
            devices: List of devices to use.
        """
        self._model_class = model_class
        self._model_init_kwargs = model_init_kwargs
        self._devices = devices
        self._is_single_device = len(devices) == 1

        # Maximum batches outstanding per worker. Bounds how many results can
        # sit in shared memory at once, independent of dataset size.
        self._submission_window = DEFAULT_SUBMISSION_WINDOW

        # Worker management (initialized lazily)
        self._workers: list[WorkerHandle] | None = None
        self._result_queue: mp.Queue | None = None
        self._mp_context: SpawnContext | None = None

        # Single-device mode: local model (initialized lazily)
        self._local_model: BaseAbLM | None = None

    @property
    def is_initialized(self) -> bool:
        """Check if workers have been spawned or local model created."""
        return self._workers is not None or self._local_model is not None

    @property
    def num_devices(self) -> int:
        """Get the number of devices."""
        return len(self._devices)

    def _ensure_initialized(self) -> None:
        """Lazily initialize workers on first use."""
        if self.is_initialized:
            return

        if self._is_single_device:
            # Single GPU: no subprocess, just create model directly
            self._local_model = self._model_class(
                devices=self._devices[0],
                **self._model_init_kwargs,
            )
        else:
            # Multi-GPU: spawn worker processes
            self._spawn_workers()

    def _spawn_workers(self) -> None:
        """Spawn worker processes for each GPU."""
        from ablms.exceptions import WorkerInitializationError

        # Use 'spawn' context for CUDA compatibility
        self._mp_context = mp.get_context("spawn")
        self._result_queue = self._mp_context.Queue()

        self._workers = []
        for i, device in enumerate(self._devices):
            worker = WorkerHandle.spawn(
                worker_id=i,
                device=device,
                model_class=self._model_class,
                model_init_kwargs=self._model_init_kwargs,
                result_queue=self._result_queue,
                ctx=self._mp_context,
            )
            self._workers.append(worker)

        # Wait for all workers to signal ready
        ready_count = 0
        errors = []

        while ready_count < len(self._devices):
            try:
                msg_type, identifier, data = self._result_queue.get(
                    timeout=WORKER_TIMEOUT
                )

                if msg_type == "ready":
                    ready_count += 1
                elif msg_type == "fatal":
                    errors.append(data)
                    break

            except Exception as e:
                errors.append({"exception": e, "traceback": str(e)})
                break

        if errors:
            self.shutdown()
            error = errors[0]
            raise WorkerInitializationError(
                f"Worker initialization failed: {error['exception']}\n"
                f"{error.get('traceback', '')}"
            )

    def _require_workers(self) -> list[WorkerHandle]:
        """The worker handles, which exist only after initialization."""
        if self._workers is None:
            raise RuntimeError(
                "Workers have not been initialized; call _ensure_initialized() first."
            )
        return self._workers

    def _require_result_queue(self) -> mp.Queue:
        """The result queue, which exists only after initialization."""
        if self._result_queue is None:
            raise RuntimeError(
                "Result queue not initialized; call _ensure_initialized() first."
            )
        return self._result_queue

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

        Note:
            Batches are combined as they arrive rather than collected and
            concatenated at the end. Collecting first means holding every batch
            while the combined result is built alongside it, which peaks at
            twice the result - and does so *after* the progress bar has
            finished, which is where large runs die.
        """
        accumulator = _ResultAccumulator(total_rows=len(sequences))
        for _, result in self.execute_iter(
            method_name,
            sequences,
            batch_size,
            show_progress=show_progress,
            progress_desc=progress_desc,
            **method_kwargs,
        ):
            accumulator.add(result)
            # Release the batch before the next one is produced, so its memory
            # is available for reuse rather than accumulating.
            del result
        return accumulator.finish(self)

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
            sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)
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
        back to the worker that just reported in - which balances load toward
        whichever device is keeping up.

        The window bounds the number of *outstanding* results, and therefore how
        much shared memory is live at once; it does not bound total host memory.
        Results arriving out of order are held in a reorder buffer until the
        batch before them lands, and that buffer grows with inter-device skew: a
        single slow batch pins the yield cursor while every other worker keeps
        completing. Buffered results are heap-resident clones, not shared
        memory, so this does not reintroduce the /dev/shm problem.
        """
        from tqdm.auto import tqdm

        from ablms.exceptions import WorkerError

        total = len(batches)
        workers = self._require_workers()
        result_queue = self._require_result_queue()
        num_workers = len(workers)

        task_worker: dict[int, int] = {}
        pending: dict[int, Any] = {}
        next_to_submit = 0
        next_to_yield = 0

        def submit(batch_idx: int, worker_idx: int) -> None:
            task_worker[batch_idx] = worker_idx
            workers[worker_idx].submit_task(
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
                    msg_type, task_id, data = result_queue.get(timeout=WORKER_TIMEOUT)
                except queue.Empty:
                    error = self._stalled_error()
                    # The workers are gone, so nothing is left to reclaim.
                    task_worker.clear()
                    raise error from None

                if msg_type in ("error", "fatal"):
                    self._shutdown_workers_fast()
                    task_worker.clear()
                    raise WorkerError(
                        worker_id=data["worker_id"],
                        original_error=data["exception"],
                    )

                if msg_type != "result":
                    # A stray or stale message - e.g. a late "ready" from a
                    # respawned worker. Treating it as a result would look up an
                    # unknown task id and fail with a bare KeyError, so skip it.
                    continue

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
            self._drain_outstanding(len(task_worker))

    def _drain_outstanding(self, count: int) -> None:
        """
        Discard results for tasks that are still in flight.

        A consumer that abandons the generator partway - by breaking out of the
        loop, or via `close()` - leaves the tasks currently inside the window
        running. Their results would otherwise sit unread in the shared result
        queue, holding shared memory and, worse, being mistaken for this run's
        results by the next call. `count` is bounded by
        `num_workers * submission_window`, so this always terminates.

        Args:
            count: Number of results still expected from the workers.
        """
        for _ in range(count):
            try:
                self._require_result_queue().get(timeout=WORKER_TIMEOUT)
            except (queue.Empty, EOFError, OSError, RuntimeError, ValueError):
                # Nothing arrived within the timeout, or the queue is already
                # torn down. A worker that died mid-flight can also surface as
                # EOFError (broken pipe) or RuntimeError (torch failing to
                # rebuild an invalidated shared-memory handle). Either way there
                # is nothing left to reclaim, and this runs in a `finally` - it
                # must not mask the real error.
                break

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

        dead = [w.worker_id for w in self._require_workers() if not w.is_alive]
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

    def _combine_column(self, elements: list[Any]) -> Any:
        """
        Combine one tuple position's per-batch values into a single result.

        This is the path for values that cannot be streamed into a preallocated
        buffer as they arrive - chiefly the per-layer list of tensors returned
        by `_process_hidden_states_batch`. Positions that can stream are handled
        by `_TensorColumn`, which never reaches here.

        Args:
            elements: One value per batch, in input order. Emptied by this call.

        Returns:
            The combined value: tensors concatenated along the batch axis, lists
            flattened, None left as None.
        """
        if not elements:
            return None

        first = elements[0]

        if isinstance(first, torch.Tensor):
            return self._concat_batches(elements)

        if isinstance(first, list):
            if first and isinstance(first[0], torch.Tensor):
                # List of tensors, one per layer (hidden states).
                layer_combined = []
                for layer_idx in range(len(first)):
                    layer_tensors = [element[layer_idx] for element in elements]
                    layer_combined.append(self._concat_batches(layer_tensors))
                elements.clear()
                return layer_combined

            flat: list[Any] = []
            for element in elements:
                flat.extend(element)
            elements.clear()
            return flat

        if first is None:
            return None

        return elements

    def _concat_batches(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate along dim 0 into one preallocated tensor, freeing inputs.

        `torch.cat` builds its output while every input is still referenced, so
        peak memory is the sum of both - 2x the result for a run whose batches
        are all live, which at dataset scale is what kills the process right
        after the progress bar completes. Writing into a preallocated output and
        releasing each batch as it is copied keeps roughly one result's worth
        resident: the output faults in page by page as the inputs are freed.

        The output is allocated with `torch.empty` rather than `torch.zeros`
        even when padding is needed, because `torch.zeros` memsets the whole
        buffer up front - making it fully resident while every input is still
        alive, which is the 2x peak this method exists to avoid. Slices that
        need padding are zeroed individually, just before they are written.

        Padding is implicit, which is why no separate padding pass is needed.
        Batches are tokenized independently and so pad to different sequence
        lengths; copying each into the leading corner of its slice leaves the
        trailing region of a shorter batch as zero fill. Which axis carries
        sequence length is never assumed - it is dim 1 for [batch, seq, hidden]
        but dim 2 for the multi-layer [batch, layers, seq, hidden] - because
        every non-batch axis is sized to the largest seen and every batch is
        written at the origin.

        Args:
            tensors: Batch tensors, all of equal rank. Emptied by this call.

        Returns:
            One tensor holding every batch, padded on non-batch axes to the
            largest size seen.

        Raises:
            ValueError: If the tensors do not all have the same rank.
        """
        ndim = tensors[0].dim()
        if any(t.dim() != ndim for t in tensors):
            ranks = sorted({t.dim() for t in tensors})
            raise ValueError(
                f"Cannot combine tensors of differing rank: got ranks {ranks}. "
                f"This usually means a worker returned an unexpected shape."
            )
        if ndim == 0:
            raise ValueError("Cannot combine zero-dimensional tensors.")

        total = sum(t.shape[0] for t in tensors)
        trailing = [max(t.shape[d] for t in tensors) for d in range(1, ndim)]
        out = torch.empty((total, *trailing), dtype=tensors[0].dtype)

        # Move to a deque and empty the caller's list, so each batch has exactly
        # one reference left and popping it is enough to free it. Holding them
        # in the original list until the end would keep every input alive
        # alongside the output, which is the peak this method avoids.
        pending = deque(tensors)
        tensors.clear()

        row = 0
        while pending:
            tensor = pending.popleft()
            rows = tensor.shape[0]

            if any(tensor.shape[d] != trailing[d - 1] for d in range(1, ndim)):
                out[row : row + rows].zero_()
            out[(slice(row, row + rows), *(slice(0, s) for s in tensor.shape[1:]))] = (
                tensor
            )

            row += rows
            del tensor

        return out

    def _shutdown_workers_fast(self) -> None:
        """
        Quickly terminate all workers and clear worker state (used on error).

        Unlike shutdown(), this skips the graceful join and drops straight to
        terminate() - the caller has already given up on a clean shutdown.
        Clearing self._workers still matters: leaving it set would make
        is_initialized keep reporting True for processes that no longer exist,
        so the next call would submit tasks nothing can consume and hang for
        another WORKER_TIMEOUT before failing again.
        """
        if self._workers is None:
            return

        for worker in self._workers:
            try:
                worker.process.terminate()
            except Exception:
                pass

        self._workers = None

    def shutdown(self) -> None:
        """Gracefully shut down all workers."""
        if self._workers is not None:
            for worker in self._workers:
                worker.shutdown()
            self._workers = None

        if self._local_model is not None:
            del self._local_model
            self._local_model = None

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass
