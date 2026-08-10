"""Multi-GPU executor for parallel inference."""

from __future__ import annotations

import dataclasses
import queue
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.multiprocessing as mp

from ablms.parallel.utils import DEFAULT_SUBMISSION_WINDOW, WORKER_TIMEOUT
from ablms.parallel.worker import WorkerHandle

if TYPE_CHECKING:
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
        self._mp_context: mp.context.SpawnContext | None = None

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
                self._result_queue.get(timeout=WORKER_TIMEOUT)
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

    def _combine_results(self, results: list[Any]) -> Any:
        """
        Combine results from multiple batches.

        Handles different result types:
        - Tuple of (embeddings, mask, offsets)
        - Tuple of (hidden_states_list, mask, offsets)
        - Tuple of (tensor, mask, offsets)
        - List results

        Args:
            results: List of batch results to combine.

        Returns:
            Combined result with tensors concatenated along batch dimension.
        """
        if not results:
            return None

        first = results[0]

        # Handle tuple results (embeddings, mask, offsets)
        if isinstance(first, tuple):
            return self._combine_tuple_results(results)

        # Handle list results
        if isinstance(first, list):
            combined = []
            for r in results:
                combined.extend(r)
            return combined

        # Handle tensor results
        if isinstance(first, torch.Tensor):
            return torch.cat(results, dim=0)

        # Unknown type - return as list
        return results

    def _pad_tensors_to_max_length(
        self, tensors: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Pad tensors to max sequence length for concatenation.

        When batches are tokenized separately, they may have different sequence
        lengths due to per-batch padding. This method pads shorter tensors with
        zeros to match the maximum sequence length across all tensors.

        Args:
            tensors: List of tensors to pad.

        Returns:
            List of tensors with uniform sequence length (dimension 1).
        """
        if not tensors or tensors[0].dim() < 2:
            return tensors

        max_seq_len = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[1] < max_seq_len:
                pad_size = max_seq_len - t.shape[1]
                if t.dim() == 3:  # [batch, seq, hidden]
                    padding = torch.zeros(
                        t.shape[0], pad_size, t.shape[2], dtype=t.dtype
                    )
                else:  # [batch, seq]
                    padding = torch.zeros(t.shape[0], pad_size, dtype=t.dtype)
                t = torch.cat([t, padding], dim=1)
            padded.append(t)
        return padded

    def _combine_tuple_results(self, results: list[tuple[Any, ...]]) -> tuple[Any, ...]:
        """Combine tuple results element-wise."""
        num_elements = len(results[0])
        combined = []

        for i in range(num_elements):
            elements = [r[i] for r in results]
            first_elem = elements[0]

            if isinstance(first_elem, torch.Tensor):
                # Pad tensors to max sequence length before concatenation
                elements = self._pad_tensors_to_max_length(elements)
                # Concatenate tensors along batch dimension
                combined.append(torch.cat(elements, dim=0))

            elif isinstance(first_elem, list):
                # Flatten lists (for hidden states or offsets)
                if first_elem and isinstance(first_elem[0], torch.Tensor):
                    # List of tensors (hidden states per layer)
                    num_layers = len(first_elem)
                    layer_combined = []
                    for layer_idx in range(num_layers):
                        layer_tensors = [e[layer_idx] for e in elements]
                        # Pad tensors to max sequence length before concatenation
                        layer_tensors = self._pad_tensors_to_max_length(layer_tensors)
                        layer_combined.append(torch.cat(layer_tensors, dim=0))
                    combined.append(layer_combined)
                else:
                    # Regular list (offsets)
                    flat = []
                    for e in elements:
                        flat.extend(e)
                    combined.append(flat)

            elif first_elem is None:
                combined.append(None)

            else:
                # Unknown type
                combined.append(elements)

        return tuple(combined)

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
