"""Multi-GPU executor for parallel inference."""

from __future__ import annotations

import queue
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

import torch
import torch.multiprocessing as mp

from ablms.parallel.worker import WorkerHandle
from ablms.parallel.utils import DEFAULT_SUBMISSION_WINDOW, WORKER_TIMEOUT

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

    def _stalled_error(self) -> Exception:
        """Build the error raised when no worker reports within the timeout."""
        self._shutdown_workers_fast()
        return TimeoutError(f"No worker returned a result within {WORKER_TIMEOUT}s.")

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
        """Quickly terminate all workers (used on error)."""
        if self._workers is None:
            return

        for worker in self._workers:
            try:
                worker.process.terminate()
            except Exception:
                pass

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
