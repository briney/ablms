"""Multi-GPU executor for parallel inference."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.multiprocessing as mp

from ablms.parallel.worker import WorkerHandle
from ablms.parallel.utils import WORKER_TIMEOUT

if TYPE_CHECKING:
    from ablms.core.base import BaseAbLM
    from ablms.core.sequence import AntibodySequence


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
        Execute a model method across multiple GPUs.

        Args:
            method_name: Name of the method to call (e.g., "_process_embeddings_batch").
            sequences: Input sequences to process.
            batch_size: Per-GPU batch size.
            show_progress: Whether to show tqdm progress bar.
            progress_desc: Optional custom description for progress bar.
            **method_kwargs: Additional arguments for the method.

        Returns:
            Combined results from all workers (type depends on method).
        """
        self._ensure_initialized()

        if self._is_single_device:
            return self._execute_single(
                method_name,
                sequences,
                batch_size,
                show_progress,
                progress_desc,
                **method_kwargs,
            )
        else:
            return self._execute_multi(
                method_name,
                sequences,
                batch_size,
                show_progress,
                progress_desc,
                **method_kwargs,
            )

    def _execute_single(
        self,
        method_name: str,
        sequences: list[AntibodySequence],
        batch_size: int,
        show_progress: bool,
        progress_desc: str | None,
        **method_kwargs: Any,
    ) -> Any:
        """Execute on a single GPU (no subprocess overhead)."""
        from tqdm.auto import tqdm

        method = getattr(self._local_model, method_name)
        results = []

        # Create batches
        batches = [
            sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)
        ]

        desc = progress_desc or "Processing"
        pbar = tqdm(
            batches,
            desc=desc,
            disable=not show_progress,
        )

        try:
            for batch in pbar:
                result = method(batch, **method_kwargs)
                results.append(result)
        finally:
            pbar.close()

        return self._combine_results(results)

    def _execute_multi(
        self,
        method_name: str,
        sequences: list[AntibodySequence],
        batch_size: int,
        show_progress: bool,
        progress_desc: str | None,
        **method_kwargs: Any,
    ) -> Any:
        """Execute with multi-GPU parallelism."""
        from tqdm.auto import tqdm
        from ablms.exceptions import WorkerError

        # Distribute work across workers
        work_assignments = self._distribute_work(sequences, batch_size)
        total_batches = sum(len(w) for w in work_assignments)

        if total_batches == 0:
            return self._combine_results([])

        # Submit all tasks to workers
        task_count = 0
        task_to_batch_idx = {}  # Map task_id to original batch index

        for worker_idx, assignments in enumerate(work_assignments):
            for batch_idx, batch in assignments:
                self._workers[worker_idx].submit_task(
                    task_id=task_count,
                    method_name=method_name,
                    sequences=batch,
                    kwargs=method_kwargs,
                )
                task_to_batch_idx[task_count] = batch_idx
                task_count += 1

        # Collect results with progress bar
        results = {}
        desc = progress_desc or f"Processing ({len(self._devices)} GPUs)"
        pbar = tqdm(
            total=total_batches,
            desc=desc,
            disable=not show_progress,
        )

        try:
            while len(results) < task_count:
                msg_type, task_id, data = self._result_queue.get(timeout=WORKER_TIMEOUT)

                if msg_type == "result":
                    batch_idx = task_to_batch_idx[task_id]
                    results[batch_idx] = data
                    pbar.update(1)

                elif msg_type == "error":
                    self._shutdown_workers_fast()
                    raise WorkerError(
                        worker_id=data["worker_id"],
                        original_error=data["exception"],
                    )

                elif msg_type == "fatal":
                    self._shutdown_workers_fast()
                    raise WorkerError(
                        worker_id=data["worker_id"],
                        original_error=data["exception"],
                    )

        finally:
            pbar.close()

        # Order results by batch index
        ordered_results = [results[i] for i in range(len(results))]

        return self._combine_results(ordered_results)

    def _distribute_work(
        self,
        sequences: list[AntibodySequence],
        batch_size: int,
    ) -> list[list[tuple[int, list[AntibodySequence]]]]:
        """
        Distribute sequences across workers using round-robin.

        Args:
            sequences: All input sequences.
            batch_size: Per-GPU batch size.

        Returns:
            List of work assignments per worker. Each assignment is a list of
            (batch_index, batch) tuples for ordering results.
        """
        num_workers = len(self._devices)

        # Create batches with their indices
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            batch_idx = len(batches)
            batches.append((batch_idx, batch))

        # Round-robin distribution
        worker_assignments: list[list[tuple[int, list[AntibodySequence]]]] = [
            [] for _ in range(num_workers)
        ]

        for i, (batch_idx, batch) in enumerate(batches):
            worker_idx = i % num_workers
            worker_assignments[worker_idx].append((batch_idx, batch))

        return worker_assignments

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
