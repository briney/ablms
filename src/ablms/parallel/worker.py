"""Worker process for multi-GPU inference."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any

import torch
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from multiprocessing.context import SpawnContext, SpawnProcess

    from ablms.core.base import BaseAbLM


def worker_main(
    worker_id: int,
    device: torch.device,
    model_class: type[BaseAbLM],
    model_init_kwargs: dict[str, Any],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Main function for worker processes.

    This runs in a separate process and handles:
    1. Model initialization on the assigned GPU
    2. Processing batches from the task queue
    3. Sending results back via the result queue

    Args:
        worker_id: Unique identifier for this worker.
        device: The GPU device this worker should use.
        model_class: The model class to instantiate.
        model_init_kwargs: Keyword arguments for model instantiation.
        task_queue: Queue to receive tasks from.
        result_queue: Queue to send results to.
    """
    model = None

    try:
        # Set CUDA device for this process
        if device.type == "cuda":
            torch.cuda.set_device(device)

        # Initialize model on this worker's device
        # Pass single device to avoid nested parallelization
        model = model_class(devices=device, **model_init_kwargs)

        # Signal ready
        result_queue.put(("ready", worker_id, None))

        # Process tasks until shutdown signal
        while True:
            task = task_queue.get()

            if task is None:  # Shutdown signal
                break

            task_id, method_name, sequences, kwargs = task

            try:
                result = _execute_task(model, method_name, sequences, kwargs)
                result_queue.put(("result", task_id, result))
            except Exception as e:
                error_info = {
                    "exception": e,
                    "traceback": traceback.format_exc(),
                    "worker_id": worker_id,
                    "task_id": task_id,
                }
                result_queue.put(("error", task_id, error_info))

    except Exception as e:
        # Fatal error during initialization or main loop
        error_info = {
            "exception": e,
            "traceback": traceback.format_exc(),
            "worker_id": worker_id,
        }
        result_queue.put(("fatal", worker_id, error_info))

    finally:
        # Cleanup
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _execute_task(
    model: BaseAbLM,
    method_name: str,
    sequences: list[Any],
    kwargs: dict[str, Any],
) -> Any:
    """
    Execute a single batch task on the model.

    Args:
        model: The model instance.
        method_name: Name of the method to call.
        sequences: Batch of sequences to process.
        kwargs: Additional keyword arguments for the method.

    Returns:
        The result from the method call.
    """
    method = getattr(model, method_name)
    return method(sequences, **kwargs)


class WorkerHandle:
    """
    Handle for managing a worker process.

    Provides a cleaner interface for starting, stopping, and
    communicating with worker processes.
    """

    def __init__(
        self,
        worker_id: int,
        device: torch.device,
        model_class: type[BaseAbLM],
        model_init_kwargs: dict[str, Any],
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        process: SpawnProcess,
    ):
        self.worker_id = worker_id
        self.device = device
        self.model_class = model_class
        self.model_init_kwargs = model_init_kwargs
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.process = process
        self._is_alive = True

    @classmethod
    def spawn(
        cls,
        worker_id: int,
        device: torch.device,
        model_class: type[BaseAbLM],
        model_init_kwargs: dict[str, Any],
        result_queue: mp.Queue,
        ctx: SpawnContext,
    ) -> WorkerHandle:
        """
        Spawn a new worker process.

        Args:
            worker_id: Unique identifier for the worker.
            device: GPU device for the worker.
            model_class: Model class to instantiate.
            model_init_kwargs: Kwargs for model initialization.
            result_queue: Shared result queue.
            ctx: Multiprocessing context (should be 'spawn').

        Returns:
            WorkerHandle for the spawned process.
        """
        task_queue = ctx.Queue()

        process = ctx.Process(
            target=worker_main,
            args=(
                worker_id,
                device,
                model_class,
                model_init_kwargs,
                task_queue,
                result_queue,
            ),
            daemon=True,
        )
        process.start()

        return cls(
            worker_id=worker_id,
            device=device,
            model_class=model_class,
            model_init_kwargs=model_init_kwargs,
            task_queue=task_queue,
            result_queue=result_queue,
            process=process,
        )

    def submit_task(
        self,
        task_id: int,
        method_name: str,
        sequences: list[Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Submit a task to this worker."""
        self.task_queue.put((task_id, method_name, sequences, kwargs))

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the worker.

        Args:
            timeout: Seconds to wait for graceful shutdown before terminating.
        """
        if not self._is_alive:
            return

        try:
            # Send shutdown signal
            self.task_queue.put(None)
            self.process.join(timeout=timeout)

            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)

        except Exception:
            pass
        finally:
            self._is_alive = False

    @property
    def is_alive(self) -> bool:
        """Check if the worker process is still alive."""
        return self._is_alive and self.process.is_alive()
