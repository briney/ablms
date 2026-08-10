"""Tests for the multi-GPU executor's memory and ordering behavior."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
import torch

from ablms.parallel.executor import MultiGPUExecutor, _detach_from_shm
from ablms.parallel.worker import WorkerHandle
from tests.fake_worker_model import FakeWorkerModel


@dataclass
class _FakeScanOutput:
    """Stands in for MaskScanOutput: a dataclass holding tensors as attributes."""

    logits: torch.Tensor
    token_ids: torch.Tensor
    vocab: dict[str, int] | None = field(default=None, repr=False)


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

    def test_dataclass_fields_are_detached(self):
        """Covers MaskScanOutput, which holds its tensors as attributes."""
        original = _FakeScanOutput(
            logits=torch.randn(5, 20).share_memory_(),
            token_ids=torch.arange(5).share_memory_(),
            vocab={"A": 0},
        )

        detached = _detach_from_shm(original)

        assert isinstance(detached, _FakeScanOutput)
        assert not detached.logits.is_shared()
        assert not detached.token_ids.is_shared()
        assert torch.equal(detached.logits, original.logits)
        assert detached.vocab == {"A": 0}

    def test_dataclass_inside_a_list_is_detached(self):
        """mask_scan returns list[MaskScanOutput], so the list path must recurse."""
        results = [
            _FakeScanOutput(
                logits=torch.randn(3, 4).share_memory_(),
                token_ids=torch.arange(3).share_memory_(),
            )
            for _ in range(2)
        ]

        detached = _detach_from_shm(results)

        assert len(detached) == 2
        assert all(not r.logits.is_shared() for r in detached)
        assert all(not r.token_ids.is_shared() for r in detached)

    def test_dataclass_type_passes_through(self):
        """A class object is a dataclass too, but must not be instantiated."""
        assert _detach_from_shm(_FakeScanOutput) is _FakeScanOutput


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
        """execute() must be exactly the concatenation of execute_iter()."""
        executor = _cpu_executor(2)
        call = dict(
            method_name="_process_echo_batch",
            sequences=list(range(15)),
            batch_size=4,
            show_progress=False,
            scale=2,
        )
        try:
            combined, _, offsets = executor.execute(**call)
            streamed = list(executor.execute_iter(**call))
        finally:
            executor.shutdown()

        assert combined.shape == (15, 2)
        assert torch.equal(combined[:, 0], torch.arange(15, dtype=torch.float32) * 2)
        assert len(offsets) == 15

        # Compare only the value column: the pid column depends on which worker
        # happened to pick up each batch, which is not part of the contract.
        streamed_values = torch.cat([result[0] for _, result in streamed])
        assert torch.equal(streamed_values[:, 0], combined[:, 0])

        streamed_offsets = [o for _, result in streamed for o in result[2]]
        assert streamed_offsets == offsets

        assert all(result[1] is None for _, result in streamed)

    @pytest.mark.slow
    def test_outstanding_tasks_stay_within_the_submission_window(self, monkeypatch):
        """At most num_workers * window tasks may be outstanding at once.

        This is the property the submission window exists to provide: it bounds
        how many results can be live in shared memory simultaneously,
        independent of dataset size. Revert _iter_multi to submitting every
        batch up front and the peak becomes the batch count.

        "Outstanding" is measured as submitted-minus-returned, counting a task
        as returned the moment the parent pulls it off the result queue - that
        is the point at which the worker's shared-memory segment is released.
        Submitted-minus-*yielded* would additionally include the reorder buffer,
        which is heap-resident and deliberately unbounded.
        """
        executor = _cpu_executor(2)
        num_workers = executor.num_devices
        window = executor._submission_window

        counts = {"submitted": 0, "returned": 0, "peak": 0}

        original_submit = WorkerHandle.submit_task

        def counting_submit(handle, *args, **kwargs):
            counts["submitted"] += 1
            counts["peak"] = max(
                counts["peak"], counts["submitted"] - counts["returned"]
            )
            return original_submit(handle, *args, **kwargs)

        monkeypatch.setattr(WorkerHandle, "submit_task", counting_submit)

        yielded = 0
        try:
            # Initialize first, so the workers' "ready" messages are not
            # counted as returned results.
            executor._ensure_initialized()
            original_get = executor._result_queue.get

            def counting_get(*args, **kwargs):
                message = original_get(*args, **kwargs)
                counts["returned"] += 1
                return message

            executor._result_queue.get = counting_get

            for _ in executor.execute_iter(
                method_name="_process_echo_batch",
                sequences=list(range(60)),
                batch_size=2,
                show_progress=False,
            ):
                yielded += 1
        finally:
            executor.shutdown()

        assert yielded == 30
        assert counts["submitted"] == 30
        # The window must be filled (otherwise the devices are idle)...
        assert counts["peak"] >= num_workers * window
        # ...and never exceeded.
        assert counts["peak"] <= num_workers * window

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

    @pytest.mark.slow
    def test_executor_is_reusable_after_abandoning_the_generator(self):
        """An abandoned generator must not poison the next call on the executor.

        Closing the generator leaves the tasks inside the window still running.
        If their results are left unread on the shared queue, the next call
        picks them up as its own and fails on the unknown task id.
        """
        executor = _cpu_executor(2)
        try:
            gen = executor.execute_iter(
                method_name="_process_echo_batch",
                sequences=list(range(40)),
                batch_size=2,
                show_progress=False,
            )
            first_idx, _ = next(gen)
            assert first_idx == 0
            gen.close()

            combined, _, offsets = executor.execute(
                method_name="_process_echo_batch",
                sequences=list(range(6)),
                batch_size=2,
                show_progress=False,
            )
        finally:
            executor.shutdown()

        assert torch.equal(combined[:, 0], torch.arange(6, dtype=torch.float32))
        assert len(offsets) == 6


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

    @pytest.mark.slow
    def test_dead_worker_message_differs_from_shared_memory_message(self):
        """A worker that died outright is a different failure than a live one
        that simply couldn't hand back a result - the message must say so."""
        from ablms import SharedMemoryError
        from ablms.exceptions import MultiGPUError

        executor = _cpu_executor(1)
        executor._is_single_device = False
        try:
            executor._ensure_initialized()
            executor._workers[0].process.terminate()
            executor._workers[0].process.join(timeout=5)

            error = executor._stalled_error()
        finally:
            executor.shutdown()

        assert not isinstance(error, SharedMemoryError)
        assert isinstance(error, MultiGPUError)
        message = str(error)
        assert "no longer running" in message
        assert "shm-size" not in message

    @pytest.mark.slow
    def test_second_call_after_stall_does_not_hang(self):
        """A stall must leave the executor able to recover, not stuck.

        _shutdown_workers_fast used to terminate the worker processes without
        clearing self._workers, so is_initialized kept reporting True for
        processes that no longer existed. The next call would then submit
        tasks nothing could ever consume and block for another full
        WORKER_TIMEOUT. The thread + join(timeout=...) below turns a
        regression into a fast test failure instead of a 300s hang.
        """
        executor = _cpu_executor(1)
        executor._is_single_device = False
        try:
            executor._ensure_initialized()
            executor._stalled_error()
            assert not executor.is_initialized

            outcome: dict[str, object] = {}

            def _second_call() -> None:
                try:
                    outcome["result"] = executor.execute(
                        method_name="_process_echo_batch",
                        sequences=list(range(4)),
                        batch_size=2,
                        show_progress=False,
                    )
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    outcome["error"] = exc

            thread = threading.Thread(target=_second_call, daemon=True)
            thread.start()
            thread.join(timeout=30)

            assert not thread.is_alive(), "second call after a stall hung"
            assert "error" not in outcome, f"second call failed: {outcome.get('error')}"

            combined, _, _ = outcome["result"]
            assert combined.shape[0] == 4
        finally:
            executor.shutdown()
