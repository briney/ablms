"""Tests for the multi-GPU executor's memory and ordering behavior."""

import pytest
import torch

from ablms.parallel.executor import MultiGPUExecutor, _detach_from_shm
from tests.fake_worker_model import FakeWorkerModel


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
        assert torch.equal(combined[:, 0], torch.arange(15, dtype=torch.float32) * 2)
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
