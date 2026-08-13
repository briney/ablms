"""Tests for the multi-GPU executor's memory and ordering behavior."""

from __future__ import annotations

import threading
import weakref
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


class TestConcatBatchesPadding:
    """Combining must not assume which axis carries sequence length.

    It is dim 1 for [batch, seq, hidden] but dim 2 for the multi-layer
    [batch, layers, seq, hidden]. The rule that covers both: every non-batch
    axis is sized to the largest seen and each batch is written at the origin,
    so the leftover region is zero fill. Dim 0 is the concatenation axis.
    """

    @pytest.fixture
    def executor(self):
        return MultiGPUExecutor(FakeWorkerModel, {}, [torch.device("cpu")])

    def test_pads_three_dimensional_on_sequence_axis(self, executor):
        combined = executor._concat_batches([torch.ones(2, 5, 4), torch.ones(3, 7, 4)])

        assert combined.shape == (5, 7, 4)
        assert combined[:2, :5, :].eq(1).all()
        assert combined[:2, 5:, :].eq(0).all()

    def test_pads_two_dimensional_mask(self, executor):
        combined = executor._concat_batches([torch.ones(2, 5), torch.ones(3, 7)])

        assert combined.shape == (5, 7)
        assert combined[:2, 5:].eq(0).all()

    def test_pads_four_dimensional_on_sequence_axis(self, executor):
        """[batch, layers, seq, hidden] pads dim 2, leaving the layer axis alone."""
        combined = executor._concat_batches(
            [torch.ones(2, 3, 5, 4), torch.ones(1, 3, 7, 4)]
        )

        assert combined.shape == (3, 3, 7, 4)
        assert combined[:2, :, 5:, :].eq(0).all()
        assert combined[:2, :, :5, :].eq(1).all()

    def test_leaves_batch_axis_alone(self, executor):
        """Dim 0 is the concatenation axis and must never be padded."""
        combined = executor._concat_batches([torch.ones(2, 5, 4), torch.ones(3, 5, 4)])

        assert combined.shape == (5, 5, 4)
        assert combined.eq(1).all()

    def test_one_dimensional_batches_are_concatenated(self, executor):
        combined = executor._concat_batches([torch.ones(3), torch.ones(5)])

        assert combined.shape == (8,)
        assert combined.eq(1).all()

    def test_differing_rank_is_rejected(self, executor):
        with pytest.raises(ValueError, match="rank"):
            executor._concat_batches([torch.ones(2, 5), torch.ones(2, 5, 4)])

    def test_zero_dimensional_is_rejected(self, executor):
        with pytest.raises(ValueError, match="zero-dimensional"):
            executor._concat_batches([torch.tensor(1.0)])


class TestConcatBatchesReleasesInputs:
    """_concat_batches must not hold its inputs and its output at once.

    `torch.cat` builds the output while every input is still referenced, so the
    peak is the sum of both. Writing into a preallocated output and dropping
    each batch as it is copied keeps roughly one result's worth resident.
    """

    @pytest.fixture
    def executor(self):
        return MultiGPUExecutor(FakeWorkerModel, {}, [torch.device("cpu")])

    def test_batch_tensors_are_freed_as_they_are_combined(self, executor):
        """No input batch may survive the call that consumes it."""
        tensors = [torch.ones(2, 4) for _ in range(3)]
        refs = [weakref.ref(t) for t in tensors]

        combined = executor._concat_batches(tensors)
        del tensors

        assert combined.shape == (6, 4)
        assert combined.eq(1).all()
        assert all(ref() is None for ref in refs), (
            "batch tensors still alive after combining, so peak memory is 2x "
            "the output"
        )

    def test_dtype_is_preserved(self, executor):
        combined = executor._concat_batches([torch.ones(2, 4, dtype=torch.float16)])
        assert combined.dtype == torch.float16


class _TrackingModel:
    """Records how many of the tensors it has returned are still alive."""

    instances: list[_TrackingModel] = []

    def __init__(self, devices=None, **kwargs):
        self.refs: list[weakref.ref] = []
        self.max_alive = 0
        _TrackingModel.instances.append(self)

    def _process_tracked_batch(self, sequences):
        alive = sum(1 for ref in self.refs if ref() is not None)
        self.max_alive = max(self.max_alive, alive)
        tensor = torch.ones(len(sequences), 4)
        self.refs.append(weakref.ref(tensor))
        return tensor, None, [{"item": (0, 1)} for _ in sequences]


class _RaggedModel:
    """Returns batches whose sequence axis depends on the batch contents."""

    def __init__(self, devices=None, **kwargs):
        pass

    def _process_ragged_batch(self, sequences):
        seq_len = 3 + int(sequences[0])
        tensor = torch.ones(len(sequences), seq_len, 2)
        mask = torch.ones(len(sequences), seq_len)
        return tensor, mask, [{"item": (0, 1)} for _ in sequences]


class TestExecuteStreamsIntoOutput:
    """execute() must fill its output as batches arrive, not accumulate first.

    Freeing a batch after the output is allocated does not help: a pooled batch
    is small enough to stay in the heap arena, where free() returns nothing to
    the OS, and the output was already allocated as one large mapping that
    cannot reuse it. The output therefore has to exist before the batches do,
    so each batch's memory is recycled by the next.
    """

    def test_batches_are_not_all_held_at_once(self):
        """The whole point: batch k must be released before batch k+2 exists."""
        _TrackingModel.instances.clear()
        executor = MultiGPUExecutor(_TrackingModel, {}, [torch.device("cpu")])
        try:
            combined, mask, offsets = executor.execute(
                "_process_tracked_batch",
                sequences=list(range(40)),
                batch_size=4,
                show_progress=False,
            )
        finally:
            executor.shutdown()

        model = _TrackingModel.instances[0]
        assert combined.shape == (40, 4)
        assert combined.eq(1).all()
        assert mask is None
        assert len(offsets) == 40
        assert model.max_alive <= 2, (
            f"{model.max_alive} of 10 batch tensors were alive at once; "
            f"execute() is accumulating every batch before combining"
        )

    def test_ragged_batches_growing_are_padded(self):
        """A later batch longer than the first must still combine correctly."""
        executor = MultiGPUExecutor(_RaggedModel, {}, [torch.device("cpu")])
        try:
            combined, mask, offsets = executor.execute(
                "_process_ragged_batch",
                sequences=[0, 0, 0, 0, 4, 4, 4, 4],
                batch_size=4,
                show_progress=False,
            )
        finally:
            executor.shutdown()

        assert combined.shape == (8, 7, 2)
        assert combined[:4, :3, :].eq(1).all()
        assert combined[:4, 3:, :].eq(0).all(), "short batch must be zero-padded"
        assert combined[4:].eq(1).all()
        assert mask.shape == (8, 7)
        assert len(offsets) == 8

    def test_ragged_batches_shrinking_are_padded(self):
        """A later batch shorter than the first pads into the existing width."""
        executor = MultiGPUExecutor(_RaggedModel, {}, [torch.device("cpu")])
        try:
            combined, _, _ = executor.execute(
                "_process_ragged_batch",
                sequences=[4, 4, 4, 4, 0, 0, 0, 0],
                batch_size=4,
                show_progress=False,
            )
        finally:
            executor.shutdown()

        assert combined.shape == (8, 7, 2)
        assert combined[:4].eq(1).all()
        assert combined[4:, :3, :].eq(1).all()
        assert combined[4:, 3:, :].eq(0).all(), "short batch must be zero-padded"

    def test_non_tensor_results_still_combine(self):
        """List-returning methods (pseudo_ll, fill_mask) are unaffected."""
        executor = MultiGPUExecutor(FakeWorkerModel, {}, [torch.device("cpu")])
        try:
            values, _, offsets = executor.execute(
                "_process_echo_batch",
                sequences=[1, 2, 3, 4, 5],
                batch_size=2,
                show_progress=False,
                scale=10,
            )
        finally:
            executor.shutdown()

        assert values.shape == (5, 2)
        assert values[:, 0].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]
        assert len(offsets) == 5
