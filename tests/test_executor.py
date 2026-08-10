"""Tests for the multi-GPU executor's memory and ordering behavior."""

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
