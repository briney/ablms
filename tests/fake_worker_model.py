"""A minimal stand-in for a model, used to exercise executor plumbing.

This is deliberately not a BaseAbLM. It exists to test submission windowing
and result ordering without paying for real model weights in a subprocess.
Model behavior itself is always tested against real models.
"""

from __future__ import annotations

import os

import torch


class FakeWorkerModel:
    """Echoes its input batch back as a tensor, tagged with the worker's PID."""

    def __init__(self, devices=None, **kwargs):
        self.devices = devices
        self.pid = os.getpid()

    def _process_echo_batch(self, sequences, scale: int = 1):
        """Return a [len(sequences), 2] tensor of (value * scale, pid)."""
        values = torch.tensor(
            [[float(s) * scale, float(self.pid)] for s in sequences],
            dtype=torch.float32,
        )
        return values, None, [{"item": (0, 1)} for _ in sequences]
