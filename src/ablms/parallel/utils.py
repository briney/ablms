"""Utilities for multi-GPU parallel processing."""

from __future__ import annotations

import os

import torch

from ablms.exceptions import DeviceError

# Environment variable defaults
DEFAULT_BATCH_SIZE = int(os.environ.get("ABLMS_DEFAULT_BATCH_SIZE", 32))
DISABLE_MULTI_GPU = os.environ.get("ABLMS_DISABLE_MULTI_GPU", "").lower() == "true"
WORKER_TIMEOUT = int(os.environ.get("ABLMS_WORKER_TIMEOUT", 300))
DEFAULT_SUBMISSION_WINDOW = int(os.environ.get("ABLMS_SUBMISSION_WINDOW", 2))


def resolve_devices(
    devices: str | int | list[int | str] | torch.device | list[torch.device] | None,
) -> list[torch.device]:
    """
    Resolve device specification to a list of torch.device objects.

    Args:
        devices: Device specification. Options:
            - None: Auto-detect all available GPUs (or MPS/CPU)
            - Single device: "cuda:0", 0, torch.device("cuda:0"), "cpu", "mps"
            - Multiple devices: [0, 1, 2], ["cuda:0", "cuda:1"]

    Returns:
        List of torch.device objects (length >= 1).

    Raises:
        DeviceError: If the specified device is not available.

    Examples:
        >>> resolve_devices(None)  # Auto-detect
        [device(type='cuda', index=0), device(type='cuda', index=1)]

        >>> resolve_devices("cuda:0")
        [device(type='cuda', index=0)]

        >>> resolve_devices([0, 2])
        [device(type='cuda', index=0), device(type='cuda', index=2)]
    """
    # Check if multi-GPU is disabled via environment variable
    if DISABLE_MULTI_GPU and devices is None:
        # Fall back to single device auto-detection
        return [_auto_detect_single_device()]

    # None -> auto-detect all available devices
    if devices is None:
        return _auto_detect_devices()

    # Single device
    if isinstance(devices, (str, int, torch.device)):
        return [resolve_single_device(devices)]

    # List of devices
    if isinstance(devices, list):
        if not devices:
            raise DeviceError("Empty device list provided")
        return [resolve_single_device(d) for d in devices]

    raise DeviceError(f"Invalid devices specification: {devices}")


def resolve_single_device(device: str | int | torch.device) -> torch.device:
    """
    Resolve a single device specification to a torch.device.

    Args:
        device: Single device specification (int, str, or torch.device).

    Returns:
        Resolved torch.device.

    Raises:
        DeviceError: If the specified device is not available.
    """
    # Integer -> CUDA device index
    if isinstance(device, int):
        if not torch.cuda.is_available():
            raise DeviceError(
                f"CUDA device {device} requested but CUDA is not available"
            )
        if device >= torch.cuda.device_count():
            raise DeviceError(
                f"CUDA device {device} not available "
                f"(have {torch.cuda.device_count()} device(s))"
            )
        return torch.device(f"cuda:{device}")

    # String -> parse as device
    if isinstance(device, str):
        device = torch.device(device)

    # Validate availability
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError("CUDA is not available on this system")
        # Validate specific CUDA device if index is specified
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise DeviceError(
                f"CUDA device {device.index} not available "
                f"(have {torch.cuda.device_count()} device(s))"
            )
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceError("MPS is not available on this system")

    return device


def _auto_detect_devices() -> list[torch.device]:
    """
    Auto-detect all available GPU devices.

    Returns:
        List of available torch.device objects.
        Prefers CUDA GPUs, then MPS, then falls back to CPU.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    elif torch.backends.mps.is_available():
        return [torch.device("mps")]
    else:
        return [torch.device("cpu")]


def _auto_detect_single_device() -> torch.device:
    """
    Auto-detect a single device (used when multi-GPU is disabled).

    Returns:
        Single torch.device (first available GPU, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info(devices: list[torch.device]) -> str:
    """
    Get a human-readable string describing the devices.

    Args:
        devices: List of torch.device objects.

    Returns:
        Description string like "4 CUDA GPUs" or "CPU".
    """
    if not devices:
        return "no devices"

    device_type = devices[0].type
    count = len(devices)

    if device_type == "cuda":
        if count == 1:
            return f"CUDA GPU {devices[0].index}"
        else:
            indices = [d.index for d in devices]
            return f"{count} CUDA GPUs ({', '.join(map(str, indices))})"
    elif device_type == "mps":
        return "Apple MPS"
    else:
        return "CPU"
