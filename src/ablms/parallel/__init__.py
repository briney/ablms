"""Multi-GPU parallel processing for ablms."""

from ablms.parallel.executor import MultiGPUExecutor
from ablms.parallel.utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SUBMISSION_WINDOW,
    DISABLE_MULTI_GPU,
    get_device_info,
    resolve_devices,
    resolve_single_device,
)

__all__ = [
    "MultiGPUExecutor",
    "resolve_devices",
    "resolve_single_device",
    "get_device_info",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SUBMISSION_WINDOW",
    "DISABLE_MULTI_GPU",
]
