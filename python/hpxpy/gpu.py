# HPXPy GPU Module
#
# SPDX-License-Identifier: BSL-1.0

"""
GPU/CUDA support for HPXPy.

This module provides GPU acceleration for HPXPy arrays and operations.

Basic usage::

    import hpxpy as hpx

    # Check GPU availability
    if hpx.gpu.is_available():
        print(f"Found {hpx.gpu.device_count()} GPU(s)")

        # List all devices
        for dev in hpx.gpu.get_devices():
            print(f"  {dev.name}: {dev.total_memory_gb():.1f} GB")

        # Create GPU array
        arr = hpx.gpu.zeros([1000, 1000])

        # Transfer from numpy
        import numpy as np
        np_arr = np.random.randn(1000)
        gpu_arr = hpx.gpu.from_numpy(np_arr)

        # Transfer back to numpy
        result = gpu_arr.to_numpy()
"""

from __future__ import annotations

try:
    from hpxpy._core import gpu as _gpu
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False


def is_available() -> bool:
    """Check if CUDA/GPU support is available.

    Returns
    -------
    bool
        True if CUDA is available and at least one GPU is found.
    """
    if not _GPU_AVAILABLE:
        return False
    return _gpu.is_available()


def device_count() -> int:
    """Get the number of available GPU devices.

    Returns
    -------
    int
        Number of CUDA devices, or 0 if CUDA is not available.
    """
    if not _GPU_AVAILABLE:
        return 0
    return _gpu.device_count()


def get_devices():
    """Get information about all available GPU devices.

    Returns
    -------
    list
        List of Device objects with GPU information.
    """
    if not _GPU_AVAILABLE:
        return []
    return _gpu.get_devices()


def get_device(device_id: int = 0):
    """Get information about a specific GPU device.

    Parameters
    ----------
    device_id : int
        Device ID to query.

    Returns
    -------
    Device
        Device information object.

    Raises
    ------
    RuntimeError
        If the device ID is invalid or CUDA is not available.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    return _gpu.get_device(device_id)


def current_device() -> int:
    """Get the current CUDA device ID.

    Returns
    -------
    int
        Current device ID.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    return _gpu.current_device()


def set_device(device_id: int) -> None:
    """Set the current CUDA device.

    Parameters
    ----------
    device_id : int
        Device ID to set as current.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    _gpu.set_device(device_id)


def synchronize(device_id: int = -1) -> None:
    """Synchronize the GPU device.

    Parameters
    ----------
    device_id : int, optional
        Device to synchronize. Default (-1) synchronizes current device.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    _gpu.synchronize(device_id)


def memory_info(device_id: int = 0) -> tuple:
    """Get GPU memory information.

    Parameters
    ----------
    device_id : int
        Device to query.

    Returns
    -------
    tuple
        (free_bytes, total_bytes)
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    return _gpu.memory_info(device_id)


def zeros(shape, device: int = 0):
    """Create a GPU array filled with zeros.

    Parameters
    ----------
    shape : tuple or list
        Shape of the array.
    device : int
        GPU device ID.

    Returns
    -------
    ArrayF64
        GPU array filled with zeros.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    if isinstance(shape, int):
        shape = [shape]
    return _gpu.zeros(list(shape), device)


def ones(shape, device: int = 0):
    """Create a GPU array filled with ones.

    Parameters
    ----------
    shape : tuple or list
        Shape of the array.
    device : int
        GPU device ID.

    Returns
    -------
    ArrayF64
        GPU array filled with ones.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    if isinstance(shape, int):
        shape = [shape]
    return _gpu.ones(list(shape), device)


def full(shape, value: float, device: int = 0):
    """Create a GPU array filled with a value.

    Parameters
    ----------
    shape : tuple or list
        Shape of the array.
    value : float
        Fill value.
    device : int
        GPU device ID.

    Returns
    -------
    ArrayF64
        GPU array filled with the value.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    if isinstance(shape, int):
        shape = [shape]
    return _gpu.full(list(shape), value, device)


def from_numpy(arr, device: int = 0):
    """Create a GPU array from a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Source numpy array.
    device : int
        GPU device ID.

    Returns
    -------
    ArrayF64
        GPU array with copied data.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    import numpy as np
    arr = np.asarray(arr, dtype=np.float64)
    return _gpu.from_numpy(arr, device)


def sum(arr):
    """Sum all elements of a GPU array.

    Parameters
    ----------
    arr : ArrayF64
        GPU array to sum.

    Returns
    -------
    float
        Sum of all elements.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    return _gpu.sum(arr)


# Export GPU array types if available
if _GPU_AVAILABLE and hasattr(_gpu, 'ArrayF64'):
    ArrayF64 = _gpu.ArrayF64
    ArrayF32 = _gpu.ArrayF32
    ArrayI64 = _gpu.ArrayI64
    ArrayI32 = _gpu.ArrayI32
    Device = _gpu.Device

__all__ = [
    "is_available",
    "device_count",
    "get_devices",
    "get_device",
    "current_device",
    "set_device",
    "synchronize",
    "memory_info",
    "zeros",
    "ones",
    "full",
    "from_numpy",
    "sum",
]
