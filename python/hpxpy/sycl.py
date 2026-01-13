# HPXPy SYCL Module
#
# SPDX-License-Identifier: BSL-1.0

"""
SYCL GPU support for HPXPy via HPX's sycl_executor.

This module provides GPU acceleration through SYCL, which supports multiple
backends including:
- Intel GPUs (via oneAPI/Level-Zero)
- NVIDIA GPUs (via CUDA backend)
- AMD GPUs (via HIP backend)
- Apple Silicon (via Metal backend with AdaptiveCpp - experimental)

Basic usage::

    import hpxpy as hpx

    # Check SYCL availability
    if hpx.sycl.is_available():
        print(f"Found {hpx.sycl.device_count()} SYCL device(s)")

        # List all devices
        for dev in hpx.sycl.get_devices():
            print(f"  {dev.name} ({dev.backend}): {dev.global_mem_size_gb():.1f} GB")

        # Create SYCL array
        arr = hpx.sycl.zeros([1000, 1000])

        # Transfer from numpy
        import numpy as np
        np_arr = np.random.randn(1000)
        sycl_arr = hpx.sycl.from_numpy(np_arr)

        # Transfer back to numpy
        result = sycl_arr.to_numpy()
"""

from __future__ import annotations

try:
    from hpxpy._core import sycl as _sycl
    _SYCL_AVAILABLE = True
except ImportError:
    _SYCL_AVAILABLE = False


def is_available() -> bool:
    """Check if SYCL GPU support is available.

    Returns
    -------
    bool
        True if SYCL is available and at least one GPU is found.
    """
    if not _SYCL_AVAILABLE:
        return False
    return _sycl.is_available()


def device_count() -> int:
    """Get the number of available SYCL GPU devices.

    Returns
    -------
    int
        Number of SYCL GPU devices, or 0 if SYCL is not available.
    """
    if not _SYCL_AVAILABLE:
        return 0
    return _sycl.device_count()


def get_devices():
    """Get information about all available SYCL GPU devices.

    Returns
    -------
    list
        List of Device objects with GPU information.
    """
    if not _SYCL_AVAILABLE:
        return []
    return _sycl.get_devices()


def get_device(device_id: int = 0):
    """Get information about a specific SYCL device.

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
        If the device ID is invalid or SYCL is not available.
    """
    if not _SYCL_AVAILABLE:
        raise RuntimeError("SYCL is not available")
    return _sycl.get_device(device_id)


def zeros(shape, device: int = 0):
    """Create a SYCL array filled with zeros.

    Parameters
    ----------
    shape : tuple or list
        Shape of the array.
    device : int
        SYCL device ID.

    Returns
    -------
    ArrayF64
        SYCL array filled with zeros.
    """
    if not _SYCL_AVAILABLE:
        raise RuntimeError("SYCL is not available")
    if isinstance(shape, int):
        shape = [shape]
    return _sycl.zeros(list(shape), device)


def ones(shape, device: int = 0):
    """Create a SYCL array filled with ones.

    Parameters
    ----------
    shape : tuple or list
        Shape of the array.
    device : int
        SYCL device ID.

    Returns
    -------
    ArrayF64
        SYCL array filled with ones.
    """
    if not _SYCL_AVAILABLE:
        raise RuntimeError("SYCL is not available")
    if isinstance(shape, int):
        shape = [shape]
    return _sycl.ones(list(shape), device)


def full(shape, value: float, device: int = 0):
    """Create a SYCL array filled with a value.

    Parameters
    ----------
    shape : tuple or list
        Shape of the array.
    value : float
        Fill value.
    device : int
        SYCL device ID.

    Returns
    -------
    ArrayF64
        SYCL array filled with the value.
    """
    if not _SYCL_AVAILABLE:
        raise RuntimeError("SYCL is not available")
    if isinstance(shape, int):
        shape = [shape]
    return _sycl.full(list(shape), value, device)


def from_numpy(arr, device: int = 0):
    """Create a SYCL array from a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Source numpy array.
    device : int
        SYCL device ID.

    Returns
    -------
    ArrayF64
        SYCL array with copied data.
    """
    if not _SYCL_AVAILABLE:
        raise RuntimeError("SYCL is not available")
    import numpy as np
    arr = np.asarray(arr, dtype=np.float64)
    return _sycl.from_numpy(arr, device)


def sum(arr):
    """Sum all elements of a SYCL array.

    Parameters
    ----------
    arr : ArrayF64
        SYCL array to sum.

    Returns
    -------
    float
        Sum of all elements.
    """
    if not _SYCL_AVAILABLE:
        raise RuntimeError("SYCL is not available")
    return _sycl.sum(arr)


# --------------------------------------------------------------------------
# Async SYCL Operations (HPX Integration)
# --------------------------------------------------------------------------

def enable_async(pool_name: str = "") -> None:
    """Enable async SYCL operations.

    This enables HPX's SYCL polling mechanism which is required for
    async SYCL operations to complete.

    Parameters
    ----------
    pool_name : str, optional
        HPX thread pool to use for polling. Default uses the first pool.

    Example
    -------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> hpx.sycl.enable_async()
    >>> arr = hpx.sycl.zeros([1000000])
    >>> future = arr.async_from_numpy(data)
    >>> future.get()
    >>> hpx.sycl.disable_async()
    """
    if not _SYCL_AVAILABLE:
        return  # No-op without SYCL
    _sycl.enable_async(pool_name)


def disable_async() -> None:
    """Disable async SYCL operations.

    Stops HPX's SYCL polling mechanism.
    """
    if not _SYCL_AVAILABLE:
        return  # No-op without SYCL
    _sycl.disable_async()


def is_async_enabled() -> bool:
    """Check if async SYCL operations are enabled.

    Returns
    -------
    bool
        True if async SYCL operations are enabled.
    """
    if not _SYCL_AVAILABLE:
        return False
    return _sycl.is_async_enabled()


class AsyncContext:
    """Context manager for async SYCL operations.

    Enables async SYCL operations within a context and automatically
    disables them when exiting.

    Example
    -------
    >>> with hpx.sycl.AsyncContext():
    ...     future1 = arr1.async_from_numpy(data1)
    ...     future2 = arr2.async_from_numpy(data2)
    ...     future1.get()
    ...     future2.get()
    >>> # Async automatically disabled here
    """

    def __init__(self, pool_name: str = ""):
        self.pool_name = pool_name

    def __enter__(self):
        enable_async(self.pool_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        disable_async()
        return False


# Export array types if available
if _SYCL_AVAILABLE and hasattr(_sycl, 'ArrayF64'):
    ArrayF64 = _sycl.ArrayF64
    ArrayF32 = _sycl.ArrayF32
    ArrayI64 = _sycl.ArrayI64
    ArrayI32 = _sycl.ArrayI32
    Device = _sycl.Device
    Future = _sycl.Future

__all__ = [
    "is_available",
    "device_count",
    "get_devices",
    "get_device",
    "zeros",
    "ones",
    "full",
    "from_numpy",
    "sum",
    # Async operations
    "enable_async",
    "disable_async",
    "is_async_enabled",
    "AsyncContext",
]
