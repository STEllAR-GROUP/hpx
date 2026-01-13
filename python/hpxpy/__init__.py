# HPXPy - High Performance Python Arrays powered by HPX
#
# SPDX-License-Identifier: BSL-1.0

"""
HPXPy: High-performance distributed NumPy-like arrays powered by HPX.

HPXPy provides a NumPy-compatible interface for parallel and distributed
array computing using the HPX C++ runtime system.

Basic usage::

    import hpxpy as hpx

    # Initialize the HPX runtime
    hpx.init()

    # Create arrays
    a = hpx.zeros((1000,))
    b = hpx.ones((1000,))

    # Perform parallel operations
    result = hpx.sum(a + b)

    # Clean up
    hpx.finalize()

Or using the context manager::

    import hpxpy as hpx

    with hpx.runtime():
        a = hpx.arange(1000000)
        print(hpx.sum(a))
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    # Version info
    "__version__",
    # Runtime management
    "init",
    "finalize",
    "is_running",
    "runtime",
    "num_threads",
    "num_localities",
    "locality_id",
    # Array creation
    "array",
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "linspace",
    "from_numpy",
    # Array class
    "ndarray",
    # Reduction algorithms
    "sum",
    "prod",
    "min",
    "max",
    "mean",
    "std",
    "var",
    # Sorting
    "sort",
    "argsort",
    "count",
    # Math functions
    "sqrt",
    "square",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "trunc",
    "abs",
    "sign",
    # Scan operations
    "cumsum",
    "cumprod",
    # Element-wise functions
    "maximum",
    "minimum",
    "clip",
    "power",
    "where",
    # Random submodule
    "random",
    # Execution policies
    "execution",
    # Distribution (Phase 3)
    "distribution",
    # Collective operations (Phase 4)
    "collectives",
    "all_reduce",
    "broadcast",
    "gather",
    "scatter",
    "barrier",
    # Distributed arrays (Phase 4)
    "DistributionPolicy",
    "distributed_zeros",
    "distributed_ones",
    "distributed_full",
    "distributed_from_numpy",
    # Multi-locality launcher (Phase 4)
    "launcher",
    # GPU support (Phase 5)
    "gpu",
    # SYCL support (Phase 5) - HPX-native backend for Intel/AMD/Apple GPUs
    "sycl",
]

# Import the compiled extension module
try:
    from hpxpy._core import (
        # Runtime
        init as _init,
        finalize as _finalize,
        is_running,
        num_threads,
        num_localities,
        locality_id,
        # Array class
        ndarray,
        # Array creation
        _zeros,
        _ones,
        _empty,
        _arange,
        _array_from_numpy,
        # Algorithms
        _sum,
        _prod,
        _min,
        _max,
        _sort,
        _count,
        # Math functions
        _sqrt,
        _square,
        _exp,
        _exp2,
        _log,
        _log2,
        _log10,
        _sin,
        _cos,
        _tan,
        _arcsin,
        _arccos,
        _arctan,
        _sinh,
        _cosh,
        _tanh,
        _floor,
        _ceil,
        _trunc,
        _abs,
        _sign,
        # Scan operations
        _cumsum,
        _cumprod,
        # Element-wise functions
        _maximum,
        _minimum,
        _clip,
        _power,
        _where,
        # Random submodule
        random as _random_module,
        # Execution module
        execution,
        # Distribution module (Phase 3)
        distribution,
        # Collectives module (Phase 4)
        collectives,
        all_reduce as _all_reduce,
        broadcast as _broadcast,
        gather as _gather,
        scatter as _scatter,
        barrier as _barrier,
        # Distributed arrays (Phase 4)
        distributed_zeros,
        distributed_ones,
        distributed_full,
        distributed_from_numpy,
    )
    # DistributionPolicy is in the distribution submodule
    from hpxpy._core.distribution import DistributionPolicy

    _HPX_AVAILABLE = True
except ImportError as e:
    _HPX_AVAILABLE = False
    _IMPORT_ERROR = str(e)


def _check_available() -> None:
    """Check if HPX extension is available."""
    if not _HPX_AVAILABLE:
        raise ImportError(
            f"HPXPy extension module not available: {_IMPORT_ERROR}\n"
            "Make sure HPX is installed and the extension was built correctly."
        )


# -----------------------------------------------------------------------------
# Runtime Management
# -----------------------------------------------------------------------------


class runtime:
    """Context manager for HPX runtime initialization.

    Example::

        with hpx.runtime(num_threads=4):
            # HPX operations here
            result = hpx.sum(hpx.arange(1000))
        # Runtime automatically finalized
    """

    def __init__(self, num_threads: int | None = None, config: list[str] | None = None):
        """Initialize the runtime context.

        Parameters
        ----------
        num_threads : int, optional
            Number of OS threads to use. Defaults to hardware concurrency.
        config : list of str, optional
            Additional HPX configuration options.
        """
        self.num_threads = num_threads
        self.config = config

    def __enter__(self) -> "runtime":
        init(num_threads=self.num_threads, config=self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        finalize()
        return False


def init(num_threads: int | None = None, config: list[str] | None = None) -> None:
    """Initialize the HPX runtime system.

    This must be called before any HPX operations. Alternatively, use the
    ``runtime`` context manager.

    Parameters
    ----------
    num_threads : int, optional
        Number of OS threads to use. If not specified, uses the number
        of hardware threads available.
    config : list of str, optional
        Additional HPX configuration options, e.g.,
        ``["--hpx:threads=4", "--hpx:localities=2"]``

    Raises
    ------
    RuntimeError
        If the runtime is already initialized.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init(num_threads=4)
    >>> # ... do work ...
    >>> hpx.finalize()
    """
    _check_available()
    _init(num_threads=num_threads, config=config or [])


def finalize() -> None:
    """Finalize the HPX runtime system.

    This should be called when HPX operations are complete. After calling
    this, HPX operations cannot be performed until ``init`` is called again.

    Raises
    ------
    RuntimeError
        If the runtime is not initialized.
    """
    _check_available()
    _finalize()


# -----------------------------------------------------------------------------
# Device Selection
# -----------------------------------------------------------------------------


def _resolve_device(device) -> tuple:
    """Resolve device specification to (backend, device_id).

    Parameters
    ----------
    device : str or int or None
        Device specification:
        - None or 'cpu': Use CPU backend
        - 'gpu' or 'cuda': Use CUDA GPU if available, else raise error
        - 'sycl': Use SYCL GPU if available, else raise error
        - 'auto': Use best available (CUDA > SYCL > CPU)
        - int: Use specific GPU device ID (CUDA preferred, then SYCL)

    Returns
    -------
    tuple
        (backend, device_id) where backend is 'cpu', 'gpu', or 'sycl'
    """
    # Lazy imports to avoid circular dependency
    from hpxpy import gpu as _gpu_mod
    from hpxpy import sycl as _sycl_mod

    if device is None or device == 'cpu':
        return ('cpu', None)

    if device == 'auto':
        # Auto-select: prefer CUDA > SYCL > CPU
        # Both use HPX executors for proper integration
        if _gpu_mod.is_available():
            return ('gpu', 0)
        if _sycl_mod.is_available():
            return ('sycl', 0)
        return ('cpu', None)

    if device == 'gpu' or device == 'cuda':
        # Explicit CUDA GPU request
        if not _gpu_mod.is_available():
            raise RuntimeError(
                "CUDA GPU requested but CUDA is not available. "
                "Use device='auto' for automatic fallback, device='sycl' for SYCL GPUs, "
                "or device='cpu' for CPU."
            )
        return ('gpu', 0)

    if device == 'sycl':
        # Explicit SYCL GPU request (works on Intel, AMD, Apple Silicon via AdaptiveCpp)
        if not _sycl_mod.is_available():
            raise RuntimeError(
                "SYCL requested but not available. "
                "SYCL requires HPX built with HPX_WITH_SYCL=ON and a SYCL compiler. "
                "Use device='auto' for automatic fallback or device='cpu' for CPU."
            )
        return ('sycl', 0)

    if isinstance(device, int):
        # Specific GPU device ID - try CUDA first, then SYCL
        if _gpu_mod.is_available():
            if device >= _gpu_mod.device_count():
                raise ValueError(f"Invalid CUDA device ID {device}. Available: 0-{_gpu_mod.device_count()-1}")
            return ('gpu', device)
        if _sycl_mod.is_available():
            if device >= _sycl_mod.device_count():
                raise ValueError(f"Invalid SYCL device ID {device}. Available: 0-{_sycl_mod.device_count()-1}")
            return ('sycl', device)
        raise RuntimeError(f"GPU device {device} requested but no GPU backend is available.")

    raise ValueError(
        f"Invalid device specification: {device!r}. "
        "Use 'cpu', 'gpu', 'cuda', 'sycl', 'auto', or an integer GPU device ID."
    )


# -----------------------------------------------------------------------------
# Array Creation Functions
# -----------------------------------------------------------------------------


def array(data, dtype=None, copy: bool = True, device=None):
    """Create an HPXPy array from existing data.

    Parameters
    ----------
    data : array_like
        Input data (list, tuple, or NumPy array).
    dtype : numpy.dtype, optional
        Desired data type. If not specified, inferred from data.
    copy : bool, default True
        If True, always copy the data. If False, try to share memory
        when possible (only works with contiguous NumPy arrays).
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        A new HPXPy array on the specified device.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> a = hpx.array([1, 2, 3, 4, 5])
    >>> a = hpx.array(numpy_array, copy=False)  # Zero-copy if possible
    >>> a = hpx.array([1, 2, 3], device='auto')  # GPU if available
    """
    _check_available()
    import numpy as np

    np_array = np.asarray(data, dtype=dtype)
    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.from_numpy(np_array, device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.from_numpy(np_array, device=device_id)
    return _array_from_numpy(np_array, copy=copy)


def from_numpy(arr, copy: bool = False, device=None):
    """Create an HPXPy array from a NumPy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input NumPy array.
    copy : bool, default False
        If False, try to share memory with the input array (zero-copy).
        If True, always make a copy. Only applies to CPU arrays.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        A new HPXPy array on the specified device.

    Notes
    -----
    Zero-copy is only possible for CPU arrays when the input is C-contiguous.
    GPU arrays always copy data.
    """
    _check_available()
    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.from_numpy(arr, device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.from_numpy(arr, device=device_id)
    return _array_from_numpy(arr, copy=copy)


def zeros(shape, dtype=None, device=None):
    """Create an array filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of zeros with the given shape and dtype.

    Examples
    --------
    >>> a = hpx.zeros((10, 10))
    >>> b = hpx.zeros(1000, dtype=np.int32)
    >>> c = hpx.zeros(1000000, device='auto')  # GPU if available
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.zeros(list(shape), device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.zeros(list(shape), device=device_id)
    return _zeros(shape, dtype)


def ones(shape, dtype=None, device=None):
    """Create an array filled with ones.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of ones with the given shape and dtype.
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.ones(list(shape), device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.ones(list(shape), device=device_id)
    return _ones(shape, dtype)


def empty(shape, dtype=None, device=None):
    """Create an uninitialized array.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Uninitialized array with the given shape and dtype.

    Notes
    -----
    The values in the array are not initialized and may contain
    arbitrary data. Use ``zeros`` or ``ones`` if you need initialized values.
    Note: GPU 'empty' currently returns zero-filled array.
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        # GPU arrays don't have true 'empty' - use zeros for safety
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.zeros(list(shape), device=device_id)
    if backend == 'sycl':
        # SYCL arrays don't have true 'empty' - use zeros for safety
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.zeros(list(shape), device=device_id)
    return _empty(shape, dtype)


def arange(start, stop=None, step=1, dtype=None, device=None):
    """Create an array with evenly spaced values.

    Parameters
    ----------
    start : number
        Start of interval (or stop if stop is None).
    stop : number, optional
        End of interval (exclusive).
    step : number, default 1
        Spacing between values.
    dtype : numpy.dtype, optional
        Data type. Inferred from arguments if not specified.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of evenly spaced values.

    Examples
    --------
    >>> hpx.arange(5)           # [0, 1, 2, 3, 4]
    >>> hpx.arange(1, 5)        # [1, 2, 3, 4]
    >>> hpx.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
    >>> hpx.arange(1000000, device='auto')  # GPU if available
    """
    _check_available()
    import numpy as np

    if stop is None:
        stop = start
        start = 0

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        # Create on CPU first, then transfer
        from hpxpy import gpu as _gpu_mod
        np_arr = np.arange(start, stop, step, dtype=dtype)
        return _gpu_mod.from_numpy(np_arr, device=device_id)
    if backend == 'sycl':
        # Create on CPU first, then transfer
        from hpxpy import sycl as _sycl_mod
        np_arr = np.arange(start, stop, step, dtype=dtype)
        return _sycl_mod.from_numpy(np_arr, device=device_id)
    return _arange(start, stop, step, dtype)


def linspace(start, stop, num: int = 50, dtype=None, device=None):
    """Create an array with evenly spaced values over an interval.

    Parameters
    ----------
    start : number
        Start of interval.
    stop : number
        End of interval (inclusive).
    num : int, default 50
        Number of values to generate.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of evenly spaced values.
    """
    _check_available()
    import numpy as np

    np_arr = np.linspace(start, stop, num, dtype=dtype)
    return from_numpy(np_arr, copy=True, device=device)


def full(shape, fill_value, dtype=None, device=None):
    """Create an array filled with a specified value.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    fill_value : scalar
        Fill value.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array filled with the specified value.

    Examples
    --------
    >>> a = hpx.full((10, 10), 3.14)
    >>> b = hpx.full(1000, 42, device='auto')
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.full(list(shape), float(fill_value), device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.full(list(shape), float(fill_value), device=device_id)

    # CPU: create with numpy and convert
    np_arr = np.full(shape, fill_value, dtype=dtype)
    return from_numpy(np_arr, copy=True)


# -----------------------------------------------------------------------------
# Reduction Algorithms
# -----------------------------------------------------------------------------


def sum(arr, axis=None, dtype=None, keepdims: bool = False):
    """Sum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to sum. Default is to sum all elements.
    dtype : numpy.dtype, optional
        Type to use for accumulation.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    ndarray or scalar
        Sum of elements.
    """
    _check_available()
    # Phase 1: only support full reduction (axis=None)
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return _sum(arr)


def prod(arr, axis=None, dtype=None, keepdims: bool = False):
    """Product of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute product.
    dtype : numpy.dtype, optional
        Type to use for accumulation.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    ndarray or scalar
        Product of elements.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return _prod(arr)


def min(arr, axis=None, keepdims: bool = False):
    """Minimum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to find minimum.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    ndarray or scalar
        Minimum value.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return _min(arr)


def max(arr, axis=None, keepdims: bool = False):
    """Maximum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to find maximum.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    ndarray or scalar
        Maximum value.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return _max(arr)


def mean(arr, axis=None, dtype=None, keepdims: bool = False):
    """Compute the arithmetic mean.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute mean.
    dtype : numpy.dtype, optional
        Type to use for computation.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    ndarray or scalar
        Mean of elements.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return _sum(arr) / arr.size


def std(arr, axis=None, dtype=None, ddof: int = 0, keepdims: bool = False):
    """Compute the standard deviation.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute std.
    dtype : numpy.dtype, optional
        Type to use for computation.
    ddof : int, default 0
        Delta degrees of freedom.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    ndarray or scalar
        Standard deviation.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return var(arr, ddof=ddof) ** 0.5


def var(arr, axis=None, dtype=None, ddof: int = 0, keepdims: bool = False):
    """Compute the variance.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute variance.
    dtype : numpy.dtype, optional
        Type to use for computation.
    ddof : int, default 0
        Delta degrees of freedom.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    ndarray or scalar
        Variance.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    # Compute variance using E[X^2] - E[X]^2
    # Note: This is a simple implementation; Phase 2 will use proper parallel algorithm
    n = arr.size
    mean_val = _sum(arr) / n
    # For now, convert to numpy for the squared sum
    np_arr = arr.to_numpy()
    sum_sq = float((np_arr**2).sum())
    return sum_sq / (n - ddof) - (mean_val**2) * n / (n - ddof)


# -----------------------------------------------------------------------------
# Sorting Algorithms
# -----------------------------------------------------------------------------


def sort(arr, axis: int = -1) -> ndarray:
    """Sort an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, default -1
        Axis along which to sort. Default is -1 (last axis).

    Returns
    -------
    ndarray
        Sorted array.
    """
    _check_available()
    # Phase 1: only support 1D arrays
    if arr.ndim != 1:
        raise NotImplementedError("Multi-dimensional sort not yet supported in Phase 1")
    return _sort(arr)


def argsort(arr, axis: int = -1) -> ndarray:
    """Return indices that would sort an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, default -1
        Axis along which to sort.

    Returns
    -------
    ndarray
        Array of indices that sort the input.
    """
    _check_available()
    # Phase 1: delegate to NumPy
    import numpy as np

    np_arr = arr.to_numpy()
    indices = np.argsort(np_arr, axis=axis)
    return from_numpy(indices, copy=True)


def count(arr, value) -> int:
    """Count occurrences of a value.

    Parameters
    ----------
    arr : ndarray
        Input array.
    value : scalar
        Value to count.

    Returns
    -------
    int
        Number of occurrences.
    """
    _check_available()
    return _count(arr, value)


# -----------------------------------------------------------------------------
# Math Functions (Phase 2)
# -----------------------------------------------------------------------------


def sqrt(arr) -> ndarray:
    """Element-wise square root."""
    _check_available()
    return _sqrt(arr)


def square(arr) -> ndarray:
    """Element-wise square."""
    _check_available()
    return _square(arr)


def exp(arr) -> ndarray:
    """Element-wise exponential."""
    _check_available()
    return _exp(arr)


def exp2(arr) -> ndarray:
    """Element-wise 2**x."""
    _check_available()
    return _exp2(arr)


def log(arr) -> ndarray:
    """Element-wise natural logarithm."""
    _check_available()
    return _log(arr)


def log2(arr) -> ndarray:
    """Element-wise base-2 logarithm."""
    _check_available()
    return _log2(arr)


def log10(arr) -> ndarray:
    """Element-wise base-10 logarithm."""
    _check_available()
    return _log10(arr)


def sin(arr) -> ndarray:
    """Element-wise sine."""
    _check_available()
    return _sin(arr)


def cos(arr) -> ndarray:
    """Element-wise cosine."""
    _check_available()
    return _cos(arr)


def tan(arr) -> ndarray:
    """Element-wise tangent."""
    _check_available()
    return _tan(arr)


def arcsin(arr) -> ndarray:
    """Element-wise inverse sine."""
    _check_available()
    return _arcsin(arr)


def arccos(arr) -> ndarray:
    """Element-wise inverse cosine."""
    _check_available()
    return _arccos(arr)


def arctan(arr) -> ndarray:
    """Element-wise inverse tangent."""
    _check_available()
    return _arctan(arr)


def sinh(arr) -> ndarray:
    """Element-wise hyperbolic sine."""
    _check_available()
    return _sinh(arr)


def cosh(arr) -> ndarray:
    """Element-wise hyperbolic cosine."""
    _check_available()
    return _cosh(arr)


def tanh(arr) -> ndarray:
    """Element-wise hyperbolic tangent."""
    _check_available()
    return _tanh(arr)


def floor(arr) -> ndarray:
    """Element-wise floor."""
    _check_available()
    return _floor(arr)


def ceil(arr) -> ndarray:
    """Element-wise ceiling."""
    _check_available()
    return _ceil(arr)


def trunc(arr) -> ndarray:
    """Element-wise truncation toward zero."""
    _check_available()
    return _trunc(arr)


def abs(arr) -> ndarray:
    """Element-wise absolute value."""
    _check_available()
    return _abs(arr)


def sign(arr) -> ndarray:
    """Element-wise sign indicator (-1, 0, or 1)."""
    _check_available()
    return _sign(arr)


# -----------------------------------------------------------------------------
# Scan Operations (Phase 2)
# -----------------------------------------------------------------------------


def cumsum(arr, axis=None) -> ndarray:
    """Cumulative sum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, optional
        Axis along which to compute cumulative sum.
        If None, compute over flattened array.

    Returns
    -------
    ndarray
        Cumulative sum.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _cumsum(arr)


def cumprod(arr, axis=None) -> ndarray:
    """Cumulative product of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, optional
        Axis along which to compute cumulative product.
        If None, compute over flattened array.

    Returns
    -------
    ndarray
        Cumulative product.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _cumprod(arr)


# -----------------------------------------------------------------------------
# Element-wise Functions (Phase 2)
# -----------------------------------------------------------------------------


def maximum(a, b) -> ndarray:
    """Element-wise maximum of two arrays.

    Parameters
    ----------
    a : ndarray
        First input array.
    b : ndarray or scalar
        Second input array or scalar.

    Returns
    -------
    ndarray
        Element-wise maximum.
    """
    _check_available()
    return _maximum(a, b)


def minimum(a, b) -> ndarray:
    """Element-wise minimum of two arrays.

    Parameters
    ----------
    a : ndarray
        First input array.
    b : ndarray or scalar
        Second input array or scalar.

    Returns
    -------
    ndarray
        Element-wise minimum.
    """
    _check_available()
    return _minimum(a, b)


def clip(arr, a_min, a_max) -> ndarray:
    """Clip (limit) the values in an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    a_min : scalar
        Minimum value.
    a_max : scalar
        Maximum value.

    Returns
    -------
    ndarray
        Clipped array.
    """
    _check_available()
    return _clip(arr, a_min, a_max)


def power(arr, exponent) -> ndarray:
    """Element-wise power.

    Parameters
    ----------
    arr : ndarray
        Base array.
    exponent : ndarray or scalar
        Exponent array or scalar.

    Returns
    -------
    ndarray
        arr ** exponent element-wise.
    """
    _check_available()
    return _power(arr, exponent)


def where(condition, x, y) -> ndarray:
    """Return elements chosen from x or y depending on condition.

    Parameters
    ----------
    condition : ndarray of bool
        Where True, yield x, otherwise yield y.
    x : ndarray
        Values to use where condition is True.
    y : ndarray
        Values to use where condition is False.

    Returns
    -------
    ndarray
        Array with elements from x where condition is True, else from y.
    """
    _check_available()
    return _where(condition, x, y)


# -----------------------------------------------------------------------------
# Random Number Generation (Phase 2)
# -----------------------------------------------------------------------------


class random:
    """Random number generation module.

    Provides NumPy-like random number generation functions.
    """

    @staticmethod
    def seed(s: int) -> None:
        """Seed the random number generator.

        Parameters
        ----------
        s : int
            Seed value.
        """
        _check_available()
        _random_module.seed(s)

    @staticmethod
    def uniform(low: float = 0.0, high: float = 1.0, size=None) -> ndarray:
        """Draw samples from a uniform distribution.

        Parameters
        ----------
        low : float, default 0.0
            Lower boundary.
        high : float, default 1.0
            Upper boundary.
        size : int or tuple of ints, optional
            Output shape. Default is a single value.

        Returns
        -------
        ndarray
            Samples from uniform distribution.
        """
        _check_available()
        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)
        return _random_module._uniform(low, high, size)

    @staticmethod
    def randn(*shape) -> ndarray:
        """Return samples from the standard normal distribution.

        Parameters
        ----------
        *shape : ints
            Shape of the output array.

        Returns
        -------
        ndarray
            Samples from standard normal distribution.

        Examples
        --------
        >>> hpx.random.randn(3, 4)  # 3x4 array of standard normal samples
        """
        _check_available()
        if len(shape) == 0:
            shape = (1,)
        return _random_module._randn(list(shape))

    @staticmethod
    def randint(low: int, high: int = None, size=None) -> ndarray:
        """Return random integers from low (inclusive) to high (exclusive).

        Parameters
        ----------
        low : int
            Lowest integer (or highest if high is None).
        high : int, optional
            One above the highest integer.
        size : int or tuple of ints, optional
            Output shape.

        Returns
        -------
        ndarray
            Random integers.
        """
        _check_available()
        if high is None:
            high = low
            low = 0
        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)
        return _random_module._randint(low, high, size)

    @staticmethod
    def rand(*shape) -> ndarray:
        """Random values in a given shape from uniform [0, 1).

        Parameters
        ----------
        *shape : ints
            Shape of the output array.

        Returns
        -------
        ndarray
            Random values.

        Examples
        --------
        >>> hpx.random.rand(3, 4)  # 3x4 array of uniform random values
        """
        _check_available()
        if len(shape) == 0:
            shape = (1,)
        return _random_module._rand(list(shape))


# -----------------------------------------------------------------------------
# Collective Operations (Phase 4)
# -----------------------------------------------------------------------------


def all_reduce(arr, op: str = 'sum'):
    """Combine values from all localities using a reduction operation.

    Each locality contributes its local array, and all localities receive
    the combined result. In single-locality mode, returns the input unchanged.

    Parameters
    ----------
    arr : ndarray
        Local array to contribute.
    op : str, optional
        Reduction operation: 'sum', 'prod', 'min', 'max'. Default: 'sum'.

    Returns
    -------
    ndarray
        Combined result (same on all localities).

    Examples
    --------
    >>> local_sum = hpx.from_numpy(np.array([1.0, 2.0, 3.0]))
    >>> global_sum = hpx.all_reduce(local_sum, op='sum')
    """
    _check_available()
    from hpxpy._core import ReduceOp
    op_map = {
        'sum': ReduceOp.sum,
        'prod': ReduceOp.prod,
        'min': ReduceOp.min,
        'max': ReduceOp.max,
    }
    if op not in op_map:
        raise ValueError(f"Unknown reduction operation: {op}. Use one of: {list(op_map.keys())}")
    return _all_reduce(arr, op_map[op])


def broadcast(arr, root: int = 0):
    """Broadcast array from root locality to all localities.

    In single-locality mode, returns a copy of the input.

    Parameters
    ----------
    arr : ndarray
        Array to broadcast (only used on root locality).
    root : int, optional
        Locality ID to broadcast from. Default: 0.

    Returns
    -------
    ndarray
        Broadcasted array (same on all localities).

    Examples
    --------
    >>> data = hpx.from_numpy(np.array([1.0, 2.0, 3.0]))
    >>> shared = hpx.broadcast(data, root=0)
    """
    _check_available()
    return _broadcast(arr, root)


def gather(arr, root: int = 0):
    """Gather arrays from all localities to root.

    In single-locality mode, returns a list containing just the input.

    Parameters
    ----------
    arr : ndarray
        Local array to contribute.
    root : int, optional
        Locality ID to gather to. Default: 0.

    Returns
    -------
    list
        List of numpy arrays from all localities (only valid on root).

    Examples
    --------
    >>> local_data = hpx.from_numpy(np.array([1.0, 2.0, 3.0]))
    >>> all_data = hpx.gather(local_data, root=0)
    >>> print(len(all_data))  # Number of localities
    """
    _check_available()
    return _gather(arr, root)


def scatter(arr, root: int = 0):
    """Scatter array from root to all localities.

    The array on root is divided evenly among all localities.
    In single-locality mode, returns a copy of the input.

    Parameters
    ----------
    arr : ndarray
        Array to scatter (only used on root).
    root : int, optional
        Locality ID to scatter from. Default: 0.

    Returns
    -------
    ndarray
        This locality's portion of the scattered array.

    Examples
    --------
    >>> full_data = hpx.arange(1000)
    >>> local_chunk = hpx.scatter(full_data, root=0)
    """
    _check_available()
    return _scatter(arr, root)


def barrier(name: str = "hpxpy_barrier"):
    """Synchronize all localities.

    All localities wait until everyone reaches the barrier.
    In single-locality mode, this is a no-op.

    Parameters
    ----------
    name : str, optional
        Name for the barrier. Default: "hpxpy_barrier".

    Examples
    --------
    >>> # Ensure all localities have finished computation
    >>> hpx.barrier()
    """
    _check_available()
    _barrier(name)


# -----------------------------------------------------------------------------
# Multi-Locality Launcher (Phase 4)
# -----------------------------------------------------------------------------

# Import the launcher module for multi-locality support
from hpxpy import launcher


# -----------------------------------------------------------------------------
# GPU Support (Phase 5)
# -----------------------------------------------------------------------------

# Import the GPU module for CUDA/GPU support (via HPX cuda_executor)
from hpxpy import gpu

# Import the SYCL module for cross-platform GPU support (via HPX sycl_executor)
# Supports: Intel GPUs (oneAPI), AMD GPUs (HIP), Apple Silicon (AdaptiveCpp Metal)
from hpxpy import sycl
