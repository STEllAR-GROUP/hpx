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
    "arange",
    "linspace",
    "from_numpy",
    # Array class
    "ndarray",
    # Algorithms
    "sum",
    "prod",
    "min",
    "max",
    "mean",
    "std",
    "var",
    "sort",
    "argsort",
    "count",
    # Execution policies
    "execution",
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
        # Execution module
        execution,
    )

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
# Array Creation Functions
# -----------------------------------------------------------------------------


def array(data, dtype=None, copy: bool = True) -> ndarray:
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

    Returns
    -------
    ndarray
        A new HPXPy array.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> a = hpx.array([1, 2, 3, 4, 5])
    >>> a = hpx.array(numpy_array, copy=False)  # Zero-copy if possible
    """
    _check_available()
    import numpy as np

    np_array = np.asarray(data, dtype=dtype)
    return _array_from_numpy(np_array, copy=copy)


def from_numpy(arr, copy: bool = False) -> ndarray:
    """Create an HPXPy array from a NumPy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input NumPy array.
    copy : bool, default False
        If False, try to share memory with the input array (zero-copy).
        If True, always make a copy.

    Returns
    -------
    ndarray
        A new HPXPy array.

    Notes
    -----
    Zero-copy is only possible when the input array is C-contiguous.
    If zero-copy is used, modifying the original NumPy array will
    also modify the HPXPy array.
    """
    _check_available()
    return _array_from_numpy(arr, copy=copy)


def zeros(shape, dtype=None) -> ndarray:
    """Create an array filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.

    Returns
    -------
    ndarray
        Array of zeros with the given shape and dtype.

    Examples
    --------
    >>> a = hpx.zeros((10, 10))
    >>> b = hpx.zeros(1000, dtype=np.int32)
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)  # Ensure it's a dtype instance
    if isinstance(shape, int):
        shape = (shape,)
    return _zeros(shape, dtype)


def ones(shape, dtype=None) -> ndarray:
    """Create an array filled with ones.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.

    Returns
    -------
    ndarray
        Array of ones with the given shape and dtype.
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)  # Ensure it's a dtype instance
    if isinstance(shape, int):
        shape = (shape,)
    return _ones(shape, dtype)


def empty(shape, dtype=None) -> ndarray:
    """Create an uninitialized array.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.

    Returns
    -------
    ndarray
        Uninitialized array with the given shape and dtype.

    Notes
    -----
    The values in the array are not initialized and may contain
    arbitrary data. Use ``zeros`` or ``ones`` if you need initialized values.
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)  # Ensure it's a dtype instance
    if isinstance(shape, int):
        shape = (shape,)
    return _empty(shape, dtype)


def arange(start, stop=None, step=1, dtype=None) -> ndarray:
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

    Returns
    -------
    ndarray
        Array of evenly spaced values.

    Examples
    --------
    >>> hpx.arange(5)           # [0, 1, 2, 3, 4]
    >>> hpx.arange(1, 5)        # [1, 2, 3, 4]
    >>> hpx.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
    """
    _check_available()
    if stop is None:
        stop = start
        start = 0
    return _arange(start, stop, step, dtype)


def linspace(start, stop, num: int = 50, dtype=None) -> ndarray:
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

    Returns
    -------
    ndarray
        Array of evenly spaced values.
    """
    _check_available()
    import numpy as np

    # For Phase 1, delegate to NumPy and convert
    np_arr = np.linspace(start, stop, num, dtype=dtype)
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
