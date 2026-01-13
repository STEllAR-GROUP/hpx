# Phase 8: Reshape & Views - What Was Done

## Summary

Phase 8 adds reshape, flatten, and ravel methods to HPXPy's ndarray class.

**Status:** Complete (2026-01-13)

## Objectives

1. Add `reshape()` method with view semantics
2. Add `flatten()` method (always returns copy)
3. Add `ravel()` method (view when possible)
4. Support -1 dimension inference
5. Ensure NumPy-compatible behavior

## Implemented Features

- [x] **reshape()**: Returns array with new shape
  - Returns view for contiguous arrays
  - Returns copy for non-contiguous arrays
  - Supports -1 dimension inference
- [x] **flatten()**: Returns 1D copy of array
  - Always returns a copy
  - Handles non-contiguous arrays
- [x] **ravel()**: Returns flattened array
  - Returns view for contiguous arrays
  - Returns copy for non-contiguous arrays
- [x] **is_contiguous()**: Helper method to check C-order contiguity

## Files Modified

| File | Description |
|------|-------------|
| `python/src/bindings/ndarray.hpp` | Add `is_contiguous()`, `reshape()`, `flatten()`, `ravel()` |
| `python/src/bindings/array_bindings.cpp` | Add Python bindings with docstrings |
| `python/tests/unit/test_array_ops.py` | Add 33 new tests for reshape operations |
| `agents/future_work.md` | Update to remove implemented features |

## API Additions

```python
import hpxpy as hpx
import numpy as np

hpx.init()

# Create array
arr = hpx.arange(12)

# reshape() - change array dimensions
arr2d = arr.reshape((3, 4))        # 3x4 array
arr3d = arr.reshape((2, 2, 3))     # 2x2x3 array
arr_infer = arr.reshape((3, -1))   # Infer second dim -> (3, 4)

# flatten() - always returns copy
flat = arr2d.flatten()  # [0, 1, 2, ..., 11]

# ravel() - view if possible, copy otherwise
raveled = arr2d.ravel()  # [0, 1, 2, ..., 11]

# Non-contiguous arrays (from step slicing) get copied
sliced = arr[::2]  # [0, 2, 4, 6, 8, 10] - non-contiguous
raveled_copy = sliced.ravel()  # Returns copy

hpx.finalize()
```

## Test Results

- **33 new reshape/flatten/ravel tests**
- All existing tests continue to pass (284 total)
- Tests verify NumPy-compatible behavior with parametrized comparisons

## Implementation Notes

1. **View semantics**: `reshape()` and `ravel()` return views (shared data) for contiguous arrays, reusing the `slice_view_tag` constructor from Phase 6.

2. **Contiguity check**: `is_contiguous()` verifies C-order (row-major) contiguity by checking strides match expected pattern.

3. **Dimension inference**: The -1 value in reshape is calculated from total size divided by product of other dimensions.

4. **Non-contiguous handling**: When arrays have non-contiguous strides (e.g., from step slicing), reshape/ravel must copy data to create contiguous output.

## Future Work

Multi-dimensional slicing (`arr[1:3, 2:5]`) is deferred to Phase 8 (renamed) in `agents/future_work.md`.
