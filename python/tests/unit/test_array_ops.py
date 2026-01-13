# HPXPy Array Operations Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy array operations (slicing, indexing, etc.)."""

import numpy as np
import pytest


class TestArraySlicing:
    """Test array slicing operations."""

    def test_slice_basic(self, hpx_runtime):
        """Basic slice: arr[2:5]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[2:5]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [2, 3, 4])

    def test_slice_from_start(self, hpx_runtime):
        """Slice from start: arr[:5]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[:5]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [0, 1, 2, 3, 4])

    def test_slice_to_end(self, hpx_runtime):
        """Slice to end: arr[5:]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[5:]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [5, 6, 7, 8, 9])

    def test_slice_full(self, hpx_runtime):
        """Full slice: arr[:]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[:]

        assert sliced.shape == (10,)
        np.testing.assert_array_equal(sliced.to_numpy(), np.arange(10))

    def test_slice_step(self, hpx_runtime):
        """Step slice: arr[::2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[::2]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [0, 2, 4, 6, 8])

    def test_slice_step_with_start(self, hpx_runtime):
        """Step slice with start: arr[1::2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[1::2]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [1, 3, 5, 7, 9])

    def test_slice_step_with_range(self, hpx_runtime):
        """Step slice with range: arr[2:8:2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[2:8:2]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [2, 4, 6])

    def test_slice_negative_start(self, hpx_runtime):
        """Negative start: arr[-3:]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[-3:]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [7, 8, 9])

    def test_slice_negative_stop(self, hpx_runtime):
        """Negative stop: arr[:-3]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[:-3]

        assert sliced.shape == (7,)
        np.testing.assert_array_equal(sliced.to_numpy(), [0, 1, 2, 3, 4, 5, 6])

    def test_slice_negative_both(self, hpx_runtime):
        """Negative start and stop: arr[-5:-2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[-5:-2]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [5, 6, 7])

    def test_slice_empty_result(self, hpx_runtime):
        """Slice that results in empty array: arr[5:5]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[5:5]

        assert sliced.shape == (0,)
        assert sliced.size == 0

    def test_slice_single_element(self, hpx_runtime):
        """Slice with single element: arr[5:6]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[5:6]

        assert sliced.shape == (1,)
        np.testing.assert_array_equal(sliced.to_numpy(), [5])


class TestSliceDtypes:
    """Test slicing with different data types."""

    def test_slice_float32(self, hpx_runtime):
        """Slice float32 array."""
        np_arr = np.arange(10, dtype=np.float32)
        arr = hpx_runtime.from_numpy(np_arr)
        sliced = arr[2:5]

        assert sliced.dtype == np.float32
        np.testing.assert_array_equal(sliced.to_numpy(), np_arr[2:5])

    def test_slice_int64(self, hpx_runtime):
        """Slice int64 array."""
        np_arr = np.arange(10, dtype=np.int64)
        arr = hpx_runtime.from_numpy(np_arr)
        sliced = arr[::2]

        assert sliced.dtype == np.int64
        np.testing.assert_array_equal(sliced.to_numpy(), np_arr[::2])

    def test_slice_int32(self, hpx_runtime):
        """Slice int32 array."""
        np_arr = np.arange(10, dtype=np.int32)
        arr = hpx_runtime.from_numpy(np_arr)
        sliced = arr[-3:]

        assert sliced.dtype == np.int32
        np.testing.assert_array_equal(sliced.to_numpy(), np_arr[-3:])


class TestSliceChaining:
    """Test chaining of slice operations."""

    def test_double_slice(self, hpx_runtime):
        """Apply two slices: arr[2:8][1:4]."""
        arr = hpx_runtime.arange(10)
        first = arr[2:8]  # [2, 3, 4, 5, 6, 7]
        second = first[1:4]  # [3, 4, 5]

        assert second.shape == (3,)
        np.testing.assert_array_equal(second.to_numpy(), [3, 4, 5])

    def test_slice_then_step(self, hpx_runtime):
        """Apply slice then step: arr[1:9][::2]."""
        arr = hpx_runtime.arange(10)
        first = arr[1:9]  # [1, 2, 3, 4, 5, 6, 7, 8]
        second = first[::2]  # [1, 3, 5, 7]

        assert second.shape == (4,)
        np.testing.assert_array_equal(second.to_numpy(), [1, 3, 5, 7])


class TestSliceNumpyComparison:
    """Compare HPXPy slicing with NumPy behavior."""

    @pytest.mark.parametrize("slice_spec", [
        slice(2, 5),
        slice(None, 5),
        slice(5, None),
        slice(None, None),
        slice(None, None, 2),
        slice(1, None, 2),
        slice(2, 8, 2),
        slice(-3, None),
        slice(None, -3),
        slice(-5, -2),
        slice(5, 5),
        slice(5, 6),
    ])
    def test_slice_matches_numpy(self, hpx_runtime, slice_spec):
        """Verify HPXPy slice matches NumPy slice."""
        np_arr = np.arange(10, dtype=np.float64)
        arr = hpx_runtime.from_numpy(np_arr)

        expected = np_arr[slice_spec]
        result = arr[slice_spec]

        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.to_numpy(), expected)
