# HPXPy Array Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy array creation and properties."""

import numpy as np
import pytest


class TestArrayCreation:
    """Test array creation functions."""

    def test_zeros_1d(self, hpx_runtime):
        """Create a 1D array of zeros."""
        arr = hpx_runtime.zeros(10)
        assert arr.shape == (10,)
        assert arr.size == 10
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(10))

    def test_zeros_2d(self, hpx_runtime):
        """Create a 2D array of zeros."""
        arr = hpx_runtime.zeros((5, 10))
        assert arr.shape == (5, 10)
        assert arr.size == 50
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros((5, 10)))

    def test_ones_1d(self, hpx_runtime):
        """Create a 1D array of ones."""
        arr = hpx_runtime.ones(10)
        assert arr.shape == (10,)
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(10))

    def test_ones_2d(self, hpx_runtime):
        """Create a 2D array of ones."""
        arr = hpx_runtime.ones((3, 4))
        assert arr.shape == (3, 4)
        np.testing.assert_array_equal(arr.to_numpy(), np.ones((3, 4)))

    def test_empty(self, hpx_runtime):
        """Create an uninitialized array."""
        arr = hpx_runtime.empty(100)
        assert arr.shape == (100,)
        assert arr.size == 100
        # Can't test values since they're uninitialized

    def test_arange_basic(self, hpx_runtime):
        """Create an array with arange."""
        arr = hpx_runtime.arange(10)
        expected = np.arange(10, dtype=np.float64)
        np.testing.assert_array_equal(arr.to_numpy(), expected)

    def test_arange_start_stop(self, hpx_runtime):
        """Create an array with arange(start, stop)."""
        arr = hpx_runtime.arange(5, 15)
        expected = np.arange(5, 15, dtype=np.float64)
        np.testing.assert_array_equal(arr.to_numpy(), expected)

    def test_arange_step(self, hpx_runtime):
        """Create an array with arange(start, stop, step)."""
        arr = hpx_runtime.arange(0, 10, 2)
        expected = np.arange(0, 10, 2, dtype=np.float64)
        np.testing.assert_array_equal(arr.to_numpy(), expected)


class TestArrayDtypes:
    """Test array data types."""

    def test_zeros_float32(self, hpx_runtime):
        """Create float32 array."""
        arr = hpx_runtime.zeros(10, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_zeros_float64(self, hpx_runtime):
        """Create float64 array (default)."""
        arr = hpx_runtime.zeros(10)
        assert arr.dtype == np.float64

    def test_zeros_int32(self, hpx_runtime):
        """Create int32 array."""
        arr = hpx_runtime.zeros(10, dtype=np.int32)
        assert arr.dtype == np.int32

    def test_zeros_int64(self, hpx_runtime):
        """Create int64 array."""
        arr = hpx_runtime.zeros(10, dtype=np.int64)
        assert arr.dtype == np.int64


class TestArrayFromNumpy:
    """Test array creation from NumPy arrays."""

    def test_from_numpy_copy(self, hpx_runtime):
        """Create array from NumPy with copy."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hpx_arr = hpx_runtime.from_numpy(np_arr, copy=True)

        np.testing.assert_array_equal(hpx_arr.to_numpy(), np_arr)

        # Modifying original should not affect copy
        np_arr[0] = 999.0
        assert hpx_arr.to_numpy()[0] == 1.0

    def test_from_numpy_no_copy_is_view(self, hpx_runtime):
        """from_numpy with copy=False should share memory (zero-copy)."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hpx_arr = hpx_runtime.from_numpy(np_arr, copy=False)

        np.testing.assert_array_equal(hpx_arr.to_numpy(), np_arr)

        # Modifying original numpy array should be visible through hpx array
        np_arr[0] = 999.0
        assert hpx_arr.to_numpy()[0] == 999.0

    def test_from_numpy_no_copy_to_numpy_roundtrip(self, hpx_runtime):
        """to_numpy on a view should return same data."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hpx_arr = hpx_runtime.from_numpy(np_arr, copy=False)
        result = hpx_arr.to_numpy()

        # Both should share the same underlying data
        np.testing.assert_array_equal(result, np_arr)

        # Modifying original should be visible in result (shared memory)
        np_arr[2] = 888.0
        assert result[2] == 888.0

    def test_array_from_list(self, hpx_runtime):
        """Create array from Python list."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr = hpx_runtime.array(data)
        np.testing.assert_array_equal(arr.to_numpy(), np.array(data))

    def test_array_from_nested_list(self, hpx_runtime):
        """Create 2D array from nested list."""
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        arr = hpx_runtime.array(data)
        np.testing.assert_array_equal(arr.to_numpy(), np.array(data))


class TestArrayProperties:
    """Test array properties."""

    def test_shape(self, sample_array):
        """Test shape property."""
        assert sample_array.shape == (5,)

    def test_dtype(self, sample_array):
        """Test dtype property."""
        assert sample_array.dtype == np.float64

    def test_size(self, sample_array):
        """Test size property."""
        assert sample_array.size == 5

    def test_ndim(self, sample_array):
        """Test ndim property."""
        assert sample_array.ndim == 1

    def test_nbytes(self, sample_array):
        """Test nbytes property."""
        assert sample_array.nbytes == 5 * 8  # 5 elements * 8 bytes (float64)


class TestArrayToNumpy:
    """Test conversion to NumPy."""

    def test_to_numpy_values(self, hpx_runtime):
        """to_numpy should return correct values."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hpx_arr = hpx_runtime.from_numpy(np_arr)
        result = hpx_arr.to_numpy()

        np.testing.assert_array_equal(result, np_arr)

    def test_to_numpy_roundtrip(self, hpx_runtime):
        """Roundtrip through HPXPy should preserve data."""
        original = np.random.randn(100)
        hpx_arr = hpx_runtime.from_numpy(original)
        result = hpx_arr.to_numpy()

        np.testing.assert_array_almost_equal(result, original)

    def test_buffer_protocol(self, hpx_runtime):
        """HPXPy arrays should support buffer protocol."""
        hpx_arr = hpx_runtime.arange(10)

        # Create NumPy array using buffer protocol
        np_view = np.asarray(hpx_arr)

        expected = np.arange(10, dtype=np.float64)
        np.testing.assert_array_equal(np_view, expected)
