# HPXPy Algorithm Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy algorithms."""

import numpy as np
import pytest


class TestSumReduction:
    """Test sum reduction."""

    def test_sum_basic(self, hpx_runtime):
        """Basic sum test."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpx_runtime.sum(arr)
        assert result == 15.0

    def test_sum_single_element(self, hpx_runtime):
        """Sum of single element array."""
        arr = hpx_runtime.array([42.0])
        result = hpx_runtime.sum(arr)
        assert result == 42.0

    def test_sum_large_array(self, large_array, hpx_runtime):
        """Sum of large array should match NumPy."""
        np_arr = large_array.to_numpy()
        hpx_result = hpx_runtime.sum(large_array)
        np_result = np.sum(np_arr)
        np.testing.assert_allclose(hpx_result, np_result, rtol=1e-10)

    def test_sum_integers(self, hpx_runtime):
        """Sum of integer array."""
        arr = hpx_runtime.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = hpx_runtime.sum(arr)
        assert result == 15

    def test_sum_float32(self, hpx_runtime):
        """Sum of float32 array."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = hpx_runtime.sum(arr)
        np.testing.assert_allclose(result, 6.0, rtol=1e-5)


class TestProdReduction:
    """Test product reduction."""

    def test_prod_basic(self, hpx_runtime):
        """Basic product test."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpx_runtime.prod(arr)
        assert result == 120.0

    def test_prod_with_zero(self, hpx_runtime):
        """Product with zero should be zero."""
        arr = hpx_runtime.array([1.0, 2.0, 0.0, 4.0])
        result = hpx_runtime.prod(arr)
        assert result == 0.0

    def test_prod_single_element(self, hpx_runtime):
        """Product of single element."""
        arr = hpx_runtime.array([7.0])
        result = hpx_runtime.prod(arr)
        assert result == 7.0


class TestMinMaxReduction:
    """Test min/max reductions."""

    def test_min_basic(self, hpx_runtime):
        """Basic min test."""
        arr = hpx_runtime.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = hpx_runtime.min(arr)
        assert result == 1.0

    def test_max_basic(self, hpx_runtime):
        """Basic max test."""
        arr = hpx_runtime.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = hpx_runtime.max(arr)
        assert result == 5.0

    def test_min_negative(self, hpx_runtime):
        """Min with negative numbers."""
        arr = hpx_runtime.array([-5.0, -2.0, -10.0, -1.0])
        result = hpx_runtime.min(arr)
        assert result == -10.0

    def test_max_negative(self, hpx_runtime):
        """Max with negative numbers."""
        arr = hpx_runtime.array([-5.0, -2.0, -10.0, -1.0])
        result = hpx_runtime.max(arr)
        assert result == -1.0

    def test_min_large_array(self, large_array, hpx_runtime):
        """Min of large array should match NumPy."""
        np_arr = large_array.to_numpy()
        hpx_result = hpx_runtime.min(large_array)
        np_result = np.min(np_arr)
        np.testing.assert_allclose(hpx_result, np_result)

    def test_max_large_array(self, large_array, hpx_runtime):
        """Max of large array should match NumPy."""
        np_arr = large_array.to_numpy()
        hpx_result = hpx_runtime.max(large_array)
        np_result = np.max(np_arr)
        np.testing.assert_allclose(hpx_result, np_result)

    def test_min_empty_raises(self, hpx_runtime):
        """Min of empty array should raise."""
        arr = hpx_runtime.array([])
        with pytest.raises(RuntimeError, match="empty"):
            hpx_runtime.min(arr)

    def test_max_empty_raises(self, hpx_runtime):
        """Max of empty array should raise."""
        arr = hpx_runtime.array([])
        with pytest.raises(RuntimeError, match="empty"):
            hpx_runtime.max(arr)


class TestMeanStdVar:
    """Test mean, std, var calculations."""

    def test_mean_basic(self, hpx_runtime):
        """Basic mean test."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpx_runtime.mean(arr)
        assert result == 3.0

    def test_mean_matches_numpy(self, large_array, hpx_runtime):
        """Mean should match NumPy."""
        np_arr = large_array.to_numpy()
        hpx_result = hpx_runtime.mean(large_array)
        np_result = np.mean(np_arr)
        np.testing.assert_allclose(hpx_result, np_result, rtol=1e-10)

    def test_std_basic(self, hpx_runtime):
        """Basic std test."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpx_runtime.std(arr)
        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_var_basic(self, hpx_runtime):
        """Basic variance test."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpx_runtime.var(arr)
        expected = np.var([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestSort:
    """Test sorting algorithms."""

    def test_sort_basic(self, hpx_runtime):
        """Basic sort test."""
        arr = hpx_runtime.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        result = hpx_runtime.sort(arr)
        expected = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_sort_already_sorted(self, hpx_runtime):
        """Sort already sorted array."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpx_runtime.sort(arr)
        np.testing.assert_array_equal(result.to_numpy(), arr.to_numpy())

    def test_sort_reverse_sorted(self, hpx_runtime):
        """Sort reverse-sorted array."""
        arr = hpx_runtime.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = hpx_runtime.sort(arr)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_sort_single_element(self, hpx_runtime):
        """Sort single element array."""
        arr = hpx_runtime.array([42.0])
        result = hpx_runtime.sort(arr)
        np.testing.assert_array_equal(result.to_numpy(), [42.0])

    def test_sort_preserves_original(self, hpx_runtime):
        """Sort should not modify original array."""
        original = [3.0, 1.0, 4.0, 1.0, 5.0]
        arr = hpx_runtime.array(original)
        _ = hpx_runtime.sort(arr)
        np.testing.assert_array_equal(arr.to_numpy(), original)

    def test_sort_matches_numpy(self, hpx_runtime):
        """Sort should match NumPy sort."""
        np_arr = np.random.randn(1000)
        hpx_arr = hpx_runtime.from_numpy(np_arr)

        hpx_result = hpx_runtime.sort(hpx_arr)
        np_result = np.sort(np_arr)

        np.testing.assert_array_equal(hpx_result.to_numpy(), np_result)

    def test_sort_integers(self, hpx_runtime):
        """Sort integer array."""
        arr = hpx_runtime.array([5, 2, 8, 1, 9], dtype=np.int64)
        result = hpx_runtime.sort(arr)
        expected = np.array([1, 2, 5, 8, 9], dtype=np.int64)
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestCount:
    """Test count algorithm."""

    def test_count_basic(self, hpx_runtime):
        """Basic count test."""
        arr = hpx_runtime.array([1.0, 2.0, 1.0, 3.0, 1.0])
        result = hpx_runtime.count(arr, 1.0)
        assert result == 3

    def test_count_not_found(self, hpx_runtime):
        """Count of non-existent value."""
        arr = hpx_runtime.array([1.0, 2.0, 3.0])
        result = hpx_runtime.count(arr, 99.0)
        assert result == 0

    def test_count_integers(self, hpx_runtime):
        """Count in integer array."""
        arr = hpx_runtime.array([1, 2, 1, 3, 1, 2], dtype=np.int64)
        assert hpx_runtime.count(arr, 1) == 3
        assert hpx_runtime.count(arr, 2) == 2
        assert hpx_runtime.count(arr, 3) == 1


class TestArgsort:
    """Test argsort algorithm."""

    def test_argsort_basic(self, hpx_runtime):
        """Basic argsort test."""
        arr = hpx_runtime.array([3.0, 1.0, 2.0])
        result = hpx_runtime.argsort(arr)
        expected = np.array([1, 2, 0])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_argsort_matches_numpy(self, hpx_runtime):
        """Argsort should match NumPy argsort."""
        np_arr = np.random.randn(100)
        hpx_arr = hpx_runtime.from_numpy(np_arr)

        hpx_result = hpx_runtime.argsort(hpx_arr)
        np_result = np.argsort(np_arr)

        np.testing.assert_array_equal(hpx_result.to_numpy(), np_result)
