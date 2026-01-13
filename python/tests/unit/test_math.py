# Phase 2: Math function tests
#
# SPDX-License-Identifier: BSL-1.0

import numpy as np
import pytest

import hpxpy as hpx


class TestBasicMath:
    """Test basic math functions."""

    def test_sqrt(self, hpx_runtime):
        a = hpx.array([0, 1, 4, 9, 16, 25])
        result = hpx.sqrt(a)
        expected = np.sqrt([0, 1, 4, 9, 16, 25])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_square(self, hpx_runtime):
        a = hpx.arange(10)
        result = hpx.square(a)
        expected = np.arange(10) ** 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_abs(self, hpx_runtime):
        a = hpx.array([-3, -2, -1, 0, 1, 2, 3])
        result = hpx.abs(a)
        expected = np.abs([-3, -2, -1, 0, 1, 2, 3])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_sign(self, hpx_runtime):
        a = hpx.array([-5, -1, 0, 1, 5])
        result = hpx.sign(a)
        expected = np.sign([-5, -1, 0, 1, 5])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestExponentialLog:
    """Test exponential and logarithmic functions."""

    def test_exp(self, hpx_runtime):
        a = hpx.array([0.0, 1.0, 2.0])
        result = hpx.exp(a)
        expected = np.exp([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_log(self, hpx_runtime):
        a = hpx.array([1, np.e, np.e**2])
        result = hpx.log(a)
        expected = np.log([1, np.e, np.e**2])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_log2(self, hpx_runtime):
        a = hpx.array([1, 2, 4, 8])
        result = hpx.log2(a)
        expected = np.log2([1, 2, 4, 8])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_log10(self, hpx_runtime):
        a = hpx.array([1, 10, 100, 1000])
        result = hpx.log10(a)
        expected = np.log10([1, 10, 100, 1000])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestTrigonometric:
    """Test trigonometric functions."""

    def test_sin(self, hpx_runtime):
        a = hpx.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
        result = hpx.sin(a)
        expected = np.sin([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_cos(self, hpx_runtime):
        a = hpx.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
        result = hpx.cos(a)
        expected = np.cos([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_tan(self, hpx_runtime):
        a = hpx.array([0, np.pi/6, np.pi/4, np.pi/3])
        result = hpx.tan(a)
        expected = np.tan([0, np.pi/6, np.pi/4, np.pi/3])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_arcsin(self, hpx_runtime):
        a = hpx.array([0, 0.5, 1])
        result = hpx.arcsin(a)
        expected = np.arcsin([0, 0.5, 1])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_arccos(self, hpx_runtime):
        a = hpx.array([0, 0.5, 1])
        result = hpx.arccos(a)
        expected = np.arccos([0, 0.5, 1])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_arctan(self, hpx_runtime):
        a = hpx.array([0, 1, np.sqrt(3)])
        result = hpx.arctan(a)
        expected = np.arctan([0, 1, np.sqrt(3)])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestHyperbolic:
    """Test hyperbolic functions."""

    def test_sinh(self, hpx_runtime):
        a = hpx.array([0.0, 1.0, 2.0])
        result = hpx.sinh(a)
        expected = np.sinh([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_cosh(self, hpx_runtime):
        a = hpx.array([0.0, 1.0, 2.0])
        result = hpx.cosh(a)
        expected = np.cosh([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_tanh(self, hpx_runtime):
        a = hpx.array([0.0, 1.0, 2.0])
        result = hpx.tanh(a)
        expected = np.tanh([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestRounding:
    """Test rounding functions."""

    def test_floor(self, hpx_runtime):
        a = hpx.array([1.2, 2.5, 3.7, -1.2, -2.5])
        result = hpx.floor(a)
        expected = np.floor([1.2, 2.5, 3.7, -1.2, -2.5])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_ceil(self, hpx_runtime):
        a = hpx.array([1.2, 2.5, 3.7, -1.2, -2.5])
        result = hpx.ceil(a)
        expected = np.ceil([1.2, 2.5, 3.7, -1.2, -2.5])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_trunc(self, hpx_runtime):
        a = hpx.array([1.2, 2.5, 3.7, -1.2, -2.5])
        result = hpx.trunc(a)
        expected = np.trunc([1.2, 2.5, 3.7, -1.2, -2.5])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestScanOperations:
    """Test scan (prefix) operations."""

    def test_cumsum_basic(self, hpx_runtime):
        a = hpx.array([1, 2, 3, 4, 5])
        result = hpx.cumsum(a)
        expected = np.cumsum([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_cumsum_float(self, hpx_runtime):
        a = hpx.array([0.5, 1.5, 2.5])
        result = hpx.cumsum(a)
        expected = np.cumsum([0.5, 1.5, 2.5])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_cumprod_basic(self, hpx_runtime):
        a = hpx.array([1, 2, 3, 4, 5])
        result = hpx.cumprod(a)
        expected = np.cumprod([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_cumprod_float(self, hpx_runtime):
        a = hpx.array([0.5, 2.0, 3.0])
        result = hpx.cumprod(a)
        expected = np.cumprod([0.5, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestElementWiseFunctions:
    """Test element-wise functions."""

    def test_maximum(self, hpx_runtime):
        a = hpx.array([1, 5, 3])
        b = hpx.array([2, 4, 6])
        result = hpx.maximum(a, b)
        expected = np.maximum([1, 5, 3], [2, 4, 6])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_minimum(self, hpx_runtime):
        a = hpx.array([1, 5, 3])
        b = hpx.array([2, 4, 6])
        result = hpx.minimum(a, b)
        expected = np.minimum([1, 5, 3], [2, 4, 6])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_clip(self, hpx_runtime):
        a = hpx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = hpx.clip(a, 3, 7)
        expected = np.clip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 7)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_power(self, hpx_runtime):
        a = hpx.arange(1, 6)
        result = hpx.power(a, 2)
        expected = np.power(np.arange(1, 6), 2)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestRandomGeneration:
    """Test random number generation."""

    def test_uniform_shape(self, hpx_runtime):
        result = hpx.random.uniform(0, 1, size=(100,))
        assert result.shape == (100,)

    def test_uniform_range(self, hpx_runtime):
        result = hpx.random.uniform(5, 10, size=(1000,))
        arr = result.to_numpy()
        assert np.all(arr >= 5)
        assert np.all(arr < 10)

    def test_randn_shape(self, hpx_runtime):
        result = hpx.random.randn(50, 50)
        assert result.shape == (50, 50)

    def test_randn_distribution(self, hpx_runtime):
        hpx.random.seed(42)
        result = hpx.random.randn(10000)
        arr = result.to_numpy()
        # Check roughly normal distribution
        assert abs(np.mean(arr)) < 0.1
        assert abs(np.std(arr) - 1.0) < 0.1

    def test_randint_range(self, hpx_runtime):
        result = hpx.random.randint(0, 10, size=(1000,))
        arr = result.to_numpy()
        assert np.all(arr >= 0)
        assert np.all(arr < 10)

    def test_rand_shape(self, hpx_runtime):
        result = hpx.random.rand(10, 20)
        assert result.shape == (10, 20)

    def test_rand_range(self, hpx_runtime):
        result = hpx.random.rand(1000)
        arr = result.to_numpy()
        assert np.all(arr >= 0)
        assert np.all(arr < 1)

    def test_seed_reproducibility(self, hpx_runtime):
        hpx.random.seed(12345)
        a = hpx.random.rand(10).to_numpy()
        hpx.random.seed(12345)
        b = hpx.random.rand(10).to_numpy()
        np.testing.assert_array_equal(a, b)
