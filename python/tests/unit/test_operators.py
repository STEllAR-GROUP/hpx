# Phase 2: Operator tests
#
# SPDX-License-Identifier: BSL-1.0

import numpy as np
import pytest

import hpxpy as hpx


class TestArithmeticOperators:
    """Test arithmetic operator overloading."""

    def test_add_arrays(self, hpx_runtime):
        a = hpx.arange(10)
        b = hpx.ones(10)
        result = a + b
        expected = np.arange(10) + 1
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_add_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a + 5
        expected = np.arange(10) + 5
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_radd_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = 5 + a
        expected = 5 + np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_sub_arrays(self, hpx_runtime):
        a = hpx.arange(10)
        b = hpx.ones(10) * 2
        result = a - b
        expected = np.arange(10) - 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_sub_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a - 3
        expected = np.arange(10) - 3
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_rsub_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = 10 - a
        expected = 10 - np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_mul_arrays(self, hpx_runtime):
        a = hpx.arange(10)
        b = hpx.arange(10)
        result = a * b
        expected = np.arange(10) * np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_mul_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a * 3
        expected = np.arange(10) * 3
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_div_arrays(self, hpx_runtime):
        a = hpx.arange(1, 11)  # 1-10 to avoid div by 0
        b = hpx.ones(10) * 2
        result = a / b
        expected = np.arange(1, 11) / 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_div_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a / 2
        expected = np.arange(10) / 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_pow(self, hpx_runtime):
        a = hpx.arange(5)
        result = a ** 2
        expected = np.arange(5) ** 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestUnaryOperators:
    """Test unary operators."""

    def test_neg(self, hpx_runtime):
        a = hpx.arange(10)
        result = -a
        expected = -np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_pos(self, hpx_runtime):
        a = hpx.arange(10)
        result = +a
        expected = +np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_abs(self, hpx_runtime):
        a = hpx.array([-3, -2, -1, 0, 1, 2, 3])
        result = abs(a)
        expected = np.abs([-3, -2, -1, 0, 1, 2, 3])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestComparisonOperators:
    """Test comparison operators."""

    def test_eq(self, hpx_runtime):
        a = hpx.arange(5)
        result = a == 2
        expected = np.arange(5) == 2
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_ne(self, hpx_runtime):
        a = hpx.arange(5)
        result = a != 2
        expected = np.arange(5) != 2
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_lt(self, hpx_runtime):
        a = hpx.arange(5)
        result = a < 3
        expected = np.arange(5) < 3
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_le(self, hpx_runtime):
        a = hpx.arange(5)
        result = a <= 3
        expected = np.arange(5) <= 3
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_gt(self, hpx_runtime):
        a = hpx.arange(5)
        result = a > 2
        expected = np.arange(5) > 2
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_ge(self, hpx_runtime):
        a = hpx.arange(5)
        result = a >= 2
        expected = np.arange(5) >= 2
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestOperatorChaining:
    """Test chained operations."""

    def test_chain_arithmetic(self, hpx_runtime):
        a = hpx.arange(10)
        result = (a + 1) * 2 - 3
        expected = (np.arange(10) + 1) * 2 - 3
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_expression_evaluation(self, hpx_runtime):
        x = hpx.arange(1, 6)  # [1, 2, 3, 4, 5]
        result = x ** 2 + 2 * x + 1  # (x+1)^2
        expected = np.arange(1, 6) ** 2 + 2 * np.arange(1, 6) + 1
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)
