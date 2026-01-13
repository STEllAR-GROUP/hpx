# Phase 3: Distribution tests
#
# SPDX-License-Identifier: BSL-1.0

import pytest

import hpxpy as hpx


class TestDistributionModule:
    """Test distribution module availability."""

    def test_distribution_module_exists(self, hpx_runtime):
        """Distribution module should be accessible."""
        assert hasattr(hpx, 'distribution')

    def test_distribution_policies_exist(self, hpx_runtime):
        """Distribution policies should be available."""
        assert hasattr(hpx.distribution, 'none')
        assert hasattr(hpx.distribution, 'block')
        assert hasattr(hpx.distribution, 'cyclic')
        assert hasattr(hpx.distribution, 'local')

    def test_locality_introspection(self, hpx_runtime):
        """Locality introspection functions should work."""
        locality_id = hpx.distribution.get_locality_id()
        num_localities = hpx.distribution.get_num_localities()

        # In single-locality mode, these should be fixed values
        assert isinstance(locality_id, int)
        assert isinstance(num_localities, int)
        assert locality_id >= 0
        assert num_localities >= 1


class TestDistributionPolicy:
    """Test distribution policy enum."""

    def test_policy_values(self, hpx_runtime):
        """Distribution policy enum values should be correct."""
        assert hpx.distribution.none == hpx.distribution.DistributionPolicy.none
        assert hpx.distribution.block == hpx.distribution.DistributionPolicy.block
        assert hpx.distribution.cyclic == hpx.distribution.DistributionPolicy.cyclic

    def test_local_alias(self, hpx_runtime):
        """'local' should be an alias for 'none'."""
        assert hpx.distribution.local == hpx.distribution.none


class TestLocalArrayWithDistributionContext:
    """Test that existing arrays still work with distribution module loaded."""

    def test_zeros_still_works(self, hpx_runtime):
        """zeros() should still work normally."""
        arr = hpx.zeros((10,))
        assert arr.shape == (10,)
        assert arr.size == 10

    def test_operations_still_work(self, hpx_runtime):
        """Operations should still work normally."""
        a = hpx.arange(10)
        b = hpx.arange(10)
        c = a + b
        assert c.size == 10

    def test_to_numpy_still_works(self, hpx_runtime):
        """to_numpy() should still work."""
        import numpy as np
        arr = hpx.array([1, 2, 3, 4, 5])
        np_arr = arr.to_numpy()
        np.testing.assert_array_equal(np_arr, [1, 2, 3, 4, 5])
