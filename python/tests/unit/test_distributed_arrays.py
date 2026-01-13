# HPXPy Distributed Array Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy distributed arrays."""

import numpy as np
import pytest


class TestDistributedArrayCreation:
    """Test distributed array creation functions."""

    def test_distributed_zeros(self, hpx_runtime):
        """distributed_zeros should create zero-filled array."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.shape == [100]
        assert arr.size == 100
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_distributed_zeros_with_shape_tuple(self, hpx_runtime):
        """distributed_zeros should work with tuple shape."""
        arr = hpx_runtime.distributed_zeros((50, 2))
        assert arr.shape == [50, 2]
        assert arr.size == 100

    def test_distributed_ones(self, hpx_runtime):
        """distributed_ones should create one-filled array."""
        arr = hpx_runtime.distributed_ones([100])
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(100))

    def test_distributed_full(self, hpx_runtime):
        """distributed_full should create array with specified value."""
        arr = hpx_runtime.distributed_full([50], 3.14)
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full(50, 3.14))

    def test_distributed_from_numpy(self, hpx_runtime):
        """distributed_from_numpy should copy numpy array data."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = hpx_runtime.distributed_from_numpy(np_arr)
        np.testing.assert_array_equal(arr.to_numpy(), np_arr)


class TestDistributionPolicy:
    """Test distribution policy handling."""

    def test_default_policy_is_none(self, hpx_runtime):
        """Default distribution should be none."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.policy == hpx_runtime.DistributionPolicy.none

    def test_block_distribution(self, hpx_runtime):
        """Block distribution should be recognized."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        assert arr.policy == hpx_runtime.DistributionPolicy.block

    def test_cyclic_distribution(self, hpx_runtime):
        """Cyclic distribution should be recognized."""
        arr = hpx_runtime.distributed_zeros([100], distribution='cyclic')
        assert arr.policy == hpx_runtime.DistributionPolicy.cyclic

    def test_none_distribution_string(self, hpx_runtime):
        """String 'none' should work as distribution."""
        arr = hpx_runtime.distributed_zeros([100], distribution='none')
        assert arr.policy == hpx_runtime.DistributionPolicy.none


class TestDistributedArrayProperties:
    """Test distributed array properties."""

    def test_shape_property(self, hpx_runtime):
        """shape property should return correct shape."""
        arr = hpx_runtime.distributed_zeros([10, 20])
        assert arr.shape == [10, 20]

    def test_size_property(self, hpx_runtime):
        """size property should return total elements."""
        arr = hpx_runtime.distributed_zeros([10, 20])
        assert arr.size == 200

    def test_ndim_property(self, hpx_runtime):
        """ndim property should return number of dimensions."""
        arr = hpx_runtime.distributed_zeros([10, 20, 30])
        assert arr.ndim == 3

    def test_num_partitions_property(self, hpx_runtime):
        """num_partitions should be at least 1."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.num_partitions >= 1

    def test_locality_id_property(self, hpx_runtime):
        """locality_id should be valid."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.locality_id >= 0


class TestDistributedArrayMethods:
    """Test distributed array methods."""

    def test_fill(self, hpx_runtime):
        """fill should set all elements to value."""
        arr = hpx_runtime.distributed_zeros([100])
        arr.fill(42.0)
        np.testing.assert_array_equal(arr.to_numpy(), np.full(100, 42.0))

    def test_to_numpy(self, hpx_runtime):
        """to_numpy should return correct numpy array."""
        arr = hpx_runtime.distributed_ones([50])
        np_arr = arr.to_numpy()
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (50,)
        np.testing.assert_array_equal(np_arr, np.ones(50))

    def test_is_distributed_single_locality(self, hpx_runtime):
        """is_distributed should be False on single locality."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        # In single-locality mode, arrays are not actually distributed
        assert arr.is_distributed() is False

    def test_get_distribution_info(self, hpx_runtime):
        """get_distribution_info should return DistributionInfo."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        info = arr.get_distribution_info()
        assert info.policy == hpx_runtime.DistributionPolicy.block
        assert info.num_partitions >= 1


class TestDistributedArrayRepr:
    """Test distributed array string representation."""

    def test_repr_contains_shape(self, hpx_runtime):
        """repr should contain shape information."""
        arr = hpx_runtime.distributed_zeros([100])
        repr_str = repr(arr)
        assert '100' in repr_str

    def test_repr_contains_distribution(self, hpx_runtime):
        """repr should contain distribution information."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        repr_str = repr(arr)
        assert 'block' in repr_str

    def test_repr_contains_partitions(self, hpx_runtime):
        """repr should contain partition information."""
        arr = hpx_runtime.distributed_zeros([100])
        repr_str = repr(arr)
        assert 'partitions' in repr_str
