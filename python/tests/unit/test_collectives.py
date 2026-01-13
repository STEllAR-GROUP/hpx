# HPXPy Collective Operations Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy collective operations."""

import numpy as np
import pytest


class TestCollectivesModule:
    """Test collectives module availability."""

    def test_collectives_module_exists(self, hpx_runtime):
        """Collectives module should be accessible."""
        assert hasattr(hpx_runtime, 'collectives')

    def test_collective_functions_exist(self, hpx_runtime):
        """All collective functions should be accessible."""
        assert hasattr(hpx_runtime, 'all_reduce')
        assert hasattr(hpx_runtime, 'broadcast')
        assert hasattr(hpx_runtime, 'gather')
        assert hasattr(hpx_runtime, 'scatter')
        assert hasattr(hpx_runtime, 'barrier')

    def test_locality_functions(self, hpx_runtime):
        """Locality introspection should work."""
        num_localities = hpx_runtime.collectives.get_num_localities()
        locality_id = hpx_runtime.collectives.get_locality_id()

        assert isinstance(num_localities, int)
        assert isinstance(locality_id, int)
        assert num_localities >= 1
        assert 0 <= locality_id < num_localities


class TestAllReduce:
    """Test all_reduce operation."""

    def test_all_reduce_sum(self, hpx_runtime):
        """all_reduce with sum should return input in single-locality mode."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.all_reduce(arr, op='sum')

        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_all_reduce_default_sum(self, hpx_runtime):
        """all_reduce default operation should be sum."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.all_reduce(arr)

        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_all_reduce_prod(self, hpx_runtime):
        """all_reduce with prod should work."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.all_reduce(arr, op='prod')

        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_all_reduce_min(self, hpx_runtime):
        """all_reduce with min should work."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.all_reduce(arr, op='min')

        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_all_reduce_max(self, hpx_runtime):
        """all_reduce with max should work."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.all_reduce(arr, op='max')

        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_all_reduce_invalid_op(self, hpx_runtime):
        """all_reduce with invalid op should raise ValueError."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError, match="Unknown reduction operation"):
            hpx_runtime.all_reduce(arr, op='invalid')


class TestBroadcast:
    """Test broadcast operation."""

    def test_broadcast_default_root(self, hpx_runtime):
        """broadcast should return copy in single-locality mode."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.broadcast(arr)

        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_broadcast_with_root(self, hpx_runtime):
        """broadcast with explicit root should work."""
        arr = hpx_runtime.from_numpy(np.array([4.0, 5.0, 6.0]))
        result = hpx_runtime.broadcast(arr, root=0)

        np.testing.assert_array_equal(result.to_numpy(), [4.0, 5.0, 6.0])


class TestGather:
    """Test gather operation."""

    def test_gather_returns_list(self, hpx_runtime):
        """gather should return a list."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.gather(arr)

        assert isinstance(result, list)

    def test_gather_single_locality(self, hpx_runtime):
        """gather in single-locality mode should return list with one element."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.gather(arr, root=0)

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])


class TestScatter:
    """Test scatter operation."""

    def test_scatter_returns_array(self, hpx_runtime):
        """scatter should return an ndarray."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.scatter(arr)

        assert hasattr(result, 'to_numpy')

    def test_scatter_single_locality(self, hpx_runtime):
        """scatter in single-locality mode should return copy."""
        arr = hpx_runtime.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = hpx_runtime.scatter(arr, root=0)

        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0])


class TestBarrier:
    """Test barrier synchronization."""

    def test_barrier_no_args(self, hpx_runtime):
        """barrier should work with default name."""
        # Should not raise
        hpx_runtime.barrier()

    def test_barrier_with_name(self, hpx_runtime):
        """barrier should work with custom name."""
        # Should not raise
        hpx_runtime.barrier("test_barrier")
