# HPXPy SYCL Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy SYCL support via HPX sycl_executor."""

import numpy as np
import pytest


class TestSYCLAvailability:
    """Test SYCL detection and availability."""

    def test_is_available_returns_bool(self, hpx_runtime):
        """is_available should return a boolean."""
        result = hpx_runtime.sycl.is_available()
        assert isinstance(result, bool)

    def test_device_count_returns_int(self, hpx_runtime):
        """device_count should return an integer."""
        result = hpx_runtime.sycl.device_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_devices_returns_list(self, hpx_runtime):
        """get_devices should return a list."""
        result = hpx_runtime.sycl.get_devices()
        assert isinstance(result, list)


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").sycl.is_available(),
    reason="SYCL not available"
)
class TestSYCLDeviceInfo:
    """Test SYCL device information (requires SYCL)."""

    def test_get_device_returns_device(self, hpx_runtime):
        """get_device should return device info."""
        dev = hpx_runtime.sycl.get_device(0)
        assert hasattr(dev, 'id')
        assert hasattr(dev, 'name')
        assert hasattr(dev, 'global_mem_size')

    def test_device_has_backend_info(self, hpx_runtime):
        """Device should report backend type."""
        dev = hpx_runtime.sycl.get_device(0)
        assert hasattr(dev, 'backend')
        assert isinstance(dev.backend, str)

    def test_device_memory_methods(self, hpx_runtime):
        """Device should report memory in GB."""
        dev = hpx_runtime.sycl.get_device(0)
        total_gb = dev.global_mem_size_gb()
        assert isinstance(total_gb, float)
        assert total_gb > 0

    def test_device_properties(self, hpx_runtime):
        """Device should have standard properties."""
        dev = hpx_runtime.sycl.get_device(0)
        assert hasattr(dev, 'is_gpu')
        assert hasattr(dev, 'max_compute_units')
        assert hasattr(dev, 'max_work_group_size')


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").sycl.is_available(),
    reason="SYCL not available"
)
class TestSYCLArrayCreation:
    """Test SYCL array creation (requires SYCL)."""

    def test_sycl_zeros(self, hpx_runtime):
        """sycl.zeros should create zero-filled array."""
        arr = hpx_runtime.sycl.zeros([100])
        assert arr.shape == [100]
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_sycl_ones(self, hpx_runtime):
        """sycl.ones should create one-filled array."""
        arr = hpx_runtime.sycl.ones([100])
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(100))

    def test_sycl_full(self, hpx_runtime):
        """sycl.full should create array filled with value."""
        arr = hpx_runtime.sycl.full([50], 3.14)
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full(50, 3.14))

    def test_sycl_from_numpy(self, hpx_runtime):
        """sycl.from_numpy should copy numpy array to SYCL device."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sycl_arr = hpx_runtime.sycl.from_numpy(np_arr)
        np.testing.assert_array_equal(sycl_arr.to_numpy(), np_arr)


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").sycl.is_available(),
    reason="SYCL not available"
)
class TestSYCLArrayProperties:
    """Test SYCL array properties (requires SYCL)."""

    def test_shape_property(self, hpx_runtime):
        """SYCL array should have shape property."""
        arr = hpx_runtime.sycl.zeros([10, 20])
        assert arr.shape == [10, 20]

    def test_size_property(self, hpx_runtime):
        """SYCL array should have size property."""
        arr = hpx_runtime.sycl.zeros([10, 20])
        assert arr.size == 200

    def test_ndim_property(self, hpx_runtime):
        """SYCL array should have ndim property."""
        arr = hpx_runtime.sycl.zeros([10, 20, 30])
        assert arr.ndim == 3

    def test_device_property(self, hpx_runtime):
        """SYCL array should report its device."""
        arr = hpx_runtime.sycl.zeros([100], device=0)
        assert arr.device == 0


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").sycl.is_available(),
    reason="SYCL not available"
)
class TestSYCLArrayMethods:
    """Test SYCL array methods (requires SYCL)."""

    def test_fill(self, hpx_runtime):
        """fill should set all elements."""
        arr = hpx_runtime.sycl.zeros([100])
        arr.fill(42.0)
        np.testing.assert_array_equal(arr.to_numpy(), np.full(100, 42.0))

    def test_to_numpy(self, hpx_runtime):
        """to_numpy should return numpy array."""
        arr = hpx_runtime.sycl.ones([50])
        np_arr = arr.to_numpy()
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (50,)


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").sycl.is_available(),
    reason="SYCL not available"
)
class TestSYCLOperations:
    """Test SYCL operations (requires SYCL)."""

    def test_sycl_sum(self, hpx_runtime):
        """sycl.sum should sum all elements."""
        arr = hpx_runtime.sycl.ones([100])
        result = hpx_runtime.sycl.sum(arr)
        assert result == 100.0


class TestSYCLStubsWhenUnavailable:
    """Test SYCL stubs work when SYCL unavailable."""

    def test_is_available_without_sycl(self, hpx_runtime):
        """is_available should work even without SYCL."""
        result = hpx_runtime.sycl.is_available()
        assert isinstance(result, bool)

    def test_device_count_without_sycl(self, hpx_runtime):
        """device_count should return 0 if no SYCL."""
        if not hpx_runtime.sycl.is_available():
            assert hpx_runtime.sycl.device_count() == 0

    def test_get_devices_without_sycl(self, hpx_runtime):
        """get_devices should return empty list if no SYCL."""
        if not hpx_runtime.sycl.is_available():
            assert hpx_runtime.sycl.get_devices() == []


class TestSYCLAsyncOperations:
    """Test SYCL async operation infrastructure."""

    def test_enable_async_no_error(self, hpx_runtime):
        """enable_async should not raise even without SYCL."""
        hpx_runtime.sycl.enable_async()
        hpx_runtime.sycl.disable_async()

    def test_disable_async_no_error(self, hpx_runtime):
        """disable_async should not raise even without SYCL."""
        hpx_runtime.sycl.disable_async()

    def test_is_async_enabled_returns_bool(self, hpx_runtime):
        """is_async_enabled should return a boolean."""
        result = hpx_runtime.sycl.is_async_enabled()
        assert isinstance(result, bool)

    def test_is_async_enabled_without_sycl(self, hpx_runtime):
        """is_async_enabled should return False without SYCL."""
        if not hpx_runtime.sycl.is_available():
            hpx_runtime.sycl.enable_async()
            assert hpx_runtime.sycl.is_async_enabled() == False

    def test_async_context_manager(self, hpx_runtime):
        """AsyncContext should enable/disable async operations."""
        with hpx_runtime.sycl.AsyncContext():
            pass


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").sycl.is_available(),
    reason="SYCL not available"
)
class TestSYCLAsyncArrayOperations:
    """Test SYCL async array operations (requires SYCL)."""

    def test_async_enabled_state(self, hpx_runtime):
        """enable_async should change the enabled state."""
        hpx_runtime.sycl.enable_async()
        assert hpx_runtime.sycl.is_async_enabled() == True
        hpx_runtime.sycl.disable_async()
        assert hpx_runtime.sycl.is_async_enabled() == False

    def test_async_from_numpy_requires_polling(self, hpx_runtime):
        """async_from_numpy should require enable_async first."""
        arr = hpx_runtime.sycl.zeros([100])
        np_data = np.ones(100)

        hpx_runtime.sycl.disable_async()

        with pytest.raises(RuntimeError, match="enable_async"):
            arr.async_from_numpy(np_data)

    def test_async_from_numpy_basic(self, hpx_runtime):
        """async_from_numpy should copy data asynchronously."""
        hpx_runtime.sycl.enable_async()
        try:
            arr = hpx_runtime.sycl.zeros([1000])
            np_data = np.arange(1000, dtype=np.float64)

            future = arr.async_from_numpy(np_data)

            assert hasattr(future, 'get')
            assert hasattr(future, 'wait')
            assert hasattr(future, 'is_ready')

            future.get()

            result = arr.to_numpy()
            np.testing.assert_array_equal(result, np_data)
        finally:
            hpx_runtime.sycl.disable_async()

    def test_async_context_with_operations(self, hpx_runtime):
        """AsyncContext should enable async operations."""
        arr = hpx_runtime.sycl.zeros([100])
        np_data = np.ones(100)

        with hpx_runtime.sycl.AsyncContext():
            future = arr.async_from_numpy(np_data)
            future.get()

        np.testing.assert_array_equal(arr.to_numpy(), np_data)
