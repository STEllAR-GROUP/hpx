# HPXPy GPU Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy GPU support."""

import numpy as np
import pytest


class TestGPUAvailability:
    """Test GPU detection and availability."""

    def test_is_available_returns_bool(self, hpx_runtime):
        """is_available should return a boolean."""
        result = hpx_runtime.gpu.is_available()
        assert isinstance(result, bool)

    def test_device_count_returns_int(self, hpx_runtime):
        """device_count should return an integer."""
        result = hpx_runtime.gpu.device_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_devices_returns_list(self, hpx_runtime):
        """get_devices should return a list."""
        result = hpx_runtime.gpu.get_devices()
        assert isinstance(result, list)


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").gpu.is_available(),
    reason="CUDA not available"
)
class TestGPUDeviceInfo:
    """Test GPU device information (requires CUDA)."""

    def test_get_device_returns_device(self, hpx_runtime):
        """get_device should return device info."""
        dev = hpx_runtime.gpu.get_device(0)
        assert hasattr(dev, 'id')
        assert hasattr(dev, 'name')
        assert hasattr(dev, 'total_memory')

    def test_device_has_compute_capability(self, hpx_runtime):
        """Device should report compute capability."""
        dev = hpx_runtime.gpu.get_device(0)
        cc = dev.compute_capability()
        assert isinstance(cc, str)
        assert '.' in cc

    def test_device_memory_methods(self, hpx_runtime):
        """Device should report memory in GB."""
        dev = hpx_runtime.gpu.get_device(0)
        total_gb = dev.total_memory_gb()
        assert isinstance(total_gb, float)
        assert total_gb > 0

    def test_current_device(self, hpx_runtime):
        """current_device should return valid device ID."""
        dev_id = hpx_runtime.gpu.current_device()
        assert isinstance(dev_id, int)
        assert dev_id >= 0

    def test_memory_info_returns_tuple(self, hpx_runtime):
        """memory_info should return (free, total) tuple."""
        free, total = hpx_runtime.gpu.memory_info(0)
        assert isinstance(free, int)
        assert isinstance(total, int)
        assert free <= total


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").gpu.is_available(),
    reason="CUDA not available"
)
class TestGPUArrayCreation:
    """Test GPU array creation (requires CUDA)."""

    def test_gpu_zeros(self, hpx_runtime):
        """gpu.zeros should create zero-filled array."""
        arr = hpx_runtime.gpu.zeros([100])
        assert arr.shape == [100]
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_gpu_ones(self, hpx_runtime):
        """gpu.ones should create one-filled array."""
        arr = hpx_runtime.gpu.ones([100])
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(100))

    def test_gpu_full(self, hpx_runtime):
        """gpu.full should create array filled with value."""
        arr = hpx_runtime.gpu.full([50], 3.14)
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full(50, 3.14))

    def test_gpu_from_numpy(self, hpx_runtime):
        """gpu.from_numpy should copy numpy array to GPU."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gpu_arr = hpx_runtime.gpu.from_numpy(np_arr)
        np.testing.assert_array_equal(gpu_arr.to_numpy(), np_arr)


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").gpu.is_available(),
    reason="CUDA not available"
)
class TestGPUArrayProperties:
    """Test GPU array properties (requires CUDA)."""

    def test_shape_property(self, hpx_runtime):
        """GPU array should have shape property."""
        arr = hpx_runtime.gpu.zeros([10, 20])
        assert arr.shape == [10, 20]

    def test_size_property(self, hpx_runtime):
        """GPU array should have size property."""
        arr = hpx_runtime.gpu.zeros([10, 20])
        assert arr.size == 200

    def test_ndim_property(self, hpx_runtime):
        """GPU array should have ndim property."""
        arr = hpx_runtime.gpu.zeros([10, 20, 30])
        assert arr.ndim == 3

    def test_device_property(self, hpx_runtime):
        """GPU array should report its device."""
        arr = hpx_runtime.gpu.zeros([100], device=0)
        assert arr.device == 0


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").gpu.is_available(),
    reason="CUDA not available"
)
class TestGPUArrayMethods:
    """Test GPU array methods (requires CUDA)."""

    def test_fill(self, hpx_runtime):
        """fill should set all elements."""
        arr = hpx_runtime.gpu.zeros([100])
        arr.fill(42.0)
        np.testing.assert_array_equal(arr.to_numpy(), np.full(100, 42.0))

    def test_to_numpy(self, hpx_runtime):
        """to_numpy should return numpy array."""
        arr = hpx_runtime.gpu.ones([50])
        np_arr = arr.to_numpy()
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (50,)


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").gpu.is_available(),
    reason="CUDA not available"
)
class TestGPUOperations:
    """Test GPU operations (requires CUDA)."""

    def test_gpu_sum(self, hpx_runtime):
        """gpu.sum should sum all elements."""
        arr = hpx_runtime.gpu.ones([100])
        result = hpx_runtime.gpu.sum(arr)
        assert result == 100.0


class TestGPUStubsWhenUnavailable:
    """Test GPU stubs work when CUDA unavailable."""

    def test_is_available_without_cuda(self, hpx_runtime):
        """is_available should work even without CUDA."""
        # This should not raise
        result = hpx_runtime.gpu.is_available()
        assert isinstance(result, bool)

    def test_device_count_without_cuda(self, hpx_runtime):
        """device_count should return 0 if no CUDA."""
        if not hpx_runtime.gpu.is_available():
            assert hpx_runtime.gpu.device_count() == 0

    def test_get_devices_without_cuda(self, hpx_runtime):
        """get_devices should return empty list if no CUDA."""
        if not hpx_runtime.gpu.is_available():
            assert hpx_runtime.gpu.get_devices() == []
