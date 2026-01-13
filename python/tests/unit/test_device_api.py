# HPXPy Device API Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy transparent device selection API."""

import numpy as np
import pytest


class TestDeviceParameterCPU:
    """Test device parameter with CPU backend."""

    def test_zeros_default_device(self, hpx_runtime):
        """zeros() with no device should create CPU array."""
        arr = hpx_runtime.zeros(100)
        assert hasattr(arr, 'to_numpy')  # CPU ndarray has to_numpy
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_zeros_cpu_explicit(self, hpx_runtime):
        """zeros(device='cpu') should create CPU array."""
        arr = hpx_runtime.zeros(100, device='cpu')
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_ones_cpu(self, hpx_runtime):
        """ones(device='cpu') should create CPU array."""
        arr = hpx_runtime.ones((10, 10), device='cpu')
        np.testing.assert_array_equal(arr.to_numpy(), np.ones((10, 10)))

    def test_empty_cpu(self, hpx_runtime):
        """empty(device='cpu') should create CPU array."""
        arr = hpx_runtime.empty(50, device='cpu')
        assert arr.size == 50

    def test_arange_cpu(self, hpx_runtime):
        """arange(device='cpu') should create CPU array."""
        arr = hpx_runtime.arange(10, device='cpu')
        np.testing.assert_array_equal(arr.to_numpy(), np.arange(10))

    def test_linspace_cpu(self, hpx_runtime):
        """linspace(device='cpu') should create CPU array."""
        arr = hpx_runtime.linspace(0, 1, 11, device='cpu')
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.linspace(0, 1, 11))

    def test_full_cpu(self, hpx_runtime):
        """full(device='cpu') should create CPU array."""
        arr = hpx_runtime.full((5, 5), 3.14, device='cpu')
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full((5, 5), 3.14))

    def test_from_numpy_cpu(self, hpx_runtime):
        """from_numpy(device='cpu') should create CPU array."""
        np_arr = np.array([1.0, 2.0, 3.0])
        arr = hpx_runtime.from_numpy(np_arr, device='cpu')
        np.testing.assert_array_equal(arr.to_numpy(), np_arr)

    def test_array_cpu(self, hpx_runtime):
        """array(device='cpu') should create CPU array."""
        arr = hpx_runtime.array([1, 2, 3, 4, 5], device='cpu')
        np.testing.assert_array_equal(arr.to_numpy(), np.array([1, 2, 3, 4, 5]))


class TestDeviceParameterAuto:
    """Test device='auto' parameter."""

    def test_zeros_auto_without_cuda(self, hpx_runtime):
        """zeros(device='auto') should fallback to CPU when no GPU."""
        if not hpx_runtime.gpu.is_available():
            arr = hpx_runtime.zeros(100, device='auto')
            np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_ones_auto_without_cuda(self, hpx_runtime):
        """ones(device='auto') should fallback to CPU when no GPU."""
        if not hpx_runtime.gpu.is_available():
            arr = hpx_runtime.ones(100, device='auto')
            np.testing.assert_array_equal(arr.to_numpy(), np.ones(100))

    def test_arange_auto_without_cuda(self, hpx_runtime):
        """arange(device='auto') should fallback to CPU when no GPU."""
        if not hpx_runtime.gpu.is_available():
            arr = hpx_runtime.arange(100, device='auto')
            np.testing.assert_array_equal(arr.to_numpy(), np.arange(100))


class TestDeviceParameterInvalid:
    """Test invalid device parameter values."""

    def test_invalid_device_string(self, hpx_runtime):
        """Invalid device string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid device specification"):
            hpx_runtime.zeros(100, device='invalid')

    def test_gpu_unavailable_raises(self, hpx_runtime):
        """device='gpu' should raise RuntimeError when CUDA not available."""
        if not hpx_runtime.gpu.is_available():
            with pytest.raises(RuntimeError, match="CUDA GPU requested but CUDA is not available"):
                hpx_runtime.zeros(100, device='gpu')

    def test_sycl_unavailable_raises(self, hpx_runtime):
        """device='sycl' should raise RuntimeError when SYCL not available."""
        if not hpx_runtime.sycl.is_available():
            with pytest.raises(RuntimeError, match="SYCL requested but not available"):
                hpx_runtime.zeros(100, device='sycl')

    def test_gpu_device_id_unavailable(self, hpx_runtime):
        """device=<int> should raise RuntimeError when no GPU backend available."""
        if not hpx_runtime.gpu.is_available() and not hpx_runtime.sycl.is_available():
            with pytest.raises(RuntimeError, match="no GPU backend is available"):
                hpx_runtime.zeros(100, device=0)


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").gpu.is_available(),
    reason="CUDA not available"
)
class TestDeviceParameterGPU:
    """Test device parameter with GPU backend (requires CUDA)."""

    def test_zeros_gpu(self, hpx_runtime):
        """zeros(device='gpu') should create GPU array."""
        arr = hpx_runtime.zeros(100, device='gpu')
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_ones_gpu(self, hpx_runtime):
        """ones(device='gpu') should create GPU array."""
        arr = hpx_runtime.ones(100, device='gpu')
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(100))

    def test_full_gpu(self, hpx_runtime):
        """full(device='gpu') should create GPU array."""
        arr = hpx_runtime.full(100, 3.14, device='gpu')
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full(100, 3.14))

    def test_zeros_device_id(self, hpx_runtime):
        """zeros(device=0) should create GPU array on device 0."""
        arr = hpx_runtime.zeros(100, device=0)
        assert arr.device == 0
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_arange_gpu(self, hpx_runtime):
        """arange(device='gpu') should create GPU array."""
        arr = hpx_runtime.arange(100, device='gpu')
        np.testing.assert_array_equal(arr.to_numpy(), np.arange(100))

    def test_from_numpy_gpu(self, hpx_runtime):
        """from_numpy(device='gpu') should transfer to GPU."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = hpx_runtime.from_numpy(np_arr, device='gpu')
        np.testing.assert_array_equal(arr.to_numpy(), np_arr)

    def test_zeros_auto_with_cuda(self, hpx_runtime):
        """zeros(device='auto') should use GPU when available."""
        arr = hpx_runtime.zeros(100, device='auto')
        # Should be a GPU array when CUDA is available
        assert hasattr(arr, 'device')  # GPU arrays have device property


@pytest.mark.skipif(
    not pytest.importorskip("hpxpy").sycl.is_available(),
    reason="SYCL not available"
)
class TestDeviceParameterSYCL:
    """Test device parameter with SYCL backend (requires SYCL)."""

    def test_zeros_sycl(self, hpx_runtime):
        """zeros(device='sycl') should create SYCL array."""
        arr = hpx_runtime.zeros(100, device='sycl')
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_ones_sycl(self, hpx_runtime):
        """ones(device='sycl') should create SYCL array."""
        arr = hpx_runtime.ones(100, device='sycl')
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(100))

    def test_full_sycl(self, hpx_runtime):
        """full(device='sycl') should create SYCL array."""
        arr = hpx_runtime.full(100, 3.14, device='sycl')
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full(100, 3.14))

    def test_arange_sycl(self, hpx_runtime):
        """arange(device='sycl') should create SYCL array."""
        arr = hpx_runtime.arange(100, device='sycl')
        np.testing.assert_array_equal(arr.to_numpy(), np.arange(100))

    def test_from_numpy_sycl(self, hpx_runtime):
        """from_numpy(device='sycl') should transfer to SYCL device."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = hpx_runtime.from_numpy(np_arr, device='sycl')
        np.testing.assert_array_equal(arr.to_numpy(), np_arr)
