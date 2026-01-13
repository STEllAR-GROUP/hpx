# Phase 5: GPU Acceleration - What Was Done

## Summary

Phase 5 adds GPU acceleration to HPXPy using **HPX's hardware backend executors**.
All GPU backends use HPX's executor infrastructure for proper async operations, futures, and distribution.

**Status:** Complete (2026-01-13)

## Objectives

1. GPU detection and runtime query API
2. GPU array creation and transfers
3. Transparent device selection with hints
4. CUDA backend via `hpx::cuda::experimental::cuda_executor`
5. SYCL backend via `hpx::sycl::experimental::sycl_executor`

## Implemented Features

### Transparent Device Selection API

- [x] `device='cpu'` - create arrays on CPU (default)
- [x] `device='gpu'` or `device='cuda'` - create arrays on CUDA GPU
- [x] `device='sycl'` - create arrays on SYCL GPU (Intel/AMD/Apple Silicon)
- [x] `device='auto'` - auto-select: CUDA > SYCL > CPU
- [x] `device=<int>` - specific GPU device ID
- [x] All array creation functions support `device` parameter:
  - `zeros()`, `ones()`, `empty()`, `full()`
  - `arange()`, `linspace()`
  - `array()`, `from_numpy()`

### CUDA Backend (via HPX cuda_executor)

- [x] **GPU Detection API**
  - `gpu.is_available()` - check CUDA availability
  - `gpu.device_count()` - number of GPUs
  - `gpu.get_devices()` - list all GPU info
  - `gpu.get_device(id)` - get specific GPU info
  - `gpu.current_device()` - current GPU ID
  - `gpu.set_device(id)` - set current GPU
  - `gpu.synchronize()` - synchronize GPU
  - `gpu.memory_info()` - get free/total memory

- [x] **GPU Arrays**
  - `GPUArray<T>` template class in C++
  - `ArrayF64`, `ArrayF32`, `ArrayI64`, `ArrayI32` types
  - `gpu.zeros()`, `gpu.ones()`, `gpu.full()` creation
  - `gpu.from_numpy()` host-to-device transfer
  - `arr.to_numpy()` device-to-host transfer

- [x] **HPX CUDA Integration**
  - `GPUPollingManager` - RAII manager for HPX CUDA event polling
  - `GPUExecutorManager` - Per-device HPX CUDA executor pool
  - `PyFuture<T>` - Wrapper to expose hpx::future to Python with GIL release
  - `gpu.enable_async()` - Enable HPX CUDA polling for async ops
  - `gpu.disable_async()` - Disable polling
  - `gpu.is_async_enabled()` - Check polling state
  - `AsyncContext` - Python context manager for async operations
  - `arr.async_from_numpy()` - Async H2D transfer returning Future

### SYCL Backend (via HPX sycl_executor)

- [x] **SYCL Detection API**
  - `sycl.is_available()` - check SYCL availability
  - `sycl.device_count()` - number of SYCL GPUs
  - `sycl.get_devices()` - list all SYCL device info
  - `sycl.get_device(id)` - get specific device info

- [x] **SYCL Device Information**
  - Device ID, name, vendor
  - Global/local memory size
  - Max compute units, work group size
  - Backend detection (CUDA, HIP, Level-Zero, Metal, OpenCL)

- [x] **SYCL Arrays**
  - `SYCLArray<T>` template class using USM (Unified Shared Memory)
  - `ArrayF64`, `ArrayF32`, `ArrayI64`, `ArrayI32` types
  - `sycl.zeros()`, `sycl.ones()`, `sycl.full()` creation
  - `sycl.from_numpy()` host-to-device transfer
  - `arr.to_numpy()` device-to-host transfer

- [x] **HPX SYCL Integration**
  - `SYCLPollingManager` - RAII manager for HPX SYCL event polling
  - `SYCLExecutorManager` - Per-device HPX SYCL executor pool
  - `PySYCLFuture<T>` - Wrapper to expose hpx::future to Python
  - `sycl.enable_async()` - Enable HPX SYCL polling
  - `sycl.disable_async()` - Disable polling
  - `AsyncContext` - Python context manager for async operations
  - `arr.async_from_numpy()` - Async transfer returning Future

## Files Created/Modified

| File | Description |
|------|-------------|
| `python/src/bindings/gpu_bindings.cpp` | CUDA bindings using HPX cuda_executor |
| `python/src/bindings/sycl_bindings.cpp` | SYCL bindings using HPX sycl_executor |
| `python/src/bindings/core_module.cpp` | Register GPU and SYCL bindings |
| `python/CMakeLists.txt` | CUDA/SYCL config, HPX component linking |
| `python/hpxpy/gpu.py` | Python CUDA module + async operations |
| `python/hpxpy/sycl.py` | Python SYCL module + async operations |
| `python/hpxpy/__init__.py` | Export GPU/SYCL modules, device selection |
| `python/tests/unit/test_gpu.py` | CUDA unit tests + async tests |
| `python/tests/unit/test_sycl.py` | SYCL unit tests + async tests |
| `python/tests/unit/test_device_api.py` | Transparent device API tests |

## API Usage

### Transparent Device Selection (Recommended)

```python
import hpxpy as hpx

# Create arrays with device hints - transparent API
arr = hpx.zeros(1000000)              # Default: CPU
arr = hpx.zeros(1000000, device='cpu')  # Explicit CPU
arr = hpx.zeros(1000000, device='gpu')  # CUDA GPU (error if unavailable)
arr = hpx.zeros(1000000, device='sycl') # SYCL GPU (Intel/AMD/Apple)
arr = hpx.zeros(1000000, device='auto') # Best available: CUDA > SYCL > CPU
arr = hpx.zeros(1000000, device=0)      # Specific GPU device ID

# All array creation functions support device parameter
arr = hpx.ones((100, 100), device='auto')
arr = hpx.full(1000, 3.14, device='auto')
arr = hpx.arange(1000000, device='auto')
arr = hpx.linspace(0, 1, 100, device='auto')
arr = hpx.from_numpy(np_arr, device='auto')

# Arrays work the same regardless of device
result = arr.to_numpy()  # Transfer back to CPU numpy array
```

### Explicit CUDA API

```python
import hpxpy as hpx

# Check CUDA availability
if hpx.gpu.is_available():
    print(f"Found {hpx.gpu.device_count()} CUDA GPU(s)")

# List all CUDA devices
for dev in hpx.gpu.get_devices():
    print(f"{dev.name}: {dev.total_memory_gb():.1f} GB")
    print(f"  Compute capability: {dev.compute_capability()}")

# CUDA array creation
arr = hpx.gpu.zeros([1000, 1000])
arr = hpx.gpu.from_numpy(np_arr)
result = arr.to_numpy()
```

### Explicit SYCL API

```python
import hpxpy as hpx

# Check SYCL availability
if hpx.sycl.is_available():
    print(f"Found {hpx.sycl.device_count()} SYCL GPU(s)")

# List all SYCL devices
for dev in hpx.sycl.get_devices():
    print(f"{dev.name} ({dev.backend}): {dev.global_mem_size_gb():.1f} GB")

# SYCL array creation
arr = hpx.sycl.zeros([1000, 1000])
arr = hpx.sycl.from_numpy(np_arr)
result = arr.to_numpy()
```

### Async Operations (HPX Integration)

```python
import hpxpy as hpx
import numpy as np

hpx.init()

# Enable async operations (starts HPX polling)
hpx.gpu.enable_async()  # For CUDA
# or
hpx.sycl.enable_async()  # For SYCL

# Create GPU array
arr = hpx.gpu.zeros([1000000])
data = np.random.rand(1000000)

# Async copy - returns immediately
future = arr.async_from_numpy(data)

# Do other work while transfer happens
print("Transfer in progress...")

# Wait for completion
future.get()

# Using context manager
with hpx.gpu.AsyncContext():
    f1 = arr1.async_from_numpy(data1)
    f2 = arr2.async_from_numpy(data2)
    f1.get()
    f2.get()

hpx.gpu.disable_async()
hpx.finalize()
```

## Build Configuration

Enable CUDA support:
```bash
cmake -DHPXPY_WITH_CUDA=ON ..
```

Enable SYCL support:
```bash
cmake -DHPXPY_WITH_SYCL=ON ..
```

Enable both:
```bash
cmake -DHPXPY_WITH_CUDA=ON -DHPXPY_WITH_SYCL=ON ..
```

## HPX Backend Architecture

All GPU backends use HPX's executor infrastructure:

```
Python API (hpx.zeros(..., device='gpu'))
         |
         v
+----------------------------------+
| Transparent Device Selection     |
| - Resolves device to backend     |
| - CUDA > SYCL > CPU priority     |
+----------------------------------+
         |
    +----+----+
    |         |
    v         v
+-------+  +-------+
| CUDA  |  | SYCL  |
+-------+  +-------+
    |         |
    v         v
+----------------------------------+  +----------------------------------+
| hpx::cuda::experimental::        |  | hpx::sycl::experimental::        |
| cuda_executor                    |  | sycl_executor                    |
| - GPUPollingManager              |  | - SYCLPollingManager             |
| - GPUExecutorManager             |  | - SYCLExecutorManager            |
| - PyFuture<T>                    |  | - PySYCLFuture<T>                |
+----------------------------------+  +----------------------------------+
         |                                    |
         v                                    v
+----------------------------------+  +----------------------------------+
| CUDA Runtime                     |  | SYCL Runtime (AdaptiveCpp/oneAPI)|
| - NVIDIA GPUs                    |  | - Intel GPUs (Level-Zero)        |
+----------------------------------+  | - AMD GPUs (HIP)                 |
                                      | - Apple Silicon (Metal)          |
                                      | - NVIDIA GPUs (CUDA backend)     |
                                      +----------------------------------+
```

## SYCL Backend Support

SYCL provides cross-platform GPU support through various backends:

| Backend | Platform | SYCL Implementation |
|---------|----------|---------------------|
| Level-Zero | Intel GPUs | Intel oneAPI |
| HIP | AMD GPUs | AdaptiveCpp |
| CUDA | NVIDIA GPUs | AdaptiveCpp, Intel oneAPI |
| Metal | Apple Silicon | AdaptiveCpp (experimental) |
| OpenCL | Various | Intel oneAPI, AdaptiveCpp |

### Apple Silicon Support

For Apple Silicon (M1/M2/M3), SYCL support is available through AdaptiveCpp's Metal backend:

1. Build AdaptiveCpp with Metal backend (experimental)
2. Build HPX with `HPX_WITH_SYCL=ON`
3. Build HPXPy with `HPXPY_WITH_SYCL=ON`

This provides true HPX-native GPU support on Apple Silicon with:
- HPX futures and async execution
- HPX distribution capabilities
- Consistent API across all platforms

## Implementation Notes

1. **HPX-Only Backends**: All GPU backends use HPX's executor infrastructure.
   No non-HPX backends (like MLX) are included.

2. **Polling Managers**: Both CUDA and SYCL use RAII polling managers
   (`enable_user_polling`) to ensure futures resolve correctly.

3. **Executor Managers**: Per-device executor pools are maintained to
   avoid recreating executors for each operation.

4. **GIL Handling**: `PyFuture<T>` releases the Python GIL when waiting
   for GPU operations, allowing other Python threads to run.

5. **Stub Mode**: When CUDA/SYCL is not available, modules provide stubs
   that return sensible defaults (False, 0, empty lists).

6. **Future Work**:
   - Add thrust-based GPU reductions for CUDA
   - Add SYCL-native reductions using parallel STL
   - Implement GPU transforms with custom kernels
   - Add multi-GPU support with HPX distribution
