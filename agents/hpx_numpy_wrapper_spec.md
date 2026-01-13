# HPX NumPy-Like Python Wrapper Specification

## Project Overview

**Project Name:** HPXPy (or hpx-numpy)

**Goal:** Create Python bindings for HPX that expose a NumPy-compatible interface for high-performance distributed computing, enabling Python users to leverage HPX's parallel and distributed execution capabilities with familiar array semantics.

**License:** BSL-1.0 (matching HPX)

---

## Motivation

HPX provides a powerful C++ runtime for parallel and distributed computing with:
- 100+ parallel algorithms with execution policies
- Asynchronous task execution with futures
- Distributed containers (`partitioned_vector`)
- Collective operations (all_reduce, broadcast, gather, etc.)
- Support for scaling from laptops to exascale systems

However, HPX currently has **no Python bindings**. This project will bridge that gap, allowing Python's scientific computing ecosystem to leverage HPX's performance.

---

## Architecture

### Binding Technology

**Primary:** pybind11
- Modern C++17/20 support (required for HPX)
- Excellent NumPy interoperability via `pybind11::array`
- Low overhead for performance-critical code
- Support for asynchronous operations and futures

**Alternative:** nanobind (lighter weight pybind11 fork) for reduced binary size

### Module Structure

```
hpxpy/
├── __init__.py           # Main package initialization
├── _core.cpython-*.so    # Core C++ bindings
├── array.py              # DistributedArray class (Python layer)
├── algorithms.py         # NumPy-style algorithm wrappers
├── execution.py          # Execution policies
├── collectives.py        # Distributed collective operations
├── futures.py            # Future/async utilities
└── runtime.py            # HPX runtime management
```

### C++ Binding Layers

```
src/
├── bindings/
│   ├── core_module.cpp       # Main pybind11 module definition
│   ├── runtime_bindings.cpp  # hpx::start(), hpx::stop(), config
│   ├── future_bindings.cpp   # hpx::future<T>, async(), dataflow()
│   ├── array_bindings.cpp    # partitioned_vector bindings
│   ├── algorithm_bindings.cpp # Parallel algorithm bindings
│   ├── execution_bindings.cpp # Execution policies
│   └── collective_bindings.cpp # Distributed collectives
├── types/
│   ├── distributed_array.hpp  # Wrapper around partitioned_vector
│   └── numpy_compat.hpp       # NumPy dtype <-> C++ type mapping
└── util/
    ├── gil_release.hpp        # GIL management for parallelism
    └── error_handling.hpp     # Exception translation
```

---

## Core API Design

### 1. Runtime Management

```python
import hpxpy as hpx

# Initialize HPX runtime (required before any operations)
hpx.init()  # Auto-detect threads/localities
hpx.init(num_threads=8)  # Specify thread count
hpx.init(config=["--hpx:threads=8", "--hpx:localities=4"])

# Runtime info
hpx.num_localities()  # Number of distributed nodes
hpx.locality_id()     # Current node ID
hpx.num_threads()     # Threads per locality

# Cleanup
hpx.finalize()

# Context manager (preferred)
with hpx.runtime(num_threads=8):
    # HPX operations here
    pass
```

### 2. Distributed Array (Primary Data Structure)

```python
import hpxpy as hpx
import numpy as np

# Creation (similar to NumPy)
a = hpx.array([1, 2, 3, 4, 5], dtype=np.float64)
b = hpx.zeros((1000, 1000), dtype=np.float64)
c = hpx.ones((1000,), dtype=np.int32)
d = hpx.empty((500, 500))
e = hpx.arange(0, 100, 2)
f = hpx.linspace(0, 1, 100)

# From NumPy (copy or view)
np_arr = np.random.randn(1000, 1000)
h = hpx.from_numpy(np_arr, copy=False)  # Zero-copy if possible

# Distribution control
distributed = hpx.array(
    np.random.randn(10000, 10000),
    distribution=hpx.distribution.block,  # Block distribution
    num_partitions=hpx.num_localities()   # One partition per node
)

# Properties
distributed.shape      # (10000, 10000)
distributed.dtype      # dtype('float64')
distributed.size       # 100000000
distributed.partitions # Number of partitions
distributed.locality   # Home locality (for non-distributed)

# Conversion
local_np = distributed.to_numpy()  # Gather to single node
local_np = distributed.numpy()     # Alias
```

#### Distribution Syntax Options (All Compatible)

HPXPy supports multiple ways to specify distribution, providing different levels of abstraction. These are **not mutually exclusive** - they can be combined:

```python
import hpxpy as hpx
import numpy as np

# ==============================================
# Level 1: Automatic (Legate-style)
# Simplest - system decides everything
# ==============================================

with hpx.distributed():
    # All arrays created here are automatically distributed
    a = hpx.zeros((10000, 10000))    # Auto-partitioned across localities
    b = hpx.ones((10000, 10000))     # Auto-partitioned
    c = a + b                         # Distributed computation

# Or set globally
hpx.set_default_distribution('auto')
arr = hpx.zeros((10000,))  # Automatically distributed if large enough

# ==============================================
# Level 2: Chunk-based (Dask-style)
# Familiar to Dask users, controls partition size
# ==============================================

# Auto chunk sizing
arr = hpx.zeros((10000,), chunks='auto')

# Explicit chunk sizes (each chunk = one partition)
arr = hpx.zeros((10000, 10000), chunks=(2500, 2500))  # 16 partitions

# Chunk only along one axis
arr = hpx.zeros((10000, 10000), chunks=(2500, -1))    # 4 row partitions

# ==============================================
# Level 3: Explicit Distribution (HPX-native)
# Full control over distribution policy
# ==============================================

# Block distribution (contiguous chunks)
arr = hpx.zeros(
    (10000,),
    distribution=hpx.distribution.block,
    num_partitions=4
)

# Cyclic distribution (round-robin elements)
arr = hpx.zeros(
    (10000,),
    distribution=hpx.distribution.cyclic,
    num_partitions=4
)

# Block-cyclic (blocks distributed round-robin)
arr = hpx.zeros(
    (10000,),
    distribution=hpx.distribution.block_cyclic,
    block_size=100,
    num_partitions=4
)

# Explicit locality placement
arr = hpx.zeros(
    (10000,),
    distribution=hpx.distribution.block,
    localities=[0, 1, 2, 3]  # Specific locality IDs
)

# ==============================================
# Level 4: Device-aware Distribution
# Combine distribution with device placement
# ==============================================

# Distributed across GPUs
arr = hpx.zeros(
    (1_000_000,),
    chunks='auto',
    device='gpu'  # Distribute across available GPUs
)

# Explicit multi-GPU placement
arr = hpx.zeros(
    (1_000_000,),
    distribution=hpx.distribution.block,
    devices=['gpu:0', 'gpu:1', 'gpu:2', 'gpu:3']
)

# Hybrid: some partitions on GPU, some on CPU
arr = hpx.zeros(
    (1_000_000,),
    distribution=hpx.distribution.block,
    devices=['gpu:0', 'gpu:1', 'cpu', 'cpu']
)

# ==============================================
# Combining Approaches
# ==============================================

# Context manager + explicit chunks
with hpx.distributed():
    arr = hpx.zeros((10000,), chunks=(1000,))  # Distributed with specific chunk size

# Context manager + device
with hpx.device('gpu'):
    with hpx.distributed():
        arr = hpx.zeros((10000,))  # Distributed across GPUs

# Global default + override
hpx.set_default_distribution('auto')
local_arr = hpx.zeros((100,), distribution=None)  # Force local
distributed_arr = hpx.zeros((10000,))             # Uses auto default
```

#### Local vs Distributed Execution

HPXPy distinguishes between **local** (single node) and **distributed** (multi-node) execution:

```python
import hpxpy as hpx

# ==============================================
# Local Execution (Single Locality)
# ==============================================

# Default: arrays are local, operations use local threads
arr = hpx.zeros((10000,))              # Local array
result = hpx.sum(arr)                   # Uses local thread pool

# Explicit local scope (prevents accidental distribution)
with hpx.local():
    arr = hpx.zeros((10000,))          # Guaranteed local
    result = hpx.sum(arr)              # Local parallel execution

# Force local even when distributed defaults are set
arr = hpx.zeros((10000,), distribution=None)      # Explicit: no distribution
arr = hpx.zeros((10000,), distribution='local')   # Alias for None

# ==============================================
# Local CPU Parallelism (Multithreaded)
# ==============================================

# All threads on this locality
result = hpx.sum(arr, policy=hpx.execution.par)

# Specific thread count
with hpx.execution.thread_pool(num_threads=8):
    result = hpx.sum(arr)

# NUMA-aware: threads on specific NUMA node
result = hpx.sum(arr, policy=hpx.execution.par.on_numa(0))

# ==============================================
# Local GPU (Single Node, One or More GPUs)
# ==============================================

# Single GPU on this locality
arr = hpx.zeros((10000,), device='gpu')        # Default GPU
arr = hpx.zeros((10000,), device='gpu:0')      # Specific GPU

# Multiple GPUs on this locality (local distribution)
arr = hpx.zeros(
    (1_000_000,),
    device='gpu',
    distribution='local',  # Distribute across LOCAL GPUs only
)

# Explicit list of local GPUs
arr = hpx.zeros(
    (1_000_000,),
    devices=['gpu:0', 'gpu:1'],  # Only local GPUs
    distribution='local',
)

# Context manager for local GPU scope
with hpx.local():
    with hpx.device('gpu'):
        arr = hpx.zeros((10000,))  # On local GPU, not distributed

# ==============================================
# Distributed Execution (Multi-Locality)
# ==============================================

# Distributed across localities (nodes)
arr = hpx.zeros(
    (10000,),
    distribution=hpx.distribution.block,
    localities='all',  # All available localities
)

# Specific localities
arr = hpx.zeros(
    (10000,),
    distribution=hpx.distribution.block,
    localities=[0, 1, 2, 3],  # Specific locality IDs
)

# Distributed across GPUs on MULTIPLE nodes
arr = hpx.zeros(
    (1_000_000,),
    device='gpu',
    distribution='distributed',  # Across all localities
)
```

#### Scope Summary

| Scope | Keyword | Meaning |
|-------|---------|---------|
| **Local CPU** | `distribution=None` or `'local'` | Single node, local thread pool |
| **Local GPU** | `device='gpu'`, `distribution='local'` | GPUs on current node only |
| **Distributed CPU** | `distribution=block/cyclic/...` | Across multiple nodes |
| **Distributed GPU** | `device='gpu'`, `distribution='distributed'` | GPUs across multiple nodes |

#### Execution Policy vs Distribution

These are **orthogonal** concepts:

```python
# Execution Policy: HOW to execute (threading strategy)
hpx.execution.seq        # Sequential
hpx.execution.par        # Parallel (threaded)
hpx.execution.par_unseq  # Parallel + vectorized

# Distribution: WHERE data lives (memory location)
distribution=None        # Local memory only
distribution='local'     # Same as None
distribution=block       # Distributed across localities
```

You can combine them:

```python
# Local data, parallel execution
result = hpx.sum(local_arr, policy=hpx.execution.par)

# Distributed data, parallel execution on each locality
result = hpx.sum(distributed_arr, policy=hpx.execution.par)
```

#### Distribution Policy Reference

| Policy | Description | Best For |
|--------|-------------|----------|
| `None` / `'local'` | No distribution, single locality | Local-only workloads |
| `block` | Contiguous chunks across localities | Most workloads, good locality |
| `cyclic` | Round-robin elements | Load balancing irregular data |
| `block_cyclic` | Blocks distributed round-robin | Dense linear algebra (ScaLAPACK-style) |
| `auto` | System chooses based on size/operation | Simplicity, beginners |

#### Interoperability with Other Libraries

```python
# From Dask array (preserves chunks)
import dask.array as da
dask_arr = da.zeros((10000,), chunks=(1000,))
hpx_arr = hpx.from_dask(dask_arr)  # Preserves chunking

# To Dask array
dask_arr = hpx_arr.to_dask()

# From/to CuPy (GPU arrays)
import cupy as cp
cupy_arr = cp.zeros((1000,))
hpx_arr = hpx.from_cupy(cupy_arr)  # Stays on GPU
cupy_arr = hpx_arr.to_cupy()
```

### 3. Execution Policies

```python
import hpxpy as hpx

# Execution policies (similar to C++ execution policies)
hpx.execution.seq       # Sequential execution
hpx.execution.par       # Parallel execution (default)
hpx.execution.par_unseq # Parallel + vectorization hints
hpx.execution.task      # Async execution returning future

# Custom executors
pool_exec = hpx.execution.thread_pool_executor(num_threads=4)
custom_policy = hpx.execution.par.on(pool_exec)

# Usage with algorithms
result = hpx.reduce(arr, policy=hpx.execution.par)
```

### 4. Parallel Algorithms (NumPy-Compatible)

```python
import hpxpy as hpx
import numpy as np

a = hpx.array(np.random.randn(10000))
b = hpx.array(np.random.randn(10000))

# Reductions (return scalars or futures)
total = hpx.sum(a)                    # Sum all elements
mean = hpx.mean(a)                    # Arithmetic mean
prod = hpx.prod(a)                    # Product
min_val = hpx.min(a)                  # Minimum
max_val = hpx.max(a)                  # Maximum
argmin = hpx.argmin(a)                # Index of minimum
argmax = hpx.argmax(a)                # Index of maximum

# With axis support
col_sums = hpx.sum(matrix, axis=0)   # Sum along axis

# Async versions (return futures)
future_sum = hpx.sum(a, policy=hpx.execution.task)
result = future_sum.get()

# Transform operations
c = hpx.add(a, b)                     # Element-wise addition
c = hpx.multiply(a, b)                # Element-wise multiplication
c = hpx.subtract(a, b)                # Element-wise subtraction
c = hpx.divide(a, b)                  # Element-wise division
c = hpx.power(a, 2)                   # Element-wise power
c = hpx.sqrt(a)                       # Element-wise sqrt
c = hpx.exp(a)                        # Element-wise exp
c = hpx.log(a)                        # Element-wise log

# Operator overloading
c = a + b
c = a * b
c = a - b
c = a / b
c = a ** 2

# Scans (prefix operations)
prefix_sum = hpx.cumsum(a)            # Inclusive prefix sum
prefix_prod = hpx.cumprod(a)          # Inclusive prefix product

# Sorting
sorted_arr = hpx.sort(a)              # Parallel sort
indices = hpx.argsort(a)              # Sort indices
partitioned = hpx.partition(a, pivot) # Partition around pivot

# Searching
idx = hpx.searchsorted(sorted_arr, values)
found = hpx.contains(a, value)

# Comparisons
mask = hpx.equal(a, b)                # Element-wise equality
mask = hpx.greater(a, b)              # Element-wise comparison
mask = a > b                          # Operator form

# Boolean reductions
all_true = hpx.all(mask)
any_true = hpx.any(mask)

# Set operations
union = hpx.union(a, b)
intersect = hpx.intersect(a, b)
diff = hpx.setdiff(a, b)
unique = hpx.unique(a)
```

### 5. Distributed Collective Operations

```python
import hpxpy as hpx

# All-reduce across all localities
global_sum = hpx.collectives.all_reduce(
    local_data,
    op=hpx.collectives.sum  # or min, max, prod, custom
)

# Broadcast from root
data = hpx.collectives.broadcast(data, root=0)

# Gather to root
gathered = hpx.collectives.gather(local_data, root=0)

# All-gather (gather to all)
all_data = hpx.collectives.all_gather(local_data)

# Scatter from root
local_chunk = hpx.collectives.scatter(global_data, root=0)

# All-to-all exchange
exchanged = hpx.collectives.all_to_all(data_to_send)

# Barrier synchronization
hpx.collectives.barrier()

# Distributed scan
prefix_sums = hpx.collectives.scan(local_data, op=hpx.collectives.sum)
```

### 6. Futures and Asynchronous Execution

```python
import hpxpy as hpx

# Async function execution
def compute(x):
    return x ** 2

future = hpx.async(compute, 42)
result = future.get()  # Block until ready

# Check readiness
if future.is_ready():
    result = future.get()

# Continuations
future2 = future.then(lambda x: x + 1)

# Wait for multiple futures
futures = [hpx.async(compute, i) for i in range(10)]
results = hpx.wait_all(futures)    # Wait for all, return results
first = hpx.wait_any(futures)       # Wait for first to complete

# Dataflow (automatic dependency resolution)
a = hpx.async(load_data, "file1.dat")
b = hpx.async(load_data, "file2.dat")
c = hpx.dataflow(process, a, b)  # Runs when a and b are ready

# Async array operations
future_sum = hpx.sum(arr, policy=hpx.execution.task)
future_sort = hpx.sort(arr, policy=hpx.execution.task)
```

### 7. Linear Algebra Extensions (Future)

```python
import hpxpy as hpx
import hpxpy.linalg as la

# Matrix operations
c = la.dot(a, b)          # Matrix multiplication
c = la.matmul(a, b)       # Alias
c = a @ b                 # Operator form

# Decompositions (distributed)
q, r = la.qr(a)
u, s, vh = la.svd(a)
eigenvalues, eigenvectors = la.eig(a)

# Solvers
x = la.solve(A, b)        # Solve Ax = b
x = la.lstsq(A, b)        # Least squares solution

# Norms
norm = la.norm(a)
norm_inf = la.norm(a, ord=np.inf)
```

---

## Implementation Phases

### Phase 1: Foundation (Core Bindings)

**Goal:** Basic HPX functionality accessible from Python

**Deliverables:**
- [ ] Build system integration (CMake + scikit-build-core)
- [ ] HPX runtime initialization/finalization
- [ ] Basic future<T> bindings for common types
- [ ] hpx::async() binding
- [ ] GIL release utilities
- [ ] Exception translation

**Key Files:**
- `src/bindings/core_module.cpp`
- `src/bindings/runtime_bindings.cpp`
- `src/bindings/future_bindings.cpp`

### Phase 2: Local Parallelism

**Goal:** NumPy-like operations with parallel execution on single node

**Deliverables:**
- [ ] Local array wrapper (std::vector-based)
- [ ] Execution policies (seq, par, par_unseq)
- [ ] Parallel algorithms: reduce, transform, sort, scan
- [ ] NumPy interop (zero-copy buffer protocol)
- [ ] Operator overloading

**Key Files:**
- `src/bindings/algorithm_bindings.cpp`
- `src/bindings/execution_bindings.cpp`
- `src/types/numpy_compat.hpp`

### Phase 3: Distributed Arrays

**Goal:** Distributed data structures across multiple localities

**Deliverables:**
- [ ] partitioned_vector bindings
- [ ] Distribution policies
- [ ] Distributed array creation functions
- [ ] to_numpy() gathering
- [ ] Partition introspection

**Key Files:**
- `src/bindings/array_bindings.cpp`
- `src/types/distributed_array.hpp`

### Phase 4: Distributed Operations

**Goal:** Full distributed computing capabilities

**Deliverables:**
- [ ] Collective operations (all_reduce, broadcast, gather, scatter)
- [ ] Distributed algorithms on partitioned_vector
- [ ] SPMD execution model support
- [ ] Multi-locality deployment support

**Key Files:**
- `src/bindings/collective_bindings.cpp`

### Phase 5: GPU Acceleration

**Goal:** GPU support with automatic device selection

**Deliverables:**
- [ ] GPU detection and runtime query API
- [ ] GPU array creation and transfers
- [ ] GPU execution policies
- [ ] CUDA backend implementation
- [ ] GPU reductions (sum, min, max, etc.)
- [ ] GPU transforms (element-wise operations)
- [ ] Multi-GPU distribution support
- [ ] Hybrid CPU+GPU execution

**Key Files:**
- `src/bindings/gpu_bindings.cpp`
- `src/types/gpu_array.hpp`
- `hpxpy/gpu.py`

### Phase 6: Advanced Features

**Goal:** Production-ready library with advanced capabilities

**Deliverables:**
- [ ] SYCL backend (Intel/AMD GPUs)
- [ ] Checkpointing
- [ ] Performance counters exposure
- [ ] Linear algebra operations (CPU + GPU)
- [ ] Comprehensive documentation
- [ ] CI/CD pipeline
- [ ] PyPI packaging

---

## Technical Considerations

### GIL Management

Python's Global Interpreter Lock (GIL) must be released during HPX operations:

```cpp
// C++ side
py::array_t<double> parallel_sum(py::array_t<double> arr) {
    py::gil_scoped_release release;  // Release GIL

    // HPX parallel operations here
    auto result = hpx::reduce(hpx::execution::par, ...);

    py::gil_scoped_acquire acquire;  // Re-acquire before returning
    return py::array_t<double>(...);
}
```

### Memory Management

- Use `py::array_t` buffer protocol for zero-copy NumPy interop
- `partitioned_vector` data stays distributed; only gather on explicit `.to_numpy()`
- Reference counting must handle both Python and HPX lifetimes

### Type Support

Priority types for template instantiation:
1. `float64` (double)
2. `float32` (float)
3. `int64` (long long)
4. `int32` (int)
5. `complex128` (std::complex<double>)
6. `complex64` (std::complex<float>)

### Error Handling

```cpp
// Translate HPX exceptions to Python
void register_exception_translators() {
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const hpx::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}
```

### Multi-Locality Deployment

```bash
# Single node (default)
python my_script.py

# Multi-node with mpirun
mpirun -np 4 python my_script.py

# HPX command line options
python my_script.py --hpx:threads=8 --hpx:localities=4
```

---

## GPU Support

### Overview

HPXPy will support GPU acceleration through HPX's compute capabilities, with automatic fallback to CPU when GPUs are unavailable. The design prioritizes:

1. **Runtime detection** - Automatically detect available GPUs
2. **Transparent execution** - Same API for CPU and GPU operations
3. **Explicit control** - Users can force specific execution targets
4. **Multi-GPU support** - Distribute work across multiple GPUs

### Supported GPU Backends

| Backend | Platform | HPX Module | Status |
|---------|----------|------------|--------|
| **CUDA** | NVIDIA GPUs | `hpx::cuda` | Primary target |
| **SYCL** | Intel/AMD/NVIDIA | `hpx::sycl` | Secondary target |
| **HIP** | AMD GPUs | via SYCL | Future |
| **ROCm** | AMD GPUs | via HIP | Future |

### GPU API Design

```python
import hpxpy as hpx
import numpy as np

# ============================================
# Runtime GPU Detection
# ============================================

# Check GPU availability
hpx.gpu.is_available()          # True if any GPU is available
hpx.gpu.device_count()          # Number of GPUs
hpx.gpu.current_device()        # Current default GPU
hpx.gpu.device_name(0)          # "NVIDIA A100-SXM4-40GB"
hpx.gpu.device_properties(0)    # Dict of device properties

# List all available devices
for i, dev in enumerate(hpx.gpu.devices()):
    print(f"GPU {i}: {dev.name}, {dev.memory_gb:.1f} GB")

# ============================================
# Execution Targets
# ============================================

# Execution target specifiers
hpx.target.cpu                  # Force CPU execution
hpx.target.gpu                  # Use default GPU (or fail if none)
hpx.target.gpu(0)               # Specific GPU by index
hpx.target.auto                 # Automatic selection (default)

# Usage with operations
result = hpx.sum(arr, target=hpx.target.gpu)
result = hpx.sort(arr, target=hpx.target.gpu(1))  # Use GPU 1

# Combined with execution policies
result = hpx.reduce(
    arr,
    policy=hpx.execution.par,
    target=hpx.target.gpu
)

# ============================================
# GPU Arrays
# ============================================

# Create array directly on GPU
gpu_arr = hpx.array([1, 2, 3, 4], device='gpu')
gpu_arr = hpx.zeros((1000, 1000), device='gpu:0')  # Specific GPU
gpu_arr = hpx.ones((1000,), device='cuda:0')       # CUDA-specific

# Transfer from CPU to GPU
cpu_arr = hpx.array(np.random.randn(10000))
gpu_arr = cpu_arr.to('gpu')           # Copy to default GPU
gpu_arr = cpu_arr.to('gpu:1')         # Copy to specific GPU
gpu_arr = cpu_arr.gpu()               # Shorthand

# Transfer from GPU to CPU
cpu_arr = gpu_arr.to('cpu')
cpu_arr = gpu_arr.cpu()               # Shorthand
np_arr = gpu_arr.to_numpy()           # Direct to NumPy (via CPU)

# Check device location
arr.device                            # 'cpu', 'gpu:0', etc.
arr.is_gpu                            # True if on GPU
arr.is_cpu                            # True if on CPU

# ============================================
# Automatic Device Selection
# ============================================

# By default, operations use 'auto' target selection:
# 1. If array is on GPU, execute on that GPU
# 2. If array is on CPU and GPU available and size > threshold, use GPU
# 3. Otherwise use CPU

# Configure auto-selection behavior
hpx.config.gpu_threshold = 100_000    # Min elements for GPU (default)
hpx.config.prefer_gpu = True          # Prefer GPU when available

# ============================================
# Multi-GPU Operations
# ============================================

# Distributed array across multiple GPUs
multi_gpu_arr = hpx.zeros(
    (1_000_000,),
    distribution=hpx.distribution.block,
    devices=['gpu:0', 'gpu:1', 'gpu:2', 'gpu:3']
)

# Collective operations across GPUs
result = hpx.sum(multi_gpu_arr)  # Automatically reduces across GPUs

# ============================================
# Context Managers
# ============================================

# Set default device for a block
with hpx.device('gpu:0'):
    a = hpx.zeros((1000, 1000))  # Created on GPU 0
    b = hpx.ones((1000, 1000))   # Created on GPU 0
    c = a + b                     # Computed on GPU 0

# Synchronization
with hpx.gpu.stream() as stream:
    # Operations are queued
    a = hpx.zeros((1000,), device='gpu')
    b = hpx.sum(a)
# Stream synchronized on exit

# ============================================
# Memory Management
# ============================================

# Query GPU memory
hpx.gpu.memory_allocated(0)     # Bytes currently allocated on GPU 0
hpx.gpu.memory_reserved(0)      # Bytes reserved (including cache)
hpx.gpu.memory_total(0)         # Total GPU memory

# Memory pool control
hpx.gpu.empty_cache()           # Release cached memory
hpx.gpu.reset_peak_memory_stats()

# Pinned (page-locked) host memory for faster transfers
pinned_arr = hpx.array(data, pinned=True)
```

### GPU Execution Policies

```python
# GPU-specific execution policies
hpx.execution.gpu              # Basic GPU execution
hpx.execution.gpu_par          # Parallel across GPU threads
hpx.execution.gpu_task         # Async GPU execution returning future

# Hybrid CPU+GPU execution
hpx.execution.hybrid           # Automatically split across CPU and GPU

# Example: Hybrid execution for very large arrays
large_arr = hpx.arange(100_000_000)
result = hpx.sum(large_arr, policy=hpx.execution.hybrid)
# Automatically partitions work between CPU and available GPUs
```

### GPU-Accelerated Operations

**Phase 1 - Core Operations (GPU):**
- Reductions: `sum`, `prod`, `min`, `max`, `mean`, `std`, `var`
- Element-wise: `add`, `multiply`, `subtract`, `divide`, `power`
- Math functions: `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`
- Comparisons: `equal`, `greater`, `less`, `logical_and/or/not`

**Phase 2 - Advanced Operations (GPU):**
- Sorting: `sort`, `argsort`, `partition`
- Scans: `cumsum`, `cumprod`
- Search: `searchsorted`, `unique`, `where`

**Phase 3 - Linear Algebra (GPU via cuBLAS/etc.):**
- Matrix multiplication: `matmul`, `dot`
- Decompositions: `qr`, `svd`, `eig`, `cholesky`
- Solvers: `solve`, `lstsq`

### C++ GPU Bindings

```cpp
// src/bindings/gpu_bindings.cpp
#include <pybind11/pybind11.h>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>

namespace py = pybind11;

// GPU executor wrapper
class GPUExecutor {
    hpx::cuda::experimental::cuda_executor exec_;
public:
    GPUExecutor(int device_id = 0)
        : exec_(hpx::cuda::experimental::target(device_id)) {}

    template<typename F, typename... Args>
    auto async(F&& f, Args&&... args) {
        return hpx::async(exec_, std::forward<F>(f),
                         std::forward<Args>(args)...);
    }
};

// GPU array with device memory
template<typename T>
class GPUArray {
    hpx::cuda::experimental::allocator<T> alloc_;
    std::vector<T, decltype(alloc_)> data_;
    int device_id_;

public:
    GPUArray(size_t size, int device = 0)
        : alloc_(hpx::cuda::experimental::target(device))
        , data_(size, alloc_)
        , device_id_(device) {}

    // Transfer to/from host
    void copy_from_host(const T* host_data, size_t size);
    void copy_to_host(T* host_data, size_t size) const;

    int device() const { return device_id_; }
    size_t size() const { return data_.size(); }
    T* data() { return data_.data(); }
};

void bind_gpu_module(py::module_& m) {
    auto gpu = m.def_submodule("gpu", "GPU support");

    gpu.def("is_available", []() {
        #ifdef HPX_HAVE_CUDA
        int count;
        cudaGetDeviceCount(&count);
        return count > 0;
        #else
        return false;
        #endif
    });

    gpu.def("device_count", []() {
        #ifdef HPX_HAVE_CUDA
        int count;
        cudaGetDeviceCount(&count);
        return count;
        #else
        return 0;
        #endif
    });

    gpu.def("device_name", [](int device) {
        #ifdef HPX_HAVE_CUDA
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        return std::string(prop.name);
        #else
        throw std::runtime_error("GPU support not available");
        #endif
    });

    // ... more bindings
}
```

### Build Configuration for GPU

```cmake
# CMakeLists.txt additions for GPU support
option(HPXPY_WITH_CUDA "Enable CUDA support" OFF)
option(HPXPY_WITH_SYCL "Enable SYCL support" OFF)

if(HPXPY_WITH_CUDA)
    find_package(HPX REQUIRED COMPONENTS cuda)
    find_package(CUDAToolkit REQUIRED)

    target_compile_definitions(_core PRIVATE HPXPY_HAVE_CUDA)
    target_link_libraries(_core PRIVATE
        HPX::hpx
        CUDA::cudart
        CUDA::cublas  # For linear algebra
    )
endif()

if(HPXPY_WITH_SYCL)
    find_package(HPX REQUIRED COMPONENTS sycl)

    target_compile_definitions(_core PRIVATE HPXPY_HAVE_SYCL)
    target_link_libraries(_core PRIVATE HPX::hpx)
endif()
```

### GPU Testing Strategy

```python
# tests/gpu/test_gpu_operations.py
import pytest
import numpy as np
import hpxpy as hpx

# Skip all tests if no GPU available
pytestmark = pytest.mark.skipif(
    not hpx.gpu.is_available(),
    reason="GPU not available"
)

class TestGPUArrays:
    def test_create_on_gpu(self):
        arr = hpx.zeros((1000,), device='gpu')
        assert arr.is_gpu
        assert arr.device.startswith('gpu')

    def test_transfer_to_gpu(self):
        cpu_arr = hpx.array(np.random.randn(1000))
        gpu_arr = cpu_arr.to('gpu')
        assert gpu_arr.is_gpu
        np.testing.assert_array_equal(
            cpu_arr.to_numpy(),
            gpu_arr.to_numpy()
        )

    def test_gpu_reduction(self):
        np_arr = np.random.randn(100000)
        gpu_arr = hpx.array(np_arr, device='gpu')
        result = hpx.sum(gpu_arr)
        np.testing.assert_allclose(result, np.sum(np_arr), rtol=1e-5)

class TestGPUBenchmarks:
    """Verify GPU provides speedup for large arrays."""

    @pytest.mark.benchmark
    def test_gpu_speedup_reduction(self, benchmark):
        size = 10_000_000
        np_arr = np.random.randn(size)
        gpu_arr = hpx.array(np_arr, device='gpu')
        cpu_arr = hpx.array(np_arr, device='cpu')

        gpu_time = benchmark(hpx.sum, gpu_arr)
        cpu_time = benchmark(hpx.sum, cpu_arr)

        # GPU should be faster for large arrays
        assert gpu_time < cpu_time * 0.5  # At least 2x speedup
```

### GPU Memory Considerations

```python
# Best practices for GPU memory management

# 1. Reuse arrays when possible (avoid repeated allocation)
gpu_buffer = hpx.empty((10000,), device='gpu')
for data in data_stream:
    gpu_buffer[:len(data)] = data  # Reuse buffer
    result = hpx.sum(gpu_buffer[:len(data)])

# 2. Use pinned memory for frequent CPU<->GPU transfers
pinned = hpx.array(large_data, pinned=True)
gpu_arr = pinned.to('gpu')  # Faster transfer

# 3. Explicit memory management for large workloads
with hpx.gpu.memory_pool(device=0, initial_size='4GB'):
    # Operations use pre-allocated pool
    for _ in range(1000):
        arr = hpx.random.randn(1000000, device='gpu')
        result = hpx.sum(arr)
    # Pool released on exit

# 4. Monitor memory usage
print(f"GPU memory: {hpx.gpu.memory_allocated(0) / 1e9:.2f} GB")
```

---

## Build System

### CMakeLists.txt Structure

```cmake
cmake_minimum_required(VERSION 3.18)
project(hpxpy VERSION 0.1.0 LANGUAGES CXX)

# Find dependencies
find_package(HPX REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Main extension module
pybind11_add_module(_core MODULE
    src/bindings/core_module.cpp
    src/bindings/runtime_bindings.cpp
    src/bindings/future_bindings.cpp
    src/bindings/algorithm_bindings.cpp
    src/bindings/array_bindings.cpp
    src/bindings/execution_bindings.cpp
    src/bindings/collective_bindings.cpp
)

target_link_libraries(_core PRIVATE HPX::hpx HPX::wrap_main)
target_include_directories(_core PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# Install
install(TARGETS _core DESTINATION hpxpy)
```

### pyproject.toml

```toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "hpxpy"
version = "0.1.0"
description = "High-performance distributed NumPy-like arrays powered by HPX"
requires-python = ">=3.9"
dependencies = ["numpy>=1.20"]

[tool.scikit-build]
cmake.args = ["-DHPX_DIR=/path/to/hpx"]
wheel.packages = ["hpxpy"]
```

---

## Testing Strategy

### Test Directory Structure

```
tests/
├── unit/                      # Fast, isolated unit tests
│   ├── test_array_creation.py
│   ├── test_array_properties.py
│   ├── test_algorithms.py
│   ├── test_reductions.py
│   ├── test_execution_policies.py
│   ├── test_futures.py
│   └── test_dtype_support.py
├── integration/               # Multi-component integration tests
│   ├── test_numpy_interop.py
│   ├── test_algorithm_chains.py
│   ├── test_async_workflows.py
│   └── test_memory_management.py
├── distributed/               # Multi-locality tests
│   ├── test_collectives.py
│   ├── test_partitioned_vector.py
│   ├── test_distributed_algorithms.py
│   └── test_multi_node.py
├── compatibility/             # NumPy API compatibility
│   ├── test_numpy_array_api.py
│   ├── test_numpy_ufuncs.py
│   └── test_numpy_behavior.py
├── property/                  # Property-based tests (Hypothesis)
│   ├── test_algorithm_properties.py
│   └── test_reduction_properties.py
├── stress/                    # Stress and edge case tests
│   ├── test_large_arrays.py
│   ├── test_many_tasks.py
│   └── test_memory_pressure.py
└── conftest.py                # Shared fixtures
```

### Unit Tests

```python
# tests/unit/test_algorithms.py
import pytest
import numpy as np
import hpxpy as hpx

@pytest.fixture(scope="module")
def hpx_runtime():
    """Initialize HPX runtime once per test module."""
    hpx.init(num_threads=4)
    yield
    hpx.finalize()

@pytest.fixture
def sample_arrays():
    """Generate test arrays of various sizes and dtypes."""
    return {
        'small_float': hpx.array(np.random.randn(100), dtype=np.float64),
        'medium_float': hpx.array(np.random.randn(10000), dtype=np.float64),
        'large_float': hpx.array(np.random.randn(1000000), dtype=np.float64),
        'int_array': hpx.array(np.random.randint(0, 100, 1000), dtype=np.int64),
        'complex_array': hpx.array(np.random.randn(1000) + 1j*np.random.randn(1000)),
    }

class TestReductions:
    """Test all reduction operations."""

    @pytest.mark.parametrize("policy", [
        hpx.execution.seq,
        hpx.execution.par,
        hpx.execution.par_unseq,
    ])
    def test_sum_policies(self, hpx_runtime, sample_arrays, policy):
        arr = sample_arrays['medium_float']
        np_arr = arr.to_numpy()
        result = hpx.sum(arr, policy=policy)
        np.testing.assert_allclose(result, np.sum(np_arr), rtol=1e-10)

    @pytest.mark.parametrize("reduction", [
        ('sum', np.sum),
        ('prod', np.prod),
        ('min', np.min),
        ('max', np.max),
        ('mean', np.mean),
        ('std', np.std),
        ('var', np.var),
    ])
    def test_reduction_correctness(self, hpx_runtime, sample_arrays, reduction):
        hpx_func_name, np_func = reduction
        hpx_func = getattr(hpx, hpx_func_name)
        arr = sample_arrays['medium_float']
        np_arr = arr.to_numpy()
        np.testing.assert_allclose(hpx_func(arr), np_func(np_arr), rtol=1e-10)

    def test_reduction_empty_array(self, hpx_runtime):
        """Test reductions on empty arrays match NumPy behavior."""
        empty = hpx.array([])
        with pytest.raises(ValueError):
            hpx.min(empty)  # Should raise like NumPy

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_reduction_dtypes(self, hpx_runtime, dtype):
        arr = hpx.array(np.arange(100), dtype=dtype)
        result = hpx.sum(arr)
        assert result == 4950

class TestTransforms:
    """Test element-wise transform operations."""

    def test_add_arrays(self, hpx_runtime, sample_arrays):
        a = sample_arrays['medium_float']
        b = sample_arrays['medium_float']
        result = hpx.add(a, b)
        expected = a.to_numpy() + b.to_numpy()
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_operator_overloading(self, hpx_runtime):
        a = hpx.array([1.0, 2.0, 3.0])
        b = hpx.array([4.0, 5.0, 6.0])

        # Test all operators
        np.testing.assert_array_equal((a + b).to_numpy(), [5.0, 7.0, 9.0])
        np.testing.assert_array_equal((a - b).to_numpy(), [-3.0, -3.0, -3.0])
        np.testing.assert_array_equal((a * b).to_numpy(), [4.0, 10.0, 18.0])
        np.testing.assert_array_equal((a / b).to_numpy(), [0.25, 0.4, 0.5])
        np.testing.assert_array_equal((a ** 2).to_numpy(), [1.0, 4.0, 9.0])

class TestSorting:
    """Test sorting algorithms."""

    def test_sort_correctness(self, hpx_runtime):
        np_arr = np.random.randn(10000)
        hpx_arr = hpx.array(np_arr)
        sorted_hpx = hpx.sort(hpx_arr)
        np.testing.assert_array_equal(sorted_hpx.to_numpy(), np.sort(np_arr))

    def test_sort_stability(self, hpx_runtime):
        """Test that stable_sort preserves order of equal elements."""
        # Create array with duplicate keys
        keys = np.array([3, 1, 2, 1, 3, 2])
        values = np.array([0, 1, 2, 3, 4, 5])
        # ... stability verification

    def test_argsort(self, hpx_runtime):
        np_arr = np.random.randn(1000)
        hpx_arr = hpx.array(np_arr)
        indices = hpx.argsort(hpx_arr)
        np.testing.assert_array_equal(indices.to_numpy(), np.argsort(np_arr))
```

### Integration Tests

```python
# tests/integration/test_numpy_interop.py
import pytest
import numpy as np
import hpxpy as hpx

class TestNumPyInteroperability:
    """Test seamless interop with NumPy arrays."""

    def test_zero_copy_from_numpy(self, hpx_runtime):
        """Verify zero-copy when possible."""
        np_arr = np.ascontiguousarray(np.random.randn(1000))
        hpx_arr = hpx.from_numpy(np_arr, copy=False)

        # Modify original, should reflect in hpx array
        np_arr[0] = 999.0
        assert hpx_arr[0] == 999.0

    def test_buffer_protocol(self, hpx_runtime):
        """Test that hpx arrays support Python buffer protocol."""
        hpx_arr = hpx.array([1, 2, 3, 4, 5])
        np_view = np.asarray(hpx_arr)  # Should use buffer protocol
        np.testing.assert_array_equal(np_view, [1, 2, 3, 4, 5])

    def test_mixed_operations(self, hpx_runtime):
        """Test operations mixing hpx and numpy arrays."""
        hpx_arr = hpx.array([1.0, 2.0, 3.0])
        np_arr = np.array([4.0, 5.0, 6.0])

        # hpx + numpy should work
        result = hpx.add(hpx_arr, hpx.from_numpy(np_arr))
        np.testing.assert_array_equal(result.to_numpy(), [5.0, 7.0, 9.0])

class TestAsyncWorkflows:
    """Test asynchronous execution patterns."""

    def test_future_chaining(self, hpx_runtime):
        arr = hpx.array(np.random.randn(10000))

        # Chain of async operations
        f1 = hpx.sum(arr, policy=hpx.execution.task)
        f2 = f1.then(lambda x: x * 2)
        f3 = f2.then(lambda x: x + 1)

        result = f3.get()
        expected = hpx.sum(arr) * 2 + 1
        np.testing.assert_allclose(result, expected)

    def test_dataflow_dependencies(self, hpx_runtime):
        """Test automatic dependency resolution with dataflow."""
        a = hpx.async(lambda: hpx.array(np.random.randn(1000)))
        b = hpx.async(lambda: hpx.array(np.random.randn(1000)))

        def combine(x, y):
            return hpx.sum(hpx.add(x, y))

        result = hpx.dataflow(combine, a, b)
        # Result should be ready only after both a and b complete
        assert result.get() is not None
```

### Distributed Tests

```python
# tests/distributed/test_collectives.py
import pytest
import numpy as np
import hpxpy as hpx

# These tests require multiple localities
pytestmark = pytest.mark.distributed

class TestCollectives:
    """Test distributed collective operations."""

    @pytest.fixture
    def multi_locality_runtime(self):
        """Require at least 2 localities for distributed tests."""
        hpx.init(config=["--hpx:localities=2"])
        if hpx.num_localities() < 2:
            pytest.skip("Requires multiple localities")
        yield
        hpx.finalize()

    def test_all_reduce_sum(self, multi_locality_runtime):
        local_data = hpx.array([hpx.locality_id() + 1.0])
        global_sum = hpx.collectives.all_reduce(local_data, op=hpx.collectives.sum)

        # Sum of 1 + 2 + ... + num_localities
        n = hpx.num_localities()
        expected = n * (n + 1) / 2
        assert global_sum.to_numpy()[0] == expected

    def test_broadcast(self, multi_locality_runtime):
        if hpx.locality_id() == 0:
            data = hpx.array([42.0, 43.0, 44.0])
        else:
            data = hpx.empty((3,))

        result = hpx.collectives.broadcast(data, root=0)
        np.testing.assert_array_equal(result.to_numpy(), [42.0, 43.0, 44.0])

    def test_gather(self, multi_locality_runtime):
        local_data = hpx.array([float(hpx.locality_id())])
        gathered = hpx.collectives.gather(local_data, root=0)

        if hpx.locality_id() == 0:
            expected = np.arange(hpx.num_localities(), dtype=np.float64)
            np.testing.assert_array_equal(gathered.to_numpy(), expected)

class TestDistributedArrays:
    """Test partitioned_vector operations."""

    def test_distributed_creation(self, multi_locality_runtime):
        arr = hpx.zeros(
            (10000,),
            distribution=hpx.distribution.block,
            num_partitions=hpx.num_localities()
        )
        assert arr.partitions == hpx.num_localities()
        assert arr.size == 10000

    def test_distributed_reduction(self, multi_locality_runtime):
        arr = hpx.arange(0, 1000, distribution=hpx.distribution.block)
        total = hpx.sum(arr)
        assert total == sum(range(1000))
```

### Property-Based Tests

```python
# tests/property/test_algorithm_properties.py
import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
import hpxpy as hpx

class TestAlgorithmProperties:
    """Property-based tests using Hypothesis."""

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10000))
    @settings(max_examples=100)
    def test_sort_is_sorted(self, data):
        """Property: sorted output should be in ascending order."""
        arr = hpx.array(data)
        sorted_arr = hpx.sort(arr)
        result = sorted_arr.to_numpy()
        assert all(result[i] <= result[i+1] for i in range(len(result)-1))

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10000))
    @settings(max_examples=100)
    def test_sort_preserves_elements(self, data):
        """Property: sorting should not add or remove elements."""
        arr = hpx.array(data)
        sorted_arr = hpx.sort(arr)
        np.testing.assert_array_equal(
            np.sort(arr.to_numpy()),
            sorted_arr.to_numpy()
        )

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10000))
    @settings(max_examples=100)
    def test_sum_commutative(self, data):
        """Property: sum should be same regardless of execution policy."""
        arr = hpx.array(data)
        seq_sum = hpx.sum(arr, policy=hpx.execution.seq)
        par_sum = hpx.sum(arr, policy=hpx.execution.par)
        np.testing.assert_allclose(seq_sum, par_sum, rtol=1e-10)

    @given(
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=1000),
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=1000)
    )
    def test_add_commutative(self, data1, data2):
        """Property: a + b == b + a."""
        min_len = min(len(data1), len(data2))
        a = hpx.array(data1[:min_len])
        b = hpx.array(data2[:min_len])

        np.testing.assert_array_equal(
            hpx.add(a, b).to_numpy(),
            hpx.add(b, a).to_numpy()
        )

class TestReductionProperties:
    """Property-based tests for reductions."""

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=10000))
    def test_min_max_bounds(self, data):
        """Property: min <= all elements <= max."""
        arr = hpx.array(data)
        min_val = hpx.min(arr)
        max_val = hpx.max(arr)

        np_arr = arr.to_numpy()
        assert all(min_val <= x <= max_val for x in np_arr)

    @given(st.lists(st.floats(min_value=0.1, max_value=10, allow_nan=False), min_size=1, max_size=100))
    def test_prod_positive(self, data):
        """Property: product of positive numbers is positive."""
        arr = hpx.array(data)
        assert hpx.prod(arr) > 0
```

### NumPy Compatibility Tests

```python
# tests/compatibility/test_numpy_array_api.py
import pytest
import numpy as np
import hpxpy as hpx

class TestNumPyArrayAPICompliance:
    """
    Test compliance with NumPy Array API standard.
    See: https://data-apis.org/array-api/latest/
    """

    # Creation functions
    @pytest.mark.parametrize("func,args,kwargs", [
        ("zeros", ((10, 10),), {}),
        ("ones", ((10, 10),), {}),
        ("empty", ((10, 10),), {}),
        ("full", ((10, 10), 3.14), {}),
        ("arange", (0, 100, 2), {}),
        ("linspace", (0, 1, 100), {}),
    ])
    def test_creation_functions(self, hpx_runtime, func, args, kwargs):
        hpx_func = getattr(hpx, func)
        np_func = getattr(np, func)

        hpx_result = hpx_func(*args, **kwargs)
        np_result = np_func(*args, **kwargs)

        assert hpx_result.shape == np_result.shape
        np.testing.assert_array_almost_equal(hpx_result.to_numpy(), np_result)

    # Element-wise functions
    @pytest.mark.parametrize("func", [
        "abs", "sqrt", "exp", "log", "sin", "cos", "tan",
        "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
        "floor", "ceil", "round",
    ])
    def test_elementwise_functions(self, hpx_runtime, func):
        np_arr = np.abs(np.random.randn(100)) + 0.1  # Ensure valid domain
        hpx_arr = hpx.from_numpy(np_arr)

        if hasattr(hpx, func):
            hpx_func = getattr(hpx, func)
            np_func = getattr(np, func)
            np.testing.assert_array_almost_equal(
                hpx_func(hpx_arr).to_numpy(),
                np_func(np_arr)
            )

    # Reduction functions
    @pytest.mark.parametrize("func", ["sum", "prod", "min", "max", "mean", "std", "var"])
    def test_reduction_functions(self, hpx_runtime, func):
        np_arr = np.random.randn(100)
        hpx_arr = hpx.from_numpy(np_arr)

        hpx_func = getattr(hpx, func)
        np_func = getattr(np, func)
        np.testing.assert_allclose(hpx_func(hpx_arr), np_func(np_arr), rtol=1e-10)

    # Axis support
    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    def test_reduction_axis(self, hpx_runtime, axis):
        np_arr = np.random.randn(10, 20)
        hpx_arr = hpx.from_numpy(np_arr)

        np.testing.assert_array_almost_equal(
            hpx.sum(hpx_arr, axis=axis).to_numpy() if axis is not None else hpx.sum(hpx_arr),
            np.sum(np_arr, axis=axis)
        )
```

---

## Benchmarking Framework

### Benchmark Directory Structure

```
benchmarks/
├── micro/                     # Microbenchmarks (single operations)
│   ├── bench_reductions.py
│   ├── bench_transforms.py
│   ├── bench_sorting.py
│   └── bench_creation.py
├── macro/                     # Macrobenchmarks (real workloads)
│   ├── bench_matrix_multiply.py
│   ├── bench_fft.py
│   ├── bench_kmeans.py
│   └── bench_pagerank.py
├── scaling/                   # Scaling studies
│   ├── bench_strong_scaling.py
│   ├── bench_weak_scaling.py
│   └── bench_thread_scaling.py
├── comparison/                # Cross-library comparisons
│   ├── bench_vs_numpy.py
│   ├── bench_vs_dask.py
│   └── bench_vs_mpi4py.py
├── memory/                    # Memory benchmarks
│   ├── bench_memory_usage.py
│   └── bench_memory_bandwidth.py
├── conftest.py                # Benchmark fixtures
├── run_benchmarks.py          # Benchmark runner
└── results/                   # Stored benchmark results
    └── .gitkeep
```

### Microbenchmarks

```python
# benchmarks/micro/bench_reductions.py
import pytest
import numpy as np
import hpxpy as hpx

# pytest-benchmark integration
class TestReductionBenchmarks:
    """Microbenchmarks for reduction operations."""

    @pytest.fixture(params=[1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    def array_size(self, request):
        return request.param

    @pytest.fixture
    def numpy_array(self, array_size):
        return np.random.randn(array_size)

    @pytest.fixture
    def hpx_array(self, numpy_array):
        return hpx.from_numpy(numpy_array)

    def test_sum_numpy(self, benchmark, numpy_array):
        benchmark(np.sum, numpy_array)

    def test_sum_hpx_seq(self, benchmark, hpx_array):
        benchmark(hpx.sum, hpx_array, policy=hpx.execution.seq)

    def test_sum_hpx_par(self, benchmark, hpx_array):
        benchmark(hpx.sum, hpx_array, policy=hpx.execution.par)

    def test_min_numpy(self, benchmark, numpy_array):
        benchmark(np.min, numpy_array)

    def test_min_hpx_par(self, benchmark, hpx_array):
        benchmark(hpx.min, hpx_array, policy=hpx.execution.par)

    def test_sort_numpy(self, benchmark, numpy_array):
        arr = numpy_array.copy()
        benchmark(np.sort, arr)

    def test_sort_hpx_par(self, benchmark, hpx_array):
        benchmark(hpx.sort, hpx_array, policy=hpx.execution.par)


# Custom benchmark runner with detailed statistics
class BenchmarkSuite:
    """Custom benchmark suite with statistical analysis."""

    def __init__(self, warmup=3, iterations=10):
        self.warmup = warmup
        self.iterations = iterations
        self.results = []

    def run(self, name, func, *args, **kwargs):
        import time

        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)

        # Timed runs
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        result = {
            'name': name,
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'iterations': self.iterations,
        }
        self.results.append(result)
        return result

    def report(self):
        """Generate benchmark report."""
        print(f"\n{'='*80}")
        print(f"{'Benchmark':<40} {'Mean (ms)':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"{'='*80}")
        for r in self.results:
            print(f"{r['name']:<40} {r['mean']*1000:<12.3f} {r['std']*1000:<10.3f} "
                  f"{r['min']*1000:<10.3f} {r['max']*1000:<10.3f}")
```

### Scaling Benchmarks

```python
# benchmarks/scaling/bench_strong_scaling.py
"""
Strong Scaling: Fixed problem size, increasing resources.
Measures parallel efficiency as we add more threads/nodes.
"""
import numpy as np
import hpxpy as hpx
import json
from datetime import datetime

def strong_scaling_benchmark(problem_size=10_000_000, max_threads=None):
    """
    Run strong scaling study for parallel algorithms.

    Strong scaling efficiency = T(1) / (P * T(P))
    where P is number of processors and T(P) is time with P processors.
    """
    import os
    max_threads = max_threads or os.cpu_count()

    results = {
        'benchmark': 'strong_scaling',
        'problem_size': problem_size,
        'timestamp': datetime.now().isoformat(),
        'results': []
    }

    np_data = np.random.randn(problem_size)

    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    thread_counts = [t for t in thread_counts if t <= max_threads]

    baseline_time = None

    for num_threads in thread_counts:
        hpx.init(num_threads=num_threads)
        hpx_arr = hpx.from_numpy(np_data)

        # Warmup
        for _ in range(3):
            hpx.sum(hpx_arr, policy=hpx.execution.par)

        # Benchmark
        times = []
        for _ in range(10):
            import time
            start = time.perf_counter()
            hpx.sum(hpx_arr, policy=hpx.execution.par)
            end = time.perf_counter()
            times.append(end - start)

        mean_time = np.mean(times)
        if baseline_time is None:
            baseline_time = mean_time

        speedup = baseline_time / mean_time
        efficiency = speedup / num_threads

        results['results'].append({
            'threads': num_threads,
            'mean_time': mean_time,
            'std_time': np.std(times),
            'speedup': speedup,
            'efficiency': efficiency,
        })

        hpx.finalize()

    return results


# benchmarks/scaling/bench_weak_scaling.py
"""
Weak Scaling: Problem size grows with resources.
Measures how well the system handles larger problems with more resources.
"""

def weak_scaling_benchmark(base_size=1_000_000, max_threads=None):
    """
    Run weak scaling study.

    Weak scaling efficiency = T(1) / T(P)
    where problem size = P * base_size
    """
    import os
    max_threads = max_threads or os.cpu_count()

    results = {
        'benchmark': 'weak_scaling',
        'base_size_per_thread': base_size,
        'timestamp': datetime.now().isoformat(),
        'results': []
    }

    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    thread_counts = [t for t in thread_counts if t <= max_threads]

    baseline_time = None

    for num_threads in thread_counts:
        problem_size = base_size * num_threads
        np_data = np.random.randn(problem_size)

        hpx.init(num_threads=num_threads)
        hpx_arr = hpx.from_numpy(np_data)

        # Benchmark
        times = []
        for _ in range(10):
            import time
            start = time.perf_counter()
            hpx.sum(hpx_arr, policy=hpx.execution.par)
            end = time.perf_counter()
            times.append(end - start)

        mean_time = np.mean(times)
        if baseline_time is None:
            baseline_time = mean_time

        efficiency = baseline_time / mean_time

        results['results'].append({
            'threads': num_threads,
            'problem_size': problem_size,
            'mean_time': mean_time,
            'efficiency': efficiency,
        })

        hpx.finalize()

    return results
```

### Comparison Benchmarks

```python
# benchmarks/comparison/bench_vs_numpy.py
"""
Compare HPXPy performance against NumPy across various operations.
"""
import numpy as np
import hpxpy as hpx
import pandas as pd

def compare_operations():
    """Compare HPXPy vs NumPy for common operations."""

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    operations = {
        'sum': (np.sum, hpx.sum),
        'min': (np.min, hpx.min),
        'max': (np.max, hpx.max),
        'sort': (np.sort, hpx.sort),
        'cumsum': (np.cumsum, hpx.cumsum),
        'argsort': (np.argsort, hpx.argsort),
    }

    results = []

    for size in sizes:
        np_arr = np.random.randn(size)
        hpx_arr = hpx.from_numpy(np_arr)

        for op_name, (np_func, hpx_func) in operations.items():
            # NumPy timing
            np_times = []
            for _ in range(10):
                start = time.perf_counter()
                np_func(np_arr.copy() if op_name == 'sort' else np_arr)
                np_times.append(time.perf_counter() - start)

            # HPX timing (parallel)
            hpx_times = []
            for _ in range(10):
                start = time.perf_counter()
                hpx_func(hpx_arr, policy=hpx.execution.par)
                hpx_times.append(time.perf_counter() - start)

            speedup = np.mean(np_times) / np.mean(hpx_times)

            results.append({
                'size': size,
                'operation': op_name,
                'numpy_ms': np.mean(np_times) * 1000,
                'hpxpy_ms': np.mean(hpx_times) * 1000,
                'speedup': speedup,
            })

    return pd.DataFrame(results)


# benchmarks/comparison/bench_vs_dask.py
"""
Compare HPXPy vs Dask for distributed array operations.
"""

def compare_with_dask():
    """Compare HPXPy distributed arrays vs Dask arrays."""
    import dask.array as da

    sizes = [(1000, 1000), (5000, 5000), (10000, 10000)]

    results = []
    for shape in sizes:
        # Dask array
        dask_arr = da.random.random(shape, chunks='auto')

        # HPX distributed array
        hpx_arr = hpx.random.randn(*shape, distribution=hpx.distribution.block)

        # Compare reduction
        dask_time = benchmark(lambda: dask_arr.sum().compute())
        hpx_time = benchmark(lambda: hpx.sum(hpx_arr))

        results.append({
            'shape': shape,
            'operation': 'sum',
            'dask_ms': dask_time * 1000,
            'hpxpy_ms': hpx_time * 1000,
            'speedup': dask_time / hpx_time,
        })

    return results
```

### GPU Benchmarks

```python
# benchmarks/gpu/bench_gpu_operations.py
"""
GPU-specific benchmarks for HPXPy.
"""
import numpy as np
import hpxpy as hpx
import time

def gpu_vs_cpu_benchmark():
    """
    Compare GPU vs CPU performance across array sizes.
    Identifies crossover point where GPU becomes beneficial.
    """
    if not hpx.gpu.is_available():
        print("GPU not available, skipping GPU benchmarks")
        return None

    sizes = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 10_000_000]
    operations = ['sum', 'min', 'max', 'sort']

    results = []

    for size in sizes:
        np_data = np.random.randn(size).astype(np.float32)

        # Create arrays on different devices
        cpu_arr = hpx.array(np_data, device='cpu')
        gpu_arr = hpx.array(np_data, device='gpu')

        for op_name in operations:
            op_func = getattr(hpx, op_name)

            # CPU timing
            cpu_times = []
            for _ in range(10):
                start = time.perf_counter()
                op_func(cpu_arr)
                cpu_times.append(time.perf_counter() - start)

            # GPU timing (includes synchronization)
            gpu_times = []
            for _ in range(10):
                hpx.gpu.synchronize()
                start = time.perf_counter()
                op_func(gpu_arr)
                hpx.gpu.synchronize()
                gpu_times.append(time.perf_counter() - start)

            cpu_mean = np.mean(cpu_times)
            gpu_mean = np.mean(gpu_times)

            results.append({
                'size': size,
                'operation': op_name,
                'cpu_ms': cpu_mean * 1000,
                'gpu_ms': gpu_mean * 1000,
                'speedup': cpu_mean / gpu_mean,
                'gpu_is_faster': gpu_mean < cpu_mean,
            })

    return results


def gpu_transfer_benchmark():
    """
    Measure CPU<->GPU transfer times to understand overhead.
    """
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    results = []

    for size in sizes:
        np_data = np.random.randn(size).astype(np.float64)
        bytes_size = size * 8

        # CPU to GPU transfer
        cpu_arr = hpx.array(np_data, device='cpu')
        h2d_times = []
        for _ in range(10):
            start = time.perf_counter()
            gpu_arr = cpu_arr.to('gpu')
            hpx.gpu.synchronize()
            h2d_times.append(time.perf_counter() - start)

        # GPU to CPU transfer
        d2h_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = gpu_arr.to('cpu')
            d2h_times.append(time.perf_counter() - start)

        h2d_mean = np.mean(h2d_times)
        d2h_mean = np.mean(d2h_times)

        results.append({
            'size': size,
            'bytes': bytes_size,
            'h2d_ms': h2d_mean * 1000,
            'd2h_ms': d2h_mean * 1000,
            'h2d_bandwidth_gbps': (bytes_size / h2d_mean) / 1e9,
            'd2h_bandwidth_gbps': (bytes_size / d2h_mean) / 1e9,
        })

    return results


def multi_gpu_scaling_benchmark():
    """
    Measure scaling across multiple GPUs.
    """
    num_gpus = hpx.gpu.device_count()
    if num_gpus < 2:
        print("Multi-GPU benchmark requires at least 2 GPUs")
        return None

    problem_size = 100_000_000
    results = []

    # Baseline: single GPU
    single_gpu_arr = hpx.zeros((problem_size,), device='gpu:0')
    start = time.perf_counter()
    hpx.sum(single_gpu_arr)
    hpx.gpu.synchronize()
    baseline_time = time.perf_counter() - start

    for num_devices in range(1, num_gpus + 1):
        devices = [f'gpu:{i}' for i in range(num_devices)]

        # Distribute array across GPUs
        multi_arr = hpx.zeros(
            (problem_size,),
            distribution=hpx.distribution.block,
            devices=devices
        )

        times = []
        for _ in range(10):
            hpx.gpu.synchronize()
            start = time.perf_counter()
            hpx.sum(multi_arr)
            hpx.gpu.synchronize()
            times.append(time.perf_counter() - start)

        mean_time = np.mean(times)
        speedup = baseline_time / mean_time
        efficiency = speedup / num_devices

        results.append({
            'num_gpus': num_devices,
            'mean_time_ms': mean_time * 1000,
            'speedup': speedup,
            'efficiency': efficiency,
        })

    return results


def gpu_kernel_latency_benchmark():
    """
    Measure GPU kernel launch latency (overhead for small operations).
    """
    # Very small array to measure pure overhead
    small_arr = hpx.array([1.0], device='gpu')

    latencies = []
    for _ in range(1000):
        hpx.gpu.synchronize()
        start = time.perf_counter()
        hpx.sum(small_arr)
        hpx.gpu.synchronize()
        latencies.append(time.perf_counter() - start)

    return {
        'mean_latency_us': np.mean(latencies) * 1e6,
        'min_latency_us': np.min(latencies) * 1e6,
        'max_latency_us': np.max(latencies) * 1e6,
        'std_latency_us': np.std(latencies) * 1e6,
    }


# benchmarks/gpu/bench_gpu_memory.py
"""
GPU memory benchmarks.
"""

def gpu_memory_bandwidth_benchmark():
    """
    Measure effective GPU memory bandwidth using a simple copy operation.
    Theoretical bandwidth: ~900 GB/s for A100, ~2 TB/s for H100.
    """
    size = 100_000_000  # 800 MB for float64
    np_data = np.random.randn(size).astype(np.float64)
    gpu_arr = hpx.array(np_data, device='gpu')

    # Memory bandwidth = 2 * bytes (read + write) / time
    # For sum: just read, no write
    # For copy/transform: read + write

    # Read bandwidth (via reduction)
    times = []
    for _ in range(20):
        hpx.gpu.synchronize()
        start = time.perf_counter()
        hpx.sum(gpu_arr)
        hpx.gpu.synchronize()
        times.append(time.perf_counter() - start)

    read_time = np.mean(times)
    bytes_read = size * 8
    read_bandwidth = bytes_read / read_time / 1e9

    # Read+Write bandwidth (via element-wise operation)
    gpu_arr2 = hpx.zeros_like(gpu_arr)
    times = []
    for _ in range(20):
        hpx.gpu.synchronize()
        start = time.perf_counter()
        hpx.add(gpu_arr, gpu_arr, out=gpu_arr2)
        hpx.gpu.synchronize()
        times.append(time.perf_counter() - start)

    rw_time = np.mean(times)
    bytes_rw = size * 8 * 3  # 2 reads + 1 write
    rw_bandwidth = bytes_rw / rw_time / 1e9

    return {
        'read_bandwidth_gbps': read_bandwidth,
        'read_write_bandwidth_gbps': rw_bandwidth,
        'array_size_gb': bytes_read / 1e9,
    }
```

### Memory Benchmarks

```python
# benchmarks/memory/bench_memory_usage.py
"""
Memory usage profiling for HPXPy operations.
"""
import tracemalloc
import numpy as np
import hpxpy as hpx

def profile_memory_usage():
    """Profile peak memory usage for various operations."""

    results = []
    sizes = [1_000_000, 10_000_000, 100_000_000]

    for size in sizes:
        # Array creation memory
        tracemalloc.start()
        arr = hpx.zeros((size,))
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        expected_bytes = size * 8  # float64
        overhead = peak - expected_bytes

        results.append({
            'operation': 'array_creation',
            'size': size,
            'peak_mb': peak / 1024 / 1024,
            'expected_mb': expected_bytes / 1024 / 1024,
            'overhead_mb': overhead / 1024 / 1024,
            'overhead_pct': (overhead / expected_bytes) * 100,
        })

        # Operation memory (should be minimal for in-place)
        arr = hpx.array(np.random.randn(size))
        tracemalloc.start()
        _ = hpx.sum(arr)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            'operation': 'reduction',
            'size': size,
            'peak_mb': peak / 1024 / 1024,
            'input_mb': size * 8 / 1024 / 1024,
        })

    return results


def profile_memory_bandwidth():
    """
    Measure effective memory bandwidth.
    Useful for understanding if we're compute or memory bound.
    """
    import time

    # Large array to ensure memory-bound behavior
    size = 100_000_000
    np_arr = np.random.randn(size)
    hpx_arr = hpx.from_numpy(np_arr)

    # Sum is a memory bandwidth benchmark (reads all data, minimal compute)
    start = time.perf_counter()
    hpx.sum(hpx_arr, policy=hpx.execution.par)
    elapsed = time.perf_counter() - start

    bytes_processed = size * 8  # float64
    bandwidth_gbps = (bytes_processed / elapsed) / 1e9

    return {
        'size_gb': bytes_processed / 1e9,
        'time_s': elapsed,
        'bandwidth_gbps': bandwidth_gbps,
    }
```

---

## Evaluation Criteria

### Performance Metrics

#### CPU Performance

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Single-thread overhead** | < 5% vs NumPy | Microbenchmark with `execution.seq` |
| **Parallel speedup** | > 0.7 * num_cores | Strong scaling study |
| **Parallel efficiency** | > 70% at 8 cores | `speedup / num_threads` |
| **Weak scaling efficiency** | > 80% | Weak scaling study |
| **Distributed efficiency** | > 60% at 4 nodes | Multi-locality benchmark |
| **Memory overhead** | < 10% vs raw data | Memory profiling |
| **GIL release coverage** | 100% parallel ops | Code audit |

#### GPU Performance

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **GPU vs CPU speedup** | > 10x for large arrays (>1M elements) | Microbenchmark comparison |
| **GPU memory bandwidth utilization** | > 70% of theoretical peak | Memory bandwidth benchmark |
| **CPU-GPU transfer overhead** | < 5% of compute time for large ops | Transfer timing |
| **Multi-GPU scaling** | > 80% efficiency at 4 GPUs | Multi-GPU benchmark |
| **GPU kernel launch latency** | < 50μs | Latency microbenchmark |
| **Hybrid CPU+GPU efficiency** | > 85% resource utilization | Hybrid execution benchmark |
| **GPU memory overhead** | < 15% vs raw data | GPU memory profiling |

#### GPU Crossover Points

| Operation | Min Size for GPU Benefit | Notes |
|-----------|-------------------------|-------|
| **Reductions** (sum, min, max) | ~50,000 elements | Transfer overhead dominates below |
| **Element-wise** (add, mul) | ~100,000 elements | Memory bandwidth limited |
| **Sort** | ~500,000 elements | GPU sort has higher constant overhead |
| **Matrix multiply** | ~500×500 matrices | cuBLAS very efficient |
| **FFT** | ~10,000 elements | cuFFT highly optimized |

### Correctness Validation

| Validation | Method | Acceptance Criteria |
|------------|--------|---------------------|
| **Numerical accuracy** | Compare vs NumPy | `rtol=1e-10` for float64 |
| **Determinism** | Repeated runs | Same results (except `par_unseq` reductions) |
| **Edge cases** | Property-based tests | No crashes, correct exceptions |
| **Empty arrays** | Unit tests | Match NumPy behavior |
| **Type preservation** | Unit tests | Output dtype matches input |
| **NaN/Inf handling** | Unit tests | Match NumPy propagation |

### Compatibility Matrix

#### Core Dependencies

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| **Python** | 3.9 | 3.11+ | 3.9, 3.10, 3.11, 3.12 |
| **NumPy** | 1.20 | 1.24+ | 1.20, 1.24, 2.0 |
| **HPX** | 1.9.0 | 1.10+ | Latest stable |
| **Compiler** | GCC 10 / Clang 12 | GCC 12+ | GCC 10-14, Clang 12-17 |
| **OS** | Linux, macOS | Linux | Ubuntu 22.04, macOS 13 |
| **MPI** (optional) | OpenMPI 4.0 | OpenMPI 4.1+ | OpenMPI, MPICH |

#### GPU Dependencies (Optional)

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| **CUDA Toolkit** | 11.0 | 12.0+ | 11.8, 12.0, 12.4 |
| **NVIDIA Driver** | 450.0 | 535.0+ | Latest stable |
| **cuBLAS** | (with CUDA) | (with CUDA) | CUDA 12.x |
| **SYCL** (Intel) | oneAPI 2023 | oneAPI 2024+ | Latest |
| **ROCm** (AMD) | 5.0 | 6.0+ | Future support |

#### GPU Hardware Support

| GPU Architecture | CUDA Compute | Support Level |
|-----------------|--------------|---------------|
| **NVIDIA Ampere** (A100, A10, RTX 30xx) | 8.0+ | Full |
| **NVIDIA Hopper** (H100) | 9.0 | Full |
| **NVIDIA Ada** (RTX 40xx) | 8.9 | Full |
| **NVIDIA Volta** (V100) | 7.0 | Full |
| **NVIDIA Turing** (RTX 20xx) | 7.5 | Full |
| **NVIDIA Pascal** (GTX 10xx, P100) | 6.0-6.1 | Basic |
| **Intel Arc** (via SYCL) | - | Planned |
| **AMD Instinct** (via ROCm/SYCL) | - | Planned |

### Usability Evaluation

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| **API similarity to NumPy** | Manual review | > 90% of common operations |
| **Error messages** | User study | Clear, actionable messages |
| **Documentation coverage** | Docstring audit | 100% public API |
| **Type hints** | mypy check | Full type coverage |
| **IDE support** | Manual test | Autocomplete works |
| **Installation** | `pip install` time | < 5 minutes |

### Regression Testing

```python
# tests/regression/test_known_issues.py
"""
Regression tests for previously fixed bugs.
Each test should reference the issue number.
"""

class TestRegressions:
    def test_issue_001_sum_overflow(self, hpx_runtime):
        """
        Issue #001: Sum of large int32 arrays caused overflow.
        Fix: Use int64 accumulator internally.
        """
        arr = hpx.array([2**30, 2**30], dtype=np.int32)
        result = hpx.sum(arr)
        assert result == 2**31  # Would overflow int32

    def test_issue_002_empty_sort(self, hpx_runtime):
        """
        Issue #002: Sorting empty array caused segfault.
        Fix: Early return for empty input.
        """
        arr = hpx.array([])
        sorted_arr = hpx.sort(arr)
        assert len(sorted_arr) == 0
```

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-22.04, macos-13]
        python: ['3.9', '3.10', '3.11', '3.12']
        numpy: ['1.24', '2.0']

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}

      - name: Install HPX
        run: |
          # Install HPX from package manager or build

      - name: Install package
        run: pip install -e ".[test]"

      - name: Run unit tests
        run: pytest tests/unit -v --cov=hpxpy

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  benchmark:
    runs-on: ubuntu-22.04
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Run benchmarks
        run: |
          pytest benchmarks/micro --benchmark-json=benchmark.json

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

---

## Documentation Plan

1. **Getting Started Guide**
   - Installation
   - Basic usage
   - Comparison with NumPy

2. **User Guide**
   - Array creation and manipulation
   - Parallel algorithms
   - Distributed computing
   - Performance tuning

3. **API Reference**
   - Auto-generated from docstrings
   - Type annotations for IDE support

4. **Examples**
   - Basic operations
   - Machine learning workloads
   - Scientific computing
   - Multi-node deployment

---

## Representative Benchmark Applications

These are classic large-scale applications that will serve as:
1. **Validation** - Prove HPXPy can handle real workloads
2. **Benchmarks** - Measure performance against NumPy, Dask, CuPy
3. **Examples** - Demonstrate API usage patterns
4. **Porting targets** - Guide API design decisions

### 1. Eigenfaces (Facial Recognition via PCA)

**Category:** Linear Algebra / Machine Learning
**Data Scale:** 10K-100K images × 10K pixels
**Key Operations:** Covariance matrix, SVD/eigendecomposition, matrix multiplication

```python
# Target API usage
import hpxpy as hpx
import numpy as np

# Load face dataset (e.g., LFW, CelebA)
faces = hpx.from_numpy(load_faces())  # Shape: (n_samples, height * width)

# Center the data
mean_face = hpx.mean(faces, axis=0)
centered = faces - mean_face

# Compute covariance matrix (or use SVD directly for large datasets)
# For n_samples << n_features, compute faces @ faces.T instead
if faces.shape[0] < faces.shape[1]:
    C = hpx.matmul(centered, centered.T) / faces.shape[0]
    eigenvalues, eigenvectors = hpx.linalg.eigh(C)
    # Project back to get eigenfaces
    eigenfaces = hpx.matmul(centered.T, eigenvectors).T
else:
    C = hpx.matmul(centered.T, centered) / faces.shape[0]
    eigenvalues, eigenfaces = hpx.linalg.eigh(C)

# Keep top k eigenfaces
k = 150
top_eigenfaces = eigenfaces[-k:]  # Sorted ascending

# Project faces into eigenface space
projections = hpx.matmul(centered, top_eigenfaces.T)
```

**Why This Example:**
- Classic computer vision benchmark (Turk & Pentland, 1991)
- Tests linear algebra operations at scale
- Memory-bound (covariance) and compute-bound (SVD) phases
- Natural fit for distributed computation
- Real-world application with measurable accuracy

---

### 2. PageRank

**Category:** Graph Algorithm / Iterative Computation
**Data Scale:** 1M-1B nodes, sparse adjacency matrix
**Key Operations:** Sparse matrix-vector multiply, convergence checking, collective reduction

```python
import hpxpy as hpx

def pagerank(adjacency, damping=0.85, max_iter=100, tol=1e-6):
    n = adjacency.shape[0]

    # Normalize columns (out-degree normalization)
    out_degree = hpx.sum(adjacency, axis=0)
    out_degree = hpx.where(out_degree == 0, 1, out_degree)  # Handle dangling nodes
    M = adjacency / out_degree

    # Initial rank vector
    rank = hpx.ones(n) / n
    teleport = (1 - damping) / n

    for i in range(max_iter):
        new_rank = damping * hpx.matmul(M, rank) + teleport

        # Check convergence
        diff = hpx.sum(hpx.abs(new_rank - rank))
        if diff < tol:
            break
        rank = new_rank

    return rank
```

**Why This Example:**
- Foundational algorithm (Google, web search)
- Tests iterative convergence patterns
- Sparse operations critical for efficiency
- Naturally distributed (partition by graph structure)
- Well-known ground truth for validation

---

### 3. K-Means Clustering

**Category:** Machine Learning / Data Mining
**Data Scale:** 1M-100M points × 100-1000 features
**Key Operations:** Distance computation, argmin, centroids update, broadcast

```python
import hpxpy as hpx

def kmeans(X, k, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape

    # Initialize centroids (k-means++)
    centroids = kmeans_plusplus_init(X, k)

    for i in range(max_iter):
        # Assign points to nearest centroid (embarrassingly parallel)
        # distances: (n_samples, k)
        distances = hpx.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = hpx.argmin(distances, axis=1)

        # Update centroids
        new_centroids = hpx.zeros((k, n_features))
        counts = hpx.zeros(k)
        for j in range(k):
            mask = labels == j
            if hpx.sum(mask) > 0:
                new_centroids[j] = hpx.mean(X[mask], axis=0)
                counts[j] = hpx.sum(mask)

        # Check convergence
        shift = hpx.sum((new_centroids - centroids) ** 2)
        if shift < tol:
            break
        centroids = new_centroids

    return labels, centroids
```

**Why This Example:**
- Fundamental clustering algorithm
- Highly data-parallel (distance computation)
- Tests reduction patterns (centroid updates)
- Memory-bound for large datasets
- Easy to validate and benchmark

---

### 4. Monte Carlo Pi Estimation

**Category:** Stochastic Simulation
**Data Scale:** 1B-1T random samples
**Key Operations:** Random number generation, comparison, reduction

```python
import hpxpy as hpx

def monte_carlo_pi(n_samples, chunk_size=10_000_000):
    """Estimate Pi using Monte Carlo method with distributed sampling."""
    total_inside = 0

    # Process in chunks to manage memory
    for _ in range(n_samples // chunk_size):
        # Generate random points in [0, 1) × [0, 1)
        x = hpx.random.uniform(0, 1, chunk_size)
        y = hpx.random.uniform(0, 1, chunk_size)

        # Count points inside unit circle
        inside = hpx.sum((x**2 + y**2) <= 1)
        total_inside += inside

    # Pi ≈ 4 * (points inside circle) / (total points)
    return 4 * total_inside / n_samples
```

**Why This Example:**
- Classic embarrassingly parallel problem
- Perfect weak scaling characteristics
- Tests random number generation
- Simple validation (known result)
- Good introductory example

---

### 5. N-Body Gravitational Simulation

**Category:** Physics Simulation / HPC
**Data Scale:** 10K-1M particles
**Key Operations:** All-pairs computation (O(n²)), force reduction, time integration

```python
import hpxpy as hpx

def nbody_step(positions, velocities, masses, dt, G=6.67430e-11, softening=1e-9):
    """Single timestep of N-body simulation."""
    n = positions.shape[0]

    # Compute pairwise displacements: (n, n, 3)
    # r_ij = positions[j] - positions[i]
    r = positions[None, :, :] - positions[:, None, :]

    # Compute distances with softening: (n, n)
    dist_sq = hpx.sum(r ** 2, axis=2) + softening ** 2
    dist_cubed = dist_sq ** 1.5

    # Compute accelerations: (n, 3)
    # a_i = G * sum_j(m_j * r_ij / |r_ij|^3)
    accel = G * hpx.sum(masses[None, :, None] * r / dist_cubed[:, :, None], axis=1)

    # Leapfrog integration
    velocities = velocities + accel * dt
    positions = positions + velocities * dt

    return positions, velocities

# Time evolution
for step in range(n_steps):
    positions, velocities = nbody_step(positions, velocities, masses, dt)
```

**Why This Example:**
- Classic HPC benchmark
- Compute-intensive (O(n²) operations)
- Tests broadcasting and reduction
- GPU acceleration shows dramatic speedups
- Natural distributed decomposition

---

### 6. 2D Heat Equation (Stencil Computation)

**Category:** PDE Solver / Scientific Computing
**Data Scale:** 10K × 10K to 100K × 100K grid
**Key Operations:** Stencil (5-point Laplacian), halo exchange, time stepping

```python
import hpxpy as hpx

def heat_equation_2d(u_initial, dx, dy, dt, alpha, n_steps):
    """Solve 2D heat equation using explicit finite difference."""
    u = u_initial.copy()

    rx = alpha * dt / (dx ** 2)
    ry = alpha * dt / (dy ** 2)

    for _ in range(n_steps):
        # 5-point stencil Laplacian
        laplacian = (
            rx * (hpx.roll(u, -1, axis=0) + hpx.roll(u, 1, axis=0) - 2*u) +
            ry * (hpx.roll(u, -1, axis=1) + hpx.roll(u, 1, axis=1) - 2*u)
        )
        u = u + laplacian

        # Apply boundary conditions (Dirichlet: fixed boundaries)
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    return u
```

**Why This Example:**
- Fundamental pattern in scientific computing
- Memory-bound (low arithmetic intensity)
- Tests halo exchange for distributed arrays
- Natural 2D domain decomposition
- Easy to validate (analytical solutions exist)

---

### 7. Matrix-Matrix Multiplication (DGEMM)

**Category:** Linear Algebra Core
**Data Scale:** 10K × 10K to 100K × 100K matrices
**Key Operations:** Blocked matrix multiply, BLAS-level optimization

```python
import hpxpy as hpx

# Dense matrix multiplication benchmark
def benchmark_matmul(n):
    A = hpx.random.randn(n, n)
    B = hpx.random.randn(n, n)

    # Warm-up
    C = hpx.matmul(A, B)

    # Timed run
    start = hpx.timer()
    C = hpx.matmul(A, B)
    elapsed = hpx.timer() - start

    # Compute GFLOPS: 2*n³ operations
    gflops = 2 * n**3 / elapsed / 1e9
    return gflops

# Distributed matrix multiplication (block-cyclic)
def distributed_matmul(A, B, block_size=1024):
    """Distributed matrix multiply using Cannon's algorithm or SUMMA."""
    # HPXPy handles distribution automatically
    return hpx.matmul(A, B, distribution='block_cyclic', block_size=block_size)
```

**Why This Example:**
- Gold standard for compute benchmarks
- LINPACK/Top500 heritage
- Tests peak FLOPS achievability
- Critical for neural network training
- Well-understood performance bounds

---

### 8. Image Convolution (2D FFT)

**Category:** Signal Processing / Computer Vision
**Data Scale:** 4K-8K images, batches of 1K-10K
**Key Operations:** 2D FFT, element-wise multiply, inverse FFT

```python
import hpxpy as hpx

def fft_convolve2d(image, kernel):
    """2D convolution via FFT (faster for large kernels)."""
    # Pad kernel to image size
    padded_kernel = hpx.zeros_like(image)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel

    # FFT both
    image_fft = hpx.fft.fft2(image)
    kernel_fft = hpx.fft.fft2(padded_kernel)

    # Multiply in frequency domain
    result_fft = image_fft * kernel_fft

    # Inverse FFT
    result = hpx.fft.ifft2(result_fft).real

    return result

# Batch processing
def process_image_batch(images, kernel):
    """Process batch of images with same kernel."""
    # images: (batch, height, width)
    kernel_fft = hpx.fft.fft2(kernel, s=images.shape[1:])

    images_fft = hpx.fft.fft2(images, axes=(1, 2))
    result_fft = images_fft * kernel_fft[None, :, :]
    return hpx.fft.ifft2(result_fft, axes=(1, 2)).real
```

**Why This Example:**
- Fundamental signal processing operation
- Tests FFT implementation quality
- Memory-bandwidth sensitive
- Batch dimension for data parallelism
- Applications: image processing, CNNs, physics

---

### Implementation Priority

| Example | Phase | Complexity | Key HPXPy Features Tested |
|---------|-------|------------|---------------------------|
| Monte Carlo Pi | 2 | Low | Random, reduction |
| K-Means | 2 | Medium | Broadcasting, argmin, masked ops |
| Heat Equation | 3 | Medium | Roll/shift, distributed arrays |
| N-Body | 3 | Medium | Broadcasting, nested parallelism |
| PageRank | 4 | Medium | Sparse ops, iteration, collectives |
| DGEMM | 4 | Medium | matmul, distributed LA |
| FFT Convolution | 5 | Medium | FFT, complex numbers, batch ops |
| Eigenfaces | 5 | High | Full linear algebra stack |

### Benchmark Targets

| Application | NumPy (1 core) | HPXPy (8 cores) | HPXPy (GPU) | Goal |
|-------------|---------------|-----------------|-------------|------|
| Monte Carlo (1B) | 45s | 6s | 0.5s | >8x CPU, >90x GPU |
| K-Means (1M×100) | 12s | 2s | 0.3s | >6x CPU, >40x GPU |
| Heat 10K×10K | 8s | 1.2s | 0.1s | >6x CPU, >80x GPU |
| N-Body (10K) | 25s | 4s | 0.4s | >6x CPU, >60x GPU |
| DGEMM (10K×10K) | 180s | 25s | 2s | >7x CPU, >90x GPU |
| Eigenfaces (10K) | 45s | 8s | 1.5s | >5x CPU, >30x GPU |

---

## Success Criteria

### Core Requirements (Phases 1-4)

1. **Functionality:** All Phase 1-4 deliverables complete
2. **CPU Performance:** Within 10% of native HPX C++ performance
3. **Compatibility:** Works with NumPy 1.20+, Python 3.9+
4. **Documentation:** Complete API docs and tutorials
5. **Testing:** >90% code coverage
6. **Packaging:** Available on PyPI via `pip install hpxpy`

### GPU Support (Phase 5)

7. **GPU Detection:** Automatic runtime detection of CUDA/SYCL GPUs
8. **GPU Performance:** >10x speedup over CPU for arrays >1M elements
9. **GPU Memory:** Efficient memory management with <15% overhead
10. **Multi-GPU:** Linear scaling efficiency >80% for 4 GPUs
11. **Fallback:** Graceful degradation to CPU when GPU unavailable
12. **Hybrid Execution:** Automatic CPU+GPU workload distribution

### Extended Goals (Phase 6)

13. **SYCL Backend:** Intel/AMD GPU support via SYCL
14. **Linear Algebra:** cuBLAS/cuSOLVER integration for matrix ops
15. **Ecosystem:** CuPy interoperability for GPU arrays

---

## References

### Core Documentation
- [HPX Documentation](https://hpx-docs.stellar-group.org/)
- [HPX GitHub](https://github.com/STEllAR-GROUP/hpx)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [NumPy C API](https://numpy.org/doc/stable/reference/c-api/)
- [NumPy Array API Standard](https://data-apis.org/array-api/latest/)

### Similar Projects
- [mpi4py](https://mpi4py.readthedocs.io/) - MPI for Python
- [Dask](https://dask.org/) - Distributed array computing
- [CuPy](https://cupy.dev/) - NumPy-compatible GPU arrays
- [JAX](https://jax.readthedocs.io/) - Composable transformations with GPU

### GPU Resources
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [HPX CUDA Integration](https://hpx-docs.stellar-group.org/latest/html/manual/hpx_runtime_and_resources.html)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [SYCL Specification](https://registry.khronos.org/SYCL/)

### Benchmarking
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Airspeed Velocity (asv)](https://asv.readthedocs.io/)
- [STREAM Benchmark](https://www.cs.virginia.edu/stream/) - Memory bandwidth reference
