# User Guide

In-depth documentation for all HPXPy features.

## Core Functionality

### Arrays

HPXPy arrays are the foundation of the library, providing NumPy-compatible interfaces with parallel execution.

```python
import hpxpy as hpx

# Array creation
arr = hpx.zeros((100, 100))      # 2D array of zeros
arr = hpx.ones(1000)             # 1D array of ones
arr = hpx.arange(0, 100, 2)      # Even numbers 0-98
arr = hpx.linspace(0, 1, 50)     # 50 points from 0 to 1
arr = hpx.full((10, 10), 3.14)   # Array filled with value

# From NumPy
import numpy as np
arr = hpx.from_numpy(np.random.rand(1000))

# To NumPy
np_arr = arr.to_numpy()
```

### Parallel Algorithms

All algorithms automatically parallelize across available CPU cores.

```python
# Reductions
total = hpx.reduce(arr)                    # Sum
minimum = hpx.reduce(arr, op=hpx.min_op)   # Minimum
maximum = hpx.reduce(arr, op=hpx.max_op)   # Maximum

# Transforms
doubled = hpx.transform(arr, lambda x: x * 2)
squared = hpx.transform(arr, lambda x: x ** 2)

# In-place operations
hpx.for_each(arr, lambda x: print(x))
```

### Execution Policies

Control how algorithms execute:

| Policy | Description | Use Case |
|--------|-------------|----------|
| `hpx.seq` | Sequential | Debugging, small arrays |
| `hpx.par` | Parallel | Default, large arrays |
| `hpx.par_unseq` | Parallel + SIMD | Vectorizable operations |

```python
# Explicit policy
result = hpx.reduce(arr, policy=hpx.par_unseq)
```

## Advanced Features

### Distributed Computing

Scale across multiple machines with distributed arrays.

```python
# Create distributed array across 4 localities
arr = hpx.distributed_zeros(1000000, localities=4)

# Collective operations
total = hpx.all_reduce(arr)
hpx.broadcast(arr, root=0)
```

See the [Distributed Computing Tutorial](../tutorials/03_distributed_computing) for details.

### GPU Acceleration

Accelerate computations on GPUs using HPX executors.

```python
# Check GPU availability
if hpx.gpu.is_available():
    # Create GPU array
    arr = hpx.zeros(1000000, device='gpu')

    # Transfer data
    arr = hpx.gpu.from_numpy(np_data)
    result = arr.to_numpy()

# SYCL for Intel/AMD/Apple GPUs
if hpx.sycl.is_available():
    arr = hpx.zeros(1000000, device='sycl')
```

See the [GPU Acceleration Tutorial](../tutorials/05_gpu_acceleration) for details.

## Best Practices

### 1. Initialize Once

```python
# Good: Initialize once at program start
hpx.init()
# ... all HPXPy operations ...
hpx.finalize()

# Bad: Multiple init/finalize cycles
for i in range(100):
    hpx.init()  # Slow!
    hpx.finalize()
```

### 2. Use Appropriate Array Sizes

Parallel overhead means small arrays may be faster with NumPy:

```python
# For small arrays, NumPy may be faster
small = np.arange(100)  # Use NumPy

# For large arrays, HPXPy shines
large = hpx.arange(10_000_000)  # Use HPXPy
```

### 3. Minimize Transfers

```python
# Bad: Transfer every iteration
for i in range(1000):
    np_arr = hpx_arr.to_numpy()  # Slow!
    # process
    hpx_arr = hpx.from_numpy(np_arr)

# Good: Batch operations
# Process in HPXPy, transfer once at end
result = hpx_arr.to_numpy()
```

## Related Documentation

- **[HPX User Guide](https://hpx-docs.stellar-group.org/latest/html/manual/index.html)** - Complete HPX documentation
- **[HPX API Reference](https://hpx-docs.stellar-group.org/latest/html/api.html)** - HPX C++ API
