# API Reference

Complete API documentation for HPXPy.

## Core Module (`hpxpy`)

### Runtime Functions

| Function | Description |
|----------|-------------|
| `init()` | Initialize HPX runtime |
| `finalize()` | Shutdown HPX runtime |
| `is_initialized()` | Check if runtime is active |

### Array Creation

| Function | Description |
|----------|-------------|
| `array(data)` | Create array from data |
| `from_numpy(arr)` | Create from NumPy array |
| `zeros(shape)` | Array of zeros |
| `ones(shape)` | Array of ones |
| `empty(shape)` | Uninitialized array |
| `full(shape, value)` | Array filled with value |
| `arange(start, stop, step)` | Evenly spaced values |
| `linspace(start, stop, num)` | Linear spacing |

All creation functions support the `device` parameter:
- `device='cpu'` - CPU array (default)
- `device='gpu'` - CUDA GPU array
- `device='sycl'` - SYCL GPU array
- `device='auto'` - Best available device

### Parallel Algorithms

| Function | Description |
|----------|-------------|
| `reduce(arr, op, init)` | Parallel reduction |
| `transform(arr, func)` | Apply function to elements |
| `for_each(arr, func)` | Apply function for side effects |
| `sort(arr)` | Parallel sort |
| `copy(src, dst)` | Parallel copy |
| `fill(arr, value)` | Fill with value |

### Execution Policies

| Policy | Description |
|--------|-------------|
| `seq` | Sequential execution |
| `par` | Parallel execution |
| `par_unseq` | Parallel + vectorized |

## GPU Module (`hpxpy.gpu`)

CUDA GPU support via HPX cuda_executor.

### Device Management

| Function | Description |
|----------|-------------|
| `is_available()` | Check CUDA availability |
| `device_count()` | Number of GPUs |
| `get_devices()` | List all GPU info |
| `get_device(id)` | Get specific GPU info |
| `current_device()` | Current GPU ID |
| `set_device(id)` | Set current GPU |
| `synchronize()` | Synchronize GPU |
| `memory_info()` | Get memory usage |

### Array Creation

| Function | Description |
|----------|-------------|
| `zeros(shape, device)` | GPU array of zeros |
| `ones(shape, device)` | GPU array of ones |
| `full(shape, value, device)` | GPU array with value |
| `from_numpy(arr, device)` | Transfer from CPU |

### Async Operations

| Function | Description |
|----------|-------------|
| `enable_async()` | Enable HPX CUDA polling |
| `disable_async()` | Disable polling |
| `is_async_enabled()` | Check polling state |
| `AsyncContext` | Context manager for async ops |

## SYCL Module (`hpxpy.sycl`)

Cross-platform GPU support via HPX sycl_executor.

### Device Management

| Function | Description |
|----------|-------------|
| `is_available()` | Check SYCL availability |
| `device_count()` | Number of SYCL devices |
| `get_devices()` | List all device info |
| `get_device(id)` | Get specific device info |

### Array Creation

| Function | Description |
|----------|-------------|
| `zeros(shape, device)` | SYCL array of zeros |
| `ones(shape, device)` | SYCL array of ones |
| `full(shape, value, device)` | SYCL array with value |
| `from_numpy(arr, device)` | Transfer from CPU |

### Async Operations

| Function | Description |
|----------|-------------|
| `enable_async()` | Enable HPX SYCL polling |
| `disable_async()` | Disable polling |
| `AsyncContext` | Context manager for async ops |

## Distributed Module

### Distributed Arrays

| Function | Description |
|----------|-------------|
| `distributed_zeros(size, localities)` | Distributed zero array |
| `distributed_ones(size, localities)` | Distributed ones array |
| `distributed_from_numpy(arr, localities)` | Distribute NumPy array |

### Collective Operations

| Function | Description |
|----------|-------------|
| `all_reduce(arr)` | Reduce across all localities |
| `broadcast(arr, root)` | Broadcast from root |
| `gather(arr, root)` | Gather to root |
| `scatter(arr, root)` | Scatter from root |

## Launcher Module (`hpxpy.launcher`)

Multi-locality job launching.

| Function | Description |
|----------|-------------|
| `run_distributed(script, localities)` | Launch multi-locality job |
| `get_locality_id()` | Current locality ID |
| `get_num_localities()` | Total localities |

## See Also

- **[HPX C++ API](https://hpx-docs.stellar-group.org/latest/html/api.html)** - Full HPX API reference
- **[NumPy API](https://numpy.org/doc/stable/reference/)** - NumPy reference (compatible interface)
