# Getting Started

Welcome to HPXPy! This section will help you get up and running quickly.

## Installation

### Prerequisites

- Python 3.9 or later
- HPX library (built and installed)
- NumPy

### Quick Install

```bash
pip install hpxpy
```

### Building from Source (Development)

For development, you can build HPXPy against an HPX build directory without installing HPX system-wide:

```bash
# Clone and build HPX
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHPX_WITH_FETCH_ASIO=ON
make -j8

# Build HPXPy
cd ../python
mkdir build && cd build
cmake .. -DHPX_DIR=../../build/lib/cmake/HPX
make -j8

# Set up environment and test
source setup_env.sh
python3 -c "import hpxpy as hpx; hpx.init(); print('HPXPy works!'); hpx.finalize()"
```

See **[Building from Source](building)** for detailed instructions including:
- Environment setup scripts
- GPU support (CUDA/SYCL)
- Troubleshooting

### GPU Support

For CUDA support (requires HPX built with `HPX_WITH_CUDA=ON`):
```bash
cmake .. -DHPXPY_WITH_CUDA=ON -DHPX_DIR=../../build/lib/cmake/HPX
```

For SYCL support (requires HPX built with `HPX_WITH_SYCL=ON`):
```bash
cmake .. -DHPXPY_WITH_SYCL=ON -DHPX_DIR=../../build/lib/cmake/HPX
```

## Quickstart

```python
import hpxpy as hpx

# Always initialize the HPX runtime first
hpx.init()

# Create arrays
arr = hpx.arange(1000)
zeros = hpx.zeros((10, 10))
ones = hpx.ones(100)

# Parallel algorithms
total = hpx.reduce(arr)
doubled = hpx.transform(arr, lambda x: x * 2)

# Convert to/from NumPy
import numpy as np
np_arr = arr.to_numpy()
hpx_arr = hpx.from_numpy(np_arr)

# Always finalize when done
hpx.finalize()
```

## Core Concepts

### HPX Runtime

HPXPy requires the HPX runtime to be initialized before use:

```python
hpx.init()      # Start HPX runtime
# ... use HPXPy ...
hpx.finalize()  # Clean shutdown
```

### Execution Policies

Control how algorithms execute:

- `hpx.seq` - Sequential execution
- `hpx.par` - Parallel execution (default)
- `hpx.par_unseq` - Parallel + vectorized

```python
# Explicit parallel execution
result = hpx.reduce(arr, policy=hpx.par)
```

### Device Selection

Arrays can be created on different devices:

```python
cpu_arr = hpx.zeros(1000, device='cpu')   # CPU (default)
gpu_arr = hpx.zeros(1000, device='gpu')   # CUDA GPU
sycl_arr = hpx.zeros(1000, device='sycl') # SYCL GPU
auto_arr = hpx.zeros(1000, device='auto') # Best available
```

## Next Steps

- **[Tutorials](../tutorials/index)** - Step-by-step notebooks
- **[User Guide](../user_guide/index)** - Detailed feature documentation
- **[API Reference](../api/index)** - Complete API documentation

## Related Documentation

- **[HPX Documentation](https://hpx-docs.stellar-group.org/)** - Full HPX C++ documentation
- **[HPX Tutorials](https://hpx-docs.stellar-group.org/latest/html/manual/getting_started.html)** - HPX getting started guide
