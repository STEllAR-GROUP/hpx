# HPXPy

High-performance distributed NumPy-like arrays powered by [HPX](https://hpx.stellar-group.org/).

## Overview

HPXPy provides a NumPy-compatible interface for parallel and distributed array computing using the HPX C++ runtime system. It enables Python users to leverage HPX's parallel algorithms and distributed computing capabilities with familiar NumPy syntax.

## Features

- **NumPy-compatible API**: Familiar array creation and manipulation functions
- **Parallel execution**: Automatic parallelization using HPX's execution policies
- **Distributed computing**: Scale from single node to clusters
- **Zero-copy interoperability**: Efficient data exchange with NumPy
- **GPU support**: Optional CUDA/SYCL acceleration

## Quick Start

```python
import hpxpy as hpx

# Initialize the HPX runtime
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

# Clean shutdown
hpx.finalize()
```

## Building from Source (Development)

HPXPy can be built and used directly from the HPX build directory without requiring HPX to be installed system-wide.

### Prerequisites

- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+, or Apple Clang 12+)
- Python 3.9+ with development headers
- Boost 1.71+
- hwloc (optional but recommended)

**macOS:**
```bash
brew install cmake boost hwloc python
```

**Ubuntu/Debian:**
```bash
sudo apt install cmake libboost-all-dev libhwloc-dev python3-dev python3-pip
```

### Build Instructions

```bash
# Clone and build HPX
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHPX_WITH_FETCH_ASIO=ON -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_TESTS=OFF
make -j8

# Build HPXPy (from hpx/build directory)
cd ../python
mkdir build && cd build
cmake .. -DHPX_DIR=../../build/lib/cmake/HPX
make -j8

# Set up environment
cat > setup_env.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export HPX_BUILD="$HPX_ROOT/build"
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$HPX_BUILD/lib:$DYLD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$HPX_BUILD/lib:$LD_LIBRARY_PATH"
fi
export PYTHONPATH="$SCRIPT_DIR:$HPX_ROOT/python:$PYTHONPATH"
echo "HPXPy environment configured"
EOF
chmod +x setup_env.sh

# Test
source setup_env.sh
python3 -c "import hpxpy as hpx; hpx.init(); print(hpx.reduce(hpx.arange(100))); hpx.finalize()"
```

### GPU Support (Optional)

For CUDA support (requires HPX built with `HPX_WITH_CUDA=ON`):
```bash
cmake .. -DHPXPY_WITH_CUDA=ON -DHPX_DIR=../../build/lib/cmake/HPX
```

For SYCL support (requires HPX built with `HPX_WITH_SYCL=ON`):
```bash
cmake .. -DHPXPY_WITH_SYCL=ON -DHPX_DIR=../../build/lib/cmake/HPX
```

## Documentation

- **[HPXPy Documentation](docs/)** - Full documentation (build with Sphinx)
- **[Getting Started](docs/source/getting_started/)** - Installation and quick start
- **[Tutorials](tutorials/)** - Interactive Jupyter notebooks
- **[API Reference](docs/source/api/)** - Complete API documentation

## License

HPXPy is distributed under the Boost Software License, Version 1.0.
See [LICENSE_1_0.txt](../LICENSE_1_0.txt) for details.
