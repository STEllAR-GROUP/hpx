# HPXPy

High-performance distributed NumPy-like arrays powered by [HPX](https://hpx.stellar-group.org/).

## Overview

HPXPy provides a NumPy-compatible interface for parallel and distributed array computing using the HPX C++ runtime system. It enables Python users to leverage HPX's parallel algorithms and distributed computing capabilities with familiar NumPy syntax.

## Features

- **NumPy-compatible API**: Familiar array creation and manipulation functions
- **Parallel execution**: Automatic parallelization using HPX's execution policies
- **Distributed computing**: Scale from single node to clusters
- **Zero-copy interoperability**: Efficient data exchange with NumPy
- **GPU support**: Optional CUDA/SYCL acceleration (coming soon)

## Installation

### Prerequisites

- HPX (1.9.0 or later) installed with CMake config files
- Python 3.9 or later
- NumPy 1.20 or later
- C++17 compatible compiler (GCC 10+, Clang 12+)

### From source

```bash
# Clone the repository
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx/python

# Install build dependencies
pip install pybind11 numpy scikit-build-core

# Build and install
pip install . -C cmake.define.HPX_DIR=/path/to/hpx/lib/cmake/HPX
```

### Development install

```bash
pip install -e ".[dev]" -C cmake.define.HPX_DIR=/path/to/hpx/lib/cmake/HPX
```

## Quick Start

```python
import hpxpy as hpx

# Initialize the HPX runtime
with hpx.runtime(num_threads=4):
    # Create arrays
    a = hpx.arange(1_000_000)
    b = hpx.ones(1_000_000)

    # Parallel operations
    result = hpx.sum(a + b)
    print(f"Sum: {result}")

    # Sorting
    data = hpx.array([3, 1, 4, 1, 5, 9, 2, 6])
    sorted_data = hpx.sort(data)
```

## Documentation

- [API Reference](https://hpx-docs.stellar-group.org/latest/html/api.html)
- [User Guide](https://hpx-docs.stellar-group.org/latest/html/manual.html)
- [Examples](examples/)

## License

HPXPy is distributed under the Boost Software License, Version 1.0.
See [LICENSE_1_0.txt](../LICENSE_1_0.txt) for details.
