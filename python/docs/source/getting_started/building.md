# Building HPXPy from Source

This guide covers building HPXPy directly from the HPX repository without requiring HPX to be installed system-wide. This is the recommended workflow for development.

## Prerequisites

- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+, or Apple Clang 12+)
- Python 3.9+ with development headers
- Boost 1.71+ (headers and libraries)
- hwloc (optional but recommended)

### macOS

```bash
brew install cmake boost hwloc python
```

### Ubuntu/Debian

```bash
sudo apt install cmake libboost-all-dev libhwloc-dev python3-dev python3-pip
```

## Build Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx
```

### Step 2: Build HPX

Create a build directory and configure HPX:

```bash
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHPX_WITH_FETCH_ASIO=ON \
    -DHPX_WITH_EXAMPLES=OFF \
    -DHPX_WITH_TESTS=OFF
```

Build HPX (this takes a while):

```bash
make -j$(nproc)  # Linux
make -j$(sysctl -n hw.ncpu)  # macOS
```

**Important:** Do NOT run `make install`. We'll use HPX directly from the build directory.

### Step 3: Build HPXPy

From the repository root, build the Python extension:

```bash
cd ../python
mkdir build && cd build

# Point to the HPX build directory
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHPX_DIR=../../build/lib/cmake/HPX

make -j$(nproc)
```

This creates `_core.cpython-*.so` in the build directory.

### Step 4: Set Up the Environment

Create a setup script to configure the environment:

```bash
# From hpx/python/build directory
cat > setup_env.sh << 'EOF'
#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# HPX build directory
export HPX_BUILD="$HPX_ROOT/build"

# Add HPX libraries to library path
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$HPX_BUILD/lib:$DYLD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$HPX_BUILD/lib:$LD_LIBRARY_PATH"
fi

# Add hpxpy to Python path
export PYTHONPATH="$SCRIPT_DIR:$HPX_ROOT/python:$PYTHONPATH"

echo "HPXPy environment configured:"
echo "  HPX_BUILD=$HPX_BUILD"
echo "  PYTHONPATH includes: $HPX_ROOT/python"
EOF

chmod +x setup_env.sh
```

### Step 5: Use HPXPy

Source the environment and run Python:

```bash
source setup_env.sh
python3
```

```python
>>> import hpxpy as hpx
>>> hpx.init()
>>> arr = hpx.arange(100)
>>> print(hpx.reduce(arr))
4950
>>> hpx.finalize()
```

## Quick Reference

From a fresh clone:

```bash
# Build HPX
cd hpx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHPX_WITH_FETCH_ASIO=ON -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_TESTS=OFF
make -j8

# Build HPXPy
cd ../python
mkdir build && cd build
cmake .. -DHPX_DIR=../../build/lib/cmake/HPX
make -j8

# Set up environment
source setup_env.sh  # (create this script first - see above)

# Test
python3 -c "import hpxpy as hpx; hpx.init(); print(hpx.reduce(hpx.arange(100))); hpx.finalize()"
```

## Directory Structure After Build

```
hpx/
├── build/                      # HPX build directory
│   ├── lib/
│   │   ├── cmake/HPX/          # HPX CMake config files
│   │   ├── libhpx.so           # HPX shared libraries
│   │   └── ...
│   └── ...
├── python/
│   ├── build/                  # HPXPy build directory
│   │   ├── _core.cpython-*.so  # Compiled Python extension
│   │   └── setup_env.sh        # Environment setup script
│   ├── hpxpy/                  # Python package
│   │   ├── __init__.py
│   │   └── ...
│   └── ...
└── ...
```

## Running Jupyter Notebooks

After setting up the environment:

```bash
source python/build/setup_env.sh
cd python/tutorials
jupyter lab
```

The notebooks will be able to import and use `hpxpy`.

## Troubleshooting

### "Module not found: hpxpy"

Ensure `PYTHONPATH` includes both:
- The hpxpy build directory (contains `_core.*.so`)
- The hpxpy source directory (contains `hpxpy/__init__.py`)

```bash
export PYTHONPATH=/path/to/hpx/python/build:/path/to/hpx/python:$PYTHONPATH
```

### "Library not found: libhpx"

Ensure the HPX libraries are in your library path:

```bash
# Linux
export LD_LIBRARY_PATH=/path/to/hpx/build/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/path/to/hpx/build/lib:$DYLD_LIBRARY_PATH
```

### CMake can't find HPX

Specify the exact path to HPX's CMake config:

```bash
cmake .. -DHPX_DIR=/path/to/hpx/build/lib/cmake/HPX
```

### Build errors about missing Boost

Ensure Boost is installed and discoverable:

```bash
cmake .. -DBOOST_ROOT=/path/to/boost -DHPX_DIR=...
```

## GPU Support (Optional)

### CUDA

```bash
cmake .. \
    -DHPX_DIR=../../build/lib/cmake/HPX \
    -DHPXPY_WITH_CUDA=ON
```

Requires CUDA toolkit and HPX built with `HPX_WITH_CUDA=ON`.

### SYCL

```bash
cmake .. \
    -DHPX_DIR=../../build/lib/cmake/HPX \
    -DHPXPY_WITH_SYCL=ON
```

Requires a SYCL compiler (Intel oneAPI or AdaptiveCpp) and HPX built with `HPX_WITH_SYCL=ON`.
