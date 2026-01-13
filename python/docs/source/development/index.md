# Development

Guide for contributing to HPXPy.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx
```

### Build HPX with Python Support

```bash
mkdir build && cd build
cmake .. \
    -DHPX_WITH_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Install Development Dependencies

```bash
cd python
pip install -e ".[dev]"
```

## Project Structure

```
python/
├── hpxpy/                 # Python package
│   ├── __init__.py        # Main module
│   ├── gpu.py             # CUDA support
│   ├── sycl.py            # SYCL support
│   └── launcher.py        # Multi-locality launcher
├── src/bindings/          # C++ pybind11 bindings
│   ├── core_module.cpp    # Module entry point
│   ├── runtime_bindings.cpp
│   ├── array_bindings.cpp
│   ├── algorithm_bindings.cpp
│   ├── gpu_bindings.cpp
│   └── sycl_bindings.cpp
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── tutorials/             # Jupyter notebooks
├── docs/                  # Documentation (Sphinx)
└── CMakeLists.txt         # Build configuration
```

## Running Tests

```bash
# Run all tests
pytest python/tests/

# Run with coverage
pytest python/tests/ --cov=hpxpy

# Run specific test file
pytest python/tests/unit/test_array.py -v
```

## Building Documentation

```bash
cd python/docs

# Install doc dependencies
pip install -r requirements.txt

# Build HTML docs
make html

# Live preview with auto-reload
make livehtml
```

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Format with `black`

```bash
black python/hpxpy/
```

### C++

- Follow HPX coding standards
- Format with `clang-format`

```bash
clang-format -i python/src/bindings/*.cpp
```

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Commit Messages

```
[hpxpy] Brief description

Detailed description of changes.

Co-Authored-By: Your Name <email@example.com>
```

### Pull Request Checklist

- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted
- [ ] Changelog updated (if applicable)

## Related Resources

- **[HPX Contributing Guide](https://github.com/STEllAR-GROUP/hpx/blob/master/CONTRIBUTING.md)** - HPX contribution guidelines
- **[HPX GitHub](https://github.com/STEllAR-GROUP/hpx)** - Issue tracker and discussions
- **[pybind11 Documentation](https://pybind11.readthedocs.io/)** - C++/Python bindings
