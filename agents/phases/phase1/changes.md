# Phase 1: Foundation - Changes

## Branch Information

- **Branch:** `hpxpy/phase1-foundation`
- **Base:** `master`
- **Started:** 2026-01-12
- **Completed:** *(in progress)*

## Commit Log

*(Updated as commits are made)*

| Commit | Date | Description |
|--------|------|-------------|
| (pending) | 2026-01-12 | Initial build system setup |

## File Changes Summary

| File | Action | Lines + | Lines - | Description |
|------|--------|---------|---------|-------------|
| `python/` | Created | - | - | New directory for HPXPy package |
| `python/CMakeLists.txt` | Created | ~150 | - | Build configuration with HPX/pybind11 |
| `python/pyproject.toml` | Created | ~130 | - | Python package metadata, pytest config |
| `python/README.md` | Created | ~70 | - | Package documentation |
| `python/hpxpy/__init__.py` | Created | ~400 | - | Package initialization, Python API |
| `python/hpxpy/py.typed` | Created | 2 | - | PEP 561 marker |
| `python/src/bindings/core_module.cpp` | Created | ~50 | - | Main pybind11 module entry |
| `python/src/bindings/runtime_bindings.cpp` | Created | ~130 | - | HPX runtime init/finalize |
| `python/src/bindings/array_bindings.cpp` | Created | ~220 | - | ndarray class, creation functions |
| `python/src/bindings/algorithm_bindings.cpp` | Created | ~250 | - | Algorithms, execution policies |
| `python/tests/conftest.py` | Created | ~35 | - | Test fixtures |
| `python/tests/unit/test_runtime.py` | Created | ~65 | - | Runtime tests (7 tests) |
| `python/tests/unit/test_array.py` | Created | ~140 | - | Array tests (~25 tests) |
| `python/tests/unit/test_algorithms.py` | Created | ~230 | - | Algorithm tests (~30 tests) |

## Breaking Changes

None (initial implementation).

## Configuration Changes

### Build System

New CMake configuration added:

```cmake
# python/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(hpxpy VERSION 0.1.0 LANGUAGES CXX)

find_package(HPX REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

pybind11_add_module(_core MODULE
    src/bindings/core_module.cpp
    src/bindings/runtime_bindings.cpp
    src/bindings/array_bindings.cpp
    src/bindings/algorithm_bindings.cpp
)

target_link_libraries(_core PRIVATE HPX::hpx HPX::wrap_main)
target_include_directories(_core PRIVATE ${Python_NumPy_INCLUDE_DIRS})
```

### Python Package

```toml
# python/pyproject.toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "hpxpy"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["numpy>=1.20"]
```

### CI/CD

*(GitHub Actions workflow changes)*

## Migration Notes

Not applicable for Phase 1 (initial implementation).

## Dependencies

### New Build Dependencies

- CMake >= 3.18
- pybind11 >= 2.11
- scikit-build-core >= 0.5

### New Runtime Dependencies

- numpy >= 1.20

### New Test Dependencies

- pytest >= 7.0
- pytest-cov >= 4.0

## Design Decisions

1. **Decision:** Separate `python/` directory at repo root
   - **Rationale:** Keeps Python bindings independent from HPX core build, allows separate versioning
   - **Alternatives considered:** Integrate into main CMakeLists.txt (rejected: too intrusive)

2. **Decision:** Use scikit-build-core for Python packaging
   - **Rationale:** Modern, well-maintained, good pybind11 support, follows PEP 517/518
   - **Alternatives considered:** setuptools with cmake extension (more complex), meson-python (less mature)

3. **Decision:** Store array data in `std::vector<char>` byte buffer
   - **Rationale:** Type-agnostic storage, simple memory management, easy NumPy interop
   - **Alternatives considered:** Template class per dtype (rejected: complex binding code)

4. **Decision:** Always copy data in Phase 1 (no zero-copy)
   - **Rationale:** Simplifies ownership model, avoids lifetime issues; zero-copy can be added later
   - **Alternatives considered:** Shared memory from start (rejected: premature optimization)

5. **Decision:** Session-scoped pytest fixture for HPX runtime
   - **Rationale:** HPX init/finalize is expensive, single init per test session is more practical
   - **Alternatives considered:** Per-test init (rejected: too slow), module-scoped (less flexible)

## Issues Encountered

*(Document any issues and their resolutions)*

| Issue | Resolution |
|-------|------------|
| | |

## Code Review Notes

*(Notes from code review, if applicable)*
