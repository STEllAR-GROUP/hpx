# HPXPy Implementation Instructions

This document outlines the development workflow for implementing HPXPy - Python wrappers around HPX providing a NumPy-compatible interface for high-performance distributed computing.

## Repository Structure

```
hpx/
├── agents/                          # Project documentation and tracking
│   ├── hpx_numpy_wrapper_spec.md    # Full specification
│   ├── hpxpy_instructions.md        # This file
│   └── phases/                      # Phase-specific tracking
│       ├── phase1/
│       │   ├── what_was_done.md
│       │   ├── changes.md
│       │   ├── performance_results.md
│       │   └── test_results.md
│       ├── phase2/
│       │   └── ...
│       └── ...
├── python/                          # HPXPy Python package (new)
│   ├── hpxpy/
│   │   ├── __init__.py
│   │   ├── _core.cpython-*.so       # Compiled extension
│   │   ├── array.py
│   │   ├── execution.py
│   │   └── ...
│   ├── src/
│   │   └── bindings/                # C++ pybind11 bindings
│   │       ├── core_module.cpp
│   │       ├── runtime_bindings.cpp
│   │       └── ...
│   ├── tests/
│   ├── benchmarks/
│   ├── CMakeLists.txt
│   └── pyproject.toml
```

## Branching Strategy

Each implementation phase gets its own branch:

| Phase | Branch Name | Description |
|-------|-------------|-------------|
| 1 | `hpxpy/phase1-foundation` | Runtime, basic arrays, sequential algorithms |
| 2 | `hpxpy/phase2-parallel` | Parallel execution policies, threading |
| 3 | `hpxpy/phase3-distributed` | Multi-locality, collectives, partitioned vectors |
| 4 | `hpxpy/phase4-numpy-compat` | Full NumPy API compatibility |
| 5 | `hpxpy/phase5-gpu` | CUDA/SYCL GPU support |
| 6 | `hpxpy/phase6-advanced` | Optimizations, ecosystem integration |

### Branch Workflow

```bash
# Start a new phase
git checkout master
git checkout -b hpxpy/phase1-foundation

# Work on the phase...
# Commit changes incrementally

# When phase is complete, create PR to master
gh pr create --title "HPXPy Phase 1: Foundation" --body "..."

# After merge, start next phase
git checkout master
git pull
git checkout -b hpxpy/phase2-parallel
```

## Phase Tracking Files

For each phase, maintain four tracking documents in `agents/phases/phaseN/`:

### 1. what_was_done.md

Records the implementation work completed in this phase.

```markdown
# Phase N: [Title] - What Was Done

## Summary
Brief overview of phase objectives and completion status.

## Implemented Features
- [ ] Feature 1: Description
- [ ] Feature 2: Description

## Files Created/Modified
- `path/to/file.cpp` - Description of changes
- `path/to/file.py` - Description of changes

## API Additions
List new public API functions/classes added.

## Dependencies Added
Any new dependencies required.

## Known Limitations
Limitations of current implementation.
```

### 2. changes.md

Detailed changelog of all modifications.

```markdown
# Phase N: [Title] - Changes

## Commit Log
List of commits with descriptions.

## Breaking Changes
Any changes that break backward compatibility.

## File Changes Summary
| File | Action | Lines Added | Lines Removed |
|------|--------|-------------|---------------|
| ... | Created/Modified/Deleted | ... | ... |

## Configuration Changes
Changes to build system, CI, etc.

## Migration Notes
Steps needed to update from previous phase.
```

### 3. performance_results.md

Benchmark results and performance analysis.

```markdown
# Phase N: [Title] - Performance Results

## Test Environment
- Hardware: CPU, RAM, GPU (if applicable)
- OS: Linux distribution/version
- Compiler: GCC/Clang version
- HPX version:
- Python version:

## Benchmark Results

### Microbenchmarks
| Operation | Array Size | NumPy (ms) | HPXPy (ms) | Speedup |
|-----------|------------|------------|------------|---------|
| sum | 1M | ... | ... | ... |
| ... | ... | ... | ... | ... |

### Scaling Results
| Threads | Time (ms) | Speedup | Efficiency |
|---------|-----------|---------|------------|
| 1 | ... | 1.0x | 100% |
| 2 | ... | ... | ... |
| ... | ... | ... | ... |

## Analysis
Interpretation of results, bottlenecks identified.

## Comparison to Targets
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Single-thread overhead | <5% | ... | Pass/Fail |
| ... | ... | ... | ... |
```

### 4. test_results.md

Test execution results and coverage.

```markdown
# Phase N: [Title] - Test Results

## Test Summary
| Category | Passed | Failed | Skipped | Total |
|----------|--------|--------|---------|-------|
| Unit | ... | ... | ... | ... |
| Integration | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

## Coverage Report
- Line coverage: X%
- Branch coverage: X%
- Function coverage: X%

## Test Output
```
pytest output...
```

## Failed Tests (if any)
Details on any test failures and their causes.

## New Tests Added
List of test files/functions added in this phase.
```

## Implementation Phases Overview

### Phase 1: Foundation
- HPX runtime initialization/finalization from Python
- Basic `hpx.array` wrapping NumPy arrays
- Sequential algorithm bindings (sum, min, max, sort)
- Build system setup (CMake + pybind11)

### Phase 2: Parallel Algorithms
- Execution policies (`seq`, `par`, `par_unseq`)
- GIL release during parallel operations
- Full parallel algorithm suite
- Thread pool configuration

### Phase 3: Distributed Computing
- Multi-locality support
- `partitioned_vector` bindings
- Collective operations (all_reduce, broadcast, gather, scatter)
- Distribution policies (block, cyclic)

### Phase 4: NumPy Compatibility
- Full NumPy Array API compliance
- Operator overloading (`+`, `-`, `*`, `/`, etc.)
- Broadcasting support
- Slicing and indexing
- Type promotion rules

### Phase 5: GPU Support
- CUDA executor bindings
- GPU array creation and transfers
- GPU-accelerated algorithms
- Multi-GPU support
- Hybrid CPU+GPU execution

### Phase 6: Advanced Features
- Performance optimizations
- SYCL backend
- Linear algebra (cuBLAS integration)
- Ecosystem interoperability (Dask, CuPy)

## Development Guidelines

### Code Style
- C++: Follow HPX coding standards
- Python: PEP 8, type hints required
- Use `clang-format` and `black` for formatting

### Testing Requirements
- All new features must have tests
- Maintain >90% code coverage
- Include both positive and negative tests
- Add property-based tests for algorithms

### Documentation Requirements
- All public APIs must have docstrings
- Update spec file with any design changes
- Keep phase tracking files current

### Commit Messages
```
[hpxpy] Phase N: Brief description

Detailed description of changes.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Getting Started

1. Ensure HPX is built and installed
2. Install Python development dependencies:
   ```bash
   pip install pybind11 numpy pytest pytest-benchmark hypothesis
   ```
3. Create the phase 1 branch and begin implementation
4. Update tracking files as you progress

## Resources

- [HPXPy Specification](hpx_numpy_wrapper_spec.md)
- [HPX Documentation](https://hpx-docs.stellar-group.org/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [NumPy Array API Standard](https://data-apis.org/array-api/latest/)
