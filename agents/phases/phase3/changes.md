# Phase 3: Distributed Arrays - Changes

## Overview

Phase 3 extends HPXPy with distributed array support using HPX's `partitioned_vector` for multi-locality data distribution.

## Goals

1. **partitioned_vector bindings** - Wrap HPX's distributed container
2. **Distribution policies** - Block, cyclic, and custom distribution
3. **Distributed array creation** - `zeros`, `ones`, etc. with distribution options
4. **to_numpy() gathering** - Collect distributed data to local array
5. **Partition introspection** - Query partition layout and locality info

## Key Concepts

- **Locality**: An HPX execution context (typically one per node)
- **Partition**: A chunk of array data residing on a single locality
- **Distribution Policy**: How data is divided across localities

## Files to Create/Modify

| File | Changes |
|------|---------|
| `python/src/types/distributed_array.hpp` | New: Distributed array wrapper class |
| `python/src/bindings/array_bindings.cpp` | Add distributed array support |
| `python/hpxpy/__init__.py` | Add distributed API |
| `python/tests/unit/test_distributed.py` | New: Distribution tests |

## API Additions

```python
# Distribution policies
hpx.distribution.block      # Block distribution (contiguous chunks)
hpx.distribution.cyclic     # Cyclic distribution (round-robin)

# Distributed array creation
arr = hpx.zeros((10000,), distribution='block')
arr = hpx.zeros((10000,), chunks=1000)

# Query distribution
arr.is_distributed         # True if data spans multiple localities
arr.num_partitions         # Number of partitions
arr.partition_sizes        # Size of each partition
arr.localities             # Locality IDs containing data

# Gather to local
local_arr = arr.to_numpy()  # Gathers all data to calling locality
```

## Implementation Status

### Completed

1. **Distribution Policy Infrastructure**
   - `DistributionPolicy` enum in C++ (None, Block, Cyclic)
   - `DistributionInfo` struct for array metadata
   - String-to-policy parsing for Python convenience

2. **Distribution Submodule**
   - `hpx.distribution` submodule accessible from Python
   - Policy constants exported (none, block, cyclic, local)
   - `DistributionPolicy` enum class accessible

3. **Locality Introspection**
   - `get_locality_id()` returns current locality ID
   - `get_num_localities()` returns total localities

4. **Tests**
   - 8 new tests for distribution module
   - 1 new test for zero-copy array views
   - All 128 tests pass (119 from Phase 1+2 + 9 new)

5. **Zero-Copy Array Views**
   - `from_numpy(arr, copy=False)` creates a view sharing memory with numpy
   - `to_numpy()` on views returns a view of the same data (no copying)
   - Proper lifetime management using pybind11 base object references
   - `is_view()` method to check if array references external data

### Future Work

The infrastructure is in place for:
- Actual multi-locality distributed arrays using partitioned_vector
- Distribution-aware array creation functions
- Distributed operations that partition work across localities
