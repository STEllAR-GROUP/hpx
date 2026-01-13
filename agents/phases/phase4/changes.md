# Phase 4: Distributed Operations - Changes

## Overview

Phase 4 extends HPXPy with full distributed computing capabilities using HPX's collective operations and multi-locality support.

## Goals

1. **Collective operations** - all_reduce, broadcast, gather, scatter
2. **Distributed algorithms** - Operations on partitioned_vector across localities
3. **SPMD execution model** - Single Program, Multiple Data support
4. **Multi-locality deployment** - Support for running across multiple nodes

## Key Concepts

- **Collective**: Operation involving all localities (e.g., all_reduce sums values from all nodes)
- **SPMD**: Each locality runs the same code but on different data partitions
- **Barrier**: Synchronization point where all localities wait for each other

## Files to Create/Modify

| File | Changes |
|------|---------|
| `python/src/bindings/collective_bindings.cpp` | New: Collective operation bindings |
| `python/src/types/distributed_array.hpp` | Add partitioned_vector support |
| `python/hpxpy/__init__.py` | Export collective operations |
| `python/tests/unit/test_collectives.py` | New: Collective operation tests |

## API Additions

```python
# Collective operations
hpx.all_reduce(arr, op='sum')    # Reduce across all localities
hpx.broadcast(arr, root=0)       # Broadcast from root to all
hpx.gather(arr, root=0)          # Gather arrays to root
hpx.scatter(arr, root=0)         # Scatter array from root

# Barrier synchronization
hpx.barrier()                    # Wait for all localities

# Distributed array creation
arr = hpx.zeros((N,), distribution='block')  # Distributed across localities
arr = hpx.from_numpy(data, distribution='block')

# Distributed queries
arr.is_distributed              # True if spans multiple localities
arr.num_partitions              # Number of partitions
arr.local_partition             # This locality's portion
```

## Implementation Status

### Pending

1. **Collective Operations**
   - [ ] `all_reduce` - combine values from all localities
   - [ ] `broadcast` - send value from root to all
   - [ ] `gather` - collect values to root
   - [ ] `scatter` - distribute values from root

2. **Distributed Array Support**
   - [ ] partitioned_vector bindings
   - [ ] Block distribution implementation
   - [ ] Local partition access
   - [ ] Cross-locality data movement

3. **SPMD Support**
   - [ ] Barrier synchronization
   - [ ] Locality-aware execution
   - [ ] Distributed for_each

4. **Multi-Locality Deployment**
   - [ ] TCP parcelport configuration
   - [ ] Multi-process launch support
   - [ ] Locality discovery
