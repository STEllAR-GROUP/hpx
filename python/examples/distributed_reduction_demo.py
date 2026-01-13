#!/usr/bin/env python3
"""
HPXPy Distributed Reduction Demo

This demo shows how collective operations enable distributed computing patterns.
It demonstrates the SPMD (Single Program, Multiple Data) execution model where
each locality runs the same code on different data partitions.

In single-locality mode, this demonstrates the API and pattern.
In multi-locality mode (when available), this would run across nodes.

Usage:
    python distributed_reduction_demo.py
"""

import os
import sys
import time
import numpy as np

# Suppress HPX output
os.environ['HPX_LOGLEVEL'] = '0'


def main():
    print("=" * 70)
    print("HPXPy Distributed Reduction Demo")
    print("SPMD Pattern: Single Program, Multiple Data")
    print("=" * 70)

    import hpxpy as hpx
    hpx.init(num_threads=4)

    try:
        # Get locality information
        num_localities = hpx.collectives.get_num_localities()
        locality_id = hpx.collectives.get_locality_id()

        print(f"\nLocality Configuration:")
        print(f"  Number of localities: {num_localities}")
        print(f"  This locality ID: {locality_id}")
        print(f"  HPX threads: {hpx.num_threads()}")

        # Demonstrate SPMD pattern
        print("\n" + "=" * 70)
        print("Demo 1: Distributed Sum (All-Reduce)")
        print("=" * 70)

        # In a real distributed scenario:
        # - Each locality would have a different portion of the data
        # - all_reduce combines all local sums into a global sum

        # Simulate local data (in multi-locality, each would have different data)
        np.random.seed(42 + locality_id)  # Different seed per locality
        local_data = np.random.randn(1000000)
        local_arr = hpx.from_numpy(local_data)

        # Compute local sum
        local_sum = float(hpx.sum(local_arr))
        print(f"\n  Local sum (locality {locality_id}): {local_sum:.4f}")

        # All-reduce to get global sum
        local_sum_arr = hpx.from_numpy(np.array([local_sum]))
        global_sum_arr = hpx.all_reduce(local_sum_arr, op='sum')
        global_sum = float(global_sum_arr.to_numpy()[0])

        print(f"  Global sum (all localities): {global_sum:.4f}")

        if num_localities == 1:
            print("\n  Note: In single-locality mode, global_sum == local_sum")
            print("  With N localities, this would sum contributions from all N")

        # Demonstrate broadcast
        print("\n" + "=" * 70)
        print("Demo 2: Parameter Broadcast")
        print("=" * 70)

        # Root locality computes some parameters
        if locality_id == 0:
            # Only root does this computation
            params = np.array([0.01, 0.99, 42.0])  # learning_rate, momentum, seed
            print(f"\n  Root computed parameters: {params}")
        else:
            params = np.zeros(3)  # Other localities wait for broadcast

        params_arr = hpx.from_numpy(params)
        params_arr = hpx.broadcast(params_arr, root=0)
        received_params = params_arr.to_numpy()

        print(f"  Locality {locality_id} received: {received_params}")

        # Demonstrate gather
        print("\n" + "=" * 70)
        print("Demo 3: Gather Local Statistics")
        print("=" * 70)

        # Each locality computes local statistics
        local_mean = float(hpx.mean(local_arr))
        local_std = float(hpx.std(local_arr))
        local_stats = np.array([local_mean, local_std])

        print(f"\n  Locality {locality_id} stats: mean={local_mean:.4f}, std={local_std:.4f}")

        local_stats_arr = hpx.from_numpy(local_stats)
        all_stats = hpx.gather(local_stats_arr, root=0)

        if locality_id == 0:
            print(f"  Root gathered {len(all_stats)} locality stats")
            for i, stats in enumerate(all_stats):
                print(f"    Locality {i}: mean={stats[0]:.4f}, std={stats[1]:.4f}")

        # Demonstrate barrier synchronization
        print("\n" + "=" * 70)
        print("Demo 4: Barrier Synchronization")
        print("=" * 70)

        print(f"\n  Locality {locality_id} reaching barrier...")
        start = time.perf_counter()
        hpx.barrier("demo_barrier")
        elapsed = time.perf_counter() - start
        print(f"  Locality {locality_id} passed barrier in {elapsed*1000:.3f} ms")

        # Show the distributed computing pattern
        print("\n" + "=" * 70)
        print("Distributed Computing Pattern")
        print("=" * 70)

        print("""
The SPMD pattern demonstrated above enables:

1. DATA PARALLELISM
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ Locality 0  │  │ Locality 1  │  │ Locality 2  │  │ Locality 3  │
   │ Data[0:N/4] │  │ Data[N/4:N/2│  │Data[N/2:3N/4│  │Data[3N/4:N] │
   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
         ↓                ↓                ↓                ↓
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ Local Comp  │  │ Local Comp  │  │ Local Comp  │  │ Local Comp  │
   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
         ↓                ↓                ↓                ↓
   ┌──────────────────────────────────────────────────────────────────┐
   │                        ALL-REDUCE                                │
   │                  Combine local results                           │
   └──────────────────────────────────────────────────────────────────┘
         ↓                ↓                ↓                ↓
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ Global Res  │  │ Global Res  │  │ Global Res  │  │ Global Res  │
   │  (same)     │  │  (same)     │  │  (same)     │  │  (same)     │
   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘

2. COLLECTIVE OPERATIONS
   - all_reduce: Combine values, result on all localities
   - broadcast:  Send from one to all
   - gather:     Collect from all to one
   - scatter:    Distribute from one to all
   - barrier:    Synchronize all localities

3. USE CASES
   - Machine Learning: Distributed gradient descent
   - Scientific Computing: Domain decomposition
   - Data Analytics: MapReduce patterns
   - Simulation: Parallel time stepping

RUNNING MULTI-LOCALITY (future):
   mpirun -n 4 python script.py --hpx:threads=8
   or
   srun -n 4 python script.py --hpx:threads=8
""")

        print("=" * 70)
        print("Demo complete!")
        print("=" * 70)

    finally:
        try:
            hpx.finalize()
        except:
            pass


if __name__ == "__main__":
    main()
