#!/usr/bin/env python3
"""
HPXPy Distribution API Demo

This script demonstrates the Phase 3 distribution infrastructure,
showing how the API is structured for future distributed array support.

Usage:
    python distribution_demo.py
"""

import numpy as np


def main():
    print("=" * 70)
    print("HPXPy Distribution API Demo")
    print("=" * 70)

    # Initialize HPX
    import hpxpy as hpx
    hpx.init(num_threads=4)

    try:
        # Section 1: Distribution Policies
        print("\n1. Distribution Policies")
        print("-" * 40)
        print("Available distribution policies:")
        print(f"  hpx.distribution.none   = {hpx.distribution.none}")
        print(f"  hpx.distribution.block  = {hpx.distribution.block}")
        print(f"  hpx.distribution.cyclic = {hpx.distribution.cyclic}")
        print(f"  hpx.distribution.local  = {hpx.distribution.local} (alias for none)")

        # Section 2: Locality Introspection
        print("\n2. Locality Introspection")
        print("-" * 40)
        locality_id = hpx.distribution.get_locality_id()
        num_localities = hpx.distribution.get_num_localities()
        print(f"  Current locality ID: {locality_id}")
        print(f"  Number of localities: {num_localities}")

        if num_localities == 1:
            print("\n  Note: Running in single-locality mode.")
            print("  In a distributed HPX deployment, these values would")
            print("  reflect the actual cluster topology.")

        # Section 3: Policy Enum Usage
        print("\n3. Distribution Policy Enum")
        print("-" * 40)
        print("The DistributionPolicy enum can be used for type-safe policy selection:")
        print(f"  DistributionPolicy.none  = {hpx.distribution.DistributionPolicy.none}")
        print(f"  DistributionPolicy.block = {hpx.distribution.DistributionPolicy.block}")
        print(f"  DistributionPolicy.cyclic = {hpx.distribution.DistributionPolicy.cyclic}")

        # Demonstrate policy comparison
        print("\nPolicy comparison:")
        print(f"  none == local: {hpx.distribution.none == hpx.distribution.local}")
        print(f"  block == cyclic: {hpx.distribution.block == hpx.distribution.cyclic}")

        # Section 4: Existing Operations Still Work
        print("\n4. Array Operations (Unchanged)")
        print("-" * 40)
        print("All existing array operations continue to work:")

        # Create arrays
        a = hpx.arange(1000000)
        b = hpx.arange(1000000)

        # Perform operations
        import time
        start = time.perf_counter()
        c = a + b
        c = hpx.sqrt(c)
        result = hpx.sum(c)
        elapsed = time.perf_counter() - start

        print(f"  Created two arrays of 1M elements")
        print(f"  Computed: sum(sqrt(a + b))")
        print(f"  Result: {result:.2f}")
        print(f"  Time: {elapsed*1000:.2f} ms")

        # Section 5: Future Distribution API Preview
        print("\n5. Future Distribution API (Preview)")
        print("-" * 40)
        print("In future phases, distributed arrays will support:")
        print("""
  # Create distributed array across localities
  arr = hpx.zeros((10000000,), distribution='block')

  # Query distribution info
  arr.is_distributed      # True if multi-locality
  arr.num_partitions      # Number of data partitions
  arr.partition_sizes     # Size of each partition
  arr.localities          # Which localities hold data

  # Gather to single locality
  local = arr.to_numpy()  # Collects all data

  # Distributed operations
  result = hpx.sum(arr)   # Parallel reduction across localities
""")

        # Section 6: Distributed Computing Concepts
        print("6. HPX Distributed Computing Concepts")
        print("-" * 40)
        print("""
  Locality: An HPX execution context (typically one per node)
            Each locality has its own memory and OS threads.

  Partition: A contiguous chunk of array data residing on
             a single locality.

  Block Distribution: Divides array into contiguous chunks.
                      Elements [0..N/P) on locality 0,
                      [N/P..2N/P) on locality 1, etc.

  Cyclic Distribution: Round-robin element assignment.
                       Element i goes to locality (i % P).
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
