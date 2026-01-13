#!/usr/bin/env python3
# HPXPy Multi-Locality Demo
#
# SPDX-License-Identifier: BSL-1.0

"""
Demonstrates HPXPy running across multiple localities (processes).

This example shows how to:
1. Launch multiple HPX localities from a single script
2. Use collective operations to communicate between localities
3. Coordinate work using barriers

Usage:
    # Run with 2 localities (default)
    python multi_locality_demo.py

    # Or launch manually with more localities:
    python -c "from hpxpy.launcher import launch_localities; launch_localities('multi_locality_demo.py', num_localities=4, verbose=True)"
"""

import numpy as np
import sys

# Import the launcher utilities
from hpxpy.launcher import (
    is_multi_locality_mode,
    parse_hpx_args,
    spmd_main,
)


def run_distributed_computation():
    """Run a simple distributed computation across localities."""
    import hpxpy as hpx

    # Parse command-line arguments to get HPX config
    script_args, hpx_args = parse_hpx_args()

    # Initialize HPX runtime with the provided arguments
    hpx.init(config=hpx_args)

    try:
        # Get locality information
        my_id = hpx.locality_id()
        num_locs = hpx.num_localities()

        print(f"[Locality {my_id}] Started (total: {num_locs} localities)")

        # Each locality creates some local data
        local_data = np.array([float(my_id + 1)] * 10)
        local_arr = hpx.from_numpy(local_data)

        print(f"[Locality {my_id}] Local data sum: {hpx.sum(local_arr)}")

        # Synchronize before collective operation
        hpx.barrier("before_reduce")

        # All-reduce: sum values from all localities
        global_sum = hpx.all_reduce(local_arr, op='sum')
        print(f"[Locality {my_id}] Global sum: {hpx.sum(global_sum)}")

        # Broadcast from locality 0
        if my_id == 0:
            broadcast_data = np.array([42.0, 43.0, 44.0])
        else:
            broadcast_data = np.zeros(3)

        broadcast_arr = hpx.from_numpy(broadcast_data)
        received = hpx.broadcast(broadcast_arr, root=0)
        print(f"[Locality {my_id}] Received broadcast: {received.to_numpy()}")

        # Gather to locality 0
        my_contribution = np.array([float(my_id * 100 + i) for i in range(3)])
        my_arr = hpx.from_numpy(my_contribution)
        gathered = hpx.gather(my_arr, root=0)

        if my_id == 0:
            print(f"[Locality 0] Gathered {len(gathered)} arrays:")
            for i, arr in enumerate(gathered):
                print(f"  From locality {i}: {arr}")

        # Final barrier before shutdown
        hpx.barrier("end")
        print(f"[Locality {my_id}] Done")

    finally:
        hpx.finalize()


# Use the spmd_main decorator for automatic multi-locality launch
@spmd_main(num_localities=2, verbose=True)
def main():
    """Main entry point."""
    run_distributed_computation()


if __name__ == "__main__":
    # Check if we're already in multi-locality mode (spawned by launcher)
    if is_multi_locality_mode():
        # We're a spawned locality - run the computation directly
        run_distributed_computation()
    else:
        # We're the original process - use the decorated main to launch
        main()
