#!/usr/bin/env python3
"""
HPXPy Heat Diffusion Simulation

This demo solves the 1D heat equation using explicit finite differences,
demonstrating parallel scalability and how computation would distribute
across multiple localities.

The heat equation: du/dt = alpha * d²u/dx²

This is a classic HPC benchmark because:
1. It's embarrassingly parallel for element-wise operations
2. It has natural domain decomposition for distributed computing
3. It demonstrates strong and weak scaling

Usage:
    python heat_diffusion_demo.py
"""

import subprocess
import sys
import os
import time

# Worker script that runs with specific thread count
WORKER_SCRIPT = '''
import time
import sys
import os
import numpy as np

os.environ['HPX_LOGLEVEL'] = '0'

n_threads = int(sys.argv[1])
grid_size = int(sys.argv[2])
n_steps = int(sys.argv[3])

import hpxpy as hpx
hpx.init(num_threads=n_threads)

try:
    # Physical parameters
    L = 1.0           # Domain length
    alpha = 0.01      # Thermal diffusivity
    dx = L / grid_size
    dt = 0.4 * dx * dx / alpha  # CFL condition for stability

    # Initial condition: hot spot in the middle
    x = np.linspace(0, L, grid_size)
    u = np.exp(-100 * (x - 0.5)**2)  # Gaussian pulse

    # Convert to HPXPy
    u_hpx = hpx.from_numpy(u)

    # Coefficients for stencil
    r = alpha * dt / (dx * dx)

    # Time stepping
    start = time.perf_counter()

    for step in range(n_steps):
        # Get numpy array for boundary handling
        u_np = u_hpx.to_numpy()

        # Create shifted arrays for stencil (u[i-1], u[i], u[i+1])
        u_left = np.roll(u_np, 1)
        u_right = np.roll(u_np, -1)

        # Fixed boundaries (Dirichlet: u=0 at edges)
        u_left[0] = 0
        u_right[-1] = 0

        # Convert to HPXPy for parallel computation
        u_l = hpx.from_numpy(u_left)
        u_r = hpx.from_numpy(u_right)

        # Apply stencil: u_new = u + r * (u_left - 2*u + u_right)
        u_hpx = u_hpx + r * (u_l - 2 * u_hpx + u_r)

    elapsed = time.perf_counter() - start

    # Get final state
    u_final = u_hpx.to_numpy()
    total_heat = float(hpx.sum(u_hpx))

    print(f"{n_threads},{elapsed:.6f},{total_heat:.6f}")

finally:
    try:
        hpx.finalize()
    except:
        pass
'''


def run_simulation(n_threads, grid_size, n_steps):
    """Run simulation with specific thread count."""
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(script_dir)
    build_dir = os.path.join(python_dir, "build")
    env["PYTHONPATH"] = f"{python_dir}:" + env.get("PYTHONPATH", "")

    venv_python = os.path.join(python_dir, ".venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    result = subprocess.run(
        [python_exe, "-c", WORKER_SCRIPT, str(n_threads), str(grid_size), str(n_steps)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr[:200]}")
        return None

    return result.stdout.strip()


def main():
    print("=" * 70)
    print("HPXPy Heat Diffusion Simulation")
    print("1D Heat Equation: du/dt = alpha * d²u/dx²")
    print("=" * 70)

    cpu_count = os.cpu_count() or 4

    # Test configurations
    thread_counts = [1, 2]
    if cpu_count >= 4:
        thread_counts.append(4)
    if cpu_count >= 8:
        thread_counts.append(8)

    # Simulation parameters
    grid_sizes = [100_000, 500_000, 1_000_000]
    n_steps = 100

    print(f"\nSystem: {cpu_count} CPU cores available")
    print(f"Simulation: {n_steps} time steps, explicit finite difference")

    # Strong scaling test
    print("\n" + "=" * 70)
    print("Strong Scaling Test")
    print("  (Fixed problem size, vary thread count)")
    print("=" * 70)

    grid_size = 1_000_000
    print(f"\nGrid size: {grid_size:,} points, {n_steps} steps")
    print(f"\n{'Threads':>8} | {'Time (s)':>10} | {'Speedup':>8} | {'Efficiency':>10}")
    print("-" * 50)

    base_time = None
    for n_threads in thread_counts:
        result = run_simulation(n_threads, grid_size, n_steps)
        if result:
            parts = result.split(",")
            elapsed = float(parts[1])

            if base_time is None:
                base_time = elapsed
                speedup = 1.0
            else:
                speedup = base_time / elapsed

            efficiency = speedup / n_threads * 100
            print(f"{n_threads:>8} | {elapsed:>10.4f} | {speedup:>7.2f}x | {efficiency:>9.1f}%")

    # Weak scaling test
    print("\n" + "=" * 70)
    print("Weak Scaling Test")
    print("  (Problem size scales with thread count)")
    print("=" * 70)

    base_size = 250_000
    print(f"\nBase grid: {base_size:,} points per thread, {n_steps} steps")
    print(f"\n{'Threads':>8} | {'Grid Size':>12} | {'Time (s)':>10} | {'Efficiency':>10}")
    print("-" * 55)

    base_time = None
    for n_threads in thread_counts:
        scaled_size = base_size * n_threads
        result = run_simulation(n_threads, scaled_size, n_steps)
        if result:
            parts = result.split(",")
            elapsed = float(parts[1])

            if base_time is None:
                base_time = elapsed

            # Ideal weak scaling: time stays constant
            efficiency = base_time / elapsed * 100
            print(f"{n_threads:>8} | {scaled_size:>12,} | {elapsed:>10.4f} | {efficiency:>9.1f}%")

    # Distributed computing simulation
    print("\n" + "=" * 70)
    print("Distributed Computing Projection")
    print("  (How this would scale across multiple nodes)")
    print("=" * 70)

    print("""
In a distributed HPX deployment, this simulation would:

1. Domain Decomposition:
   - Grid partitioned across N localities (nodes)
   - Each locality owns a contiguous chunk
   - Block distribution: locality i owns elements [i*N/n, (i+1)*N/n)

2. Communication Pattern:
   - Each step requires boundary exchange (ghost cells)
   - Only 2 values exchanged per locality pair
   - Communication: O(N) where N = number of localities

3. Expected Scaling:
   Localities |  Grid Size  | Communication | Speedup
   -----------|-------------|---------------|--------
        1     |   1,000,000 |      0        |   1x
        2     |   1,000,000 |      2 values |   ~2x
        4     |   1,000,000 |      6 values |   ~4x
        8     |   1,000,000 |     14 values |   ~8x
       16     |   1,000,000 |     30 values |  ~16x

4. HPX Advantages:
   - Asynchronous communication overlaps with computation
   - AGAS enables transparent data access across localities
   - Futurization allows pipelining of time steps
""")

    # Explain the distribution API connection
    print("=" * 70)
    print("Connection to HPXPy Distribution API")
    print("=" * 70)
    print("""
The Phase 3 distribution infrastructure enables this pattern:

```python
import hpxpy as hpx

# Future API (when distributed arrays are implemented):
# Create distributed grid across all localities
u = hpx.zeros((1_000_000,), distribution='block')

# Get distribution info
print(f"Localities: {hpx.distribution.get_num_localities()}")
print(f"Partitions: {u.num_partitions}")

# Operations automatically parallelize across localities
u_new = u + r * (u_left - 2 * u + u_right)

# Gather results to local numpy array
result = u.to_numpy()
```

Current status: Single-locality with thread parallelism
Future: Multi-locality with AGAS-backed distributed arrays
""")

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
