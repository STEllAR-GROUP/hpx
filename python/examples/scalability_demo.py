#!/usr/bin/env python3
"""
HPXPy Scalability Demonstration

This script demonstrates HPXPy's parallel scalability by running benchmarks
with different thread counts using HPX's thread configuration.

Usage:
    python scalability_demo.py

Note: This script spawns multiple Python processes to test different
thread configurations since HPX thread count must be set at initialization.
"""

import subprocess
import sys
import time
import os

# Worker script that runs with a specific thread count
WORKER_SCRIPT = """
import time
import sys
import os

# Suppress HPX output
os.environ['HPX_LOGLEVEL'] = '0'

n_threads = int(sys.argv[1])
n_samples = int(sys.argv[2])
operation = sys.argv[3]

# Initialize HPX with specific thread count
import hpxpy as hpx
hpx.init(num_threads=n_threads)

try:
    if operation == "monte_carlo":
        # Monte Carlo Pi - demonstrates operators, random, reduction
        hpx.random.seed(42)

        start = time.perf_counter()
        x = hpx.random.uniform(0, 1, size=n_samples)
        y = hpx.random.uniform(0, 1, size=n_samples)
        distances_squared = x**2 + y**2
        inside_mask = distances_squared <= 1
        inside_float = hpx.from_numpy(inside_mask.to_numpy().astype(float), copy=True)
        inside_count = hpx.sum(inside_float)
        pi_estimate = 4 * inside_count / n_samples
        elapsed = time.perf_counter() - start

        print(f"{n_threads},{elapsed:.6f},{pi_estimate:.8f}")

    elif operation == "element_wise":
        # Pure element-wise operations (no random overhead)
        import numpy as np
        np_arr = np.arange(n_samples, dtype=np.float64)
        arr = hpx.from_numpy(np_arr)

        start = time.perf_counter()
        # Chain of element-wise operations
        result = hpx.sqrt(arr + 1)
        result = hpx.exp(result * 0.001)
        result = hpx.sin(result)
        result = result ** 2
        _ = hpx.sum(result)
        elapsed = time.perf_counter() - start

        print(f"{n_threads},{elapsed:.6f}")

    elif operation == "reduction":
        # Reduction operations
        import numpy as np
        np_arr = np.random.randn(n_samples)
        arr = hpx.from_numpy(np_arr)

        start = time.perf_counter()
        for _ in range(10):  # Multiple reductions
            _ = hpx.sum(arr)
            _ = hpx.prod(arr[:1000])  # Small prod to avoid overflow
            _ = hpx.min(arr)
            _ = hpx.max(arr)
        elapsed = time.perf_counter() - start

        print(f"{n_threads},{elapsed:.6f}")

finally:
    try:
        hpx.finalize()
    except:
        pass
"""


def run_benchmark(n_threads, n_samples, operation):
    """Run a benchmark with specific thread count."""
    # Pass PYTHONPATH to subprocess
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(script_dir)
    build_dir = os.path.join(python_dir, "build")
    env["PYTHONPATH"] = f"{build_dir}:{python_dir}:" + env.get("PYTHONPATH", "")

    # Use venv Python if available
    venv_python = os.path.join(python_dir, ".venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    result = subprocess.run(
        [python_exe, "-c", WORKER_SCRIPT, str(n_threads), str(n_samples), operation],
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )

    if result.returncode != 0:
        print(f"Error with {n_threads} threads: {result.stderr}")
        return None

    return result.stdout.strip()


def get_cpu_count():
    """Get the number of available CPUs."""
    try:
        return os.cpu_count() or 4
    except:
        return 4


def main():
    print("=" * 70)
    print("HPXPy Scalability Demonstration")
    print("=" * 70)

    cpu_count = get_cpu_count()
    print(f"\nDetected {cpu_count} CPU cores")

    # Thread counts to test
    thread_counts = [1, 2]
    if cpu_count >= 4:
        thread_counts.append(4)
    if cpu_count >= 8:
        thread_counts.append(8)
    if cpu_count >= 16:
        thread_counts.append(16)

    # Benchmark 1: Monte Carlo Pi (mixed workload)
    print("\n" + "=" * 70)
    print("Benchmark 1: Monte Carlo Pi Estimation (50M samples)")
    print("  Tests: random generation, operators, comparison, reduction")
    print("=" * 70)

    n_samples = 50_000_000
    print(f"\n{'Threads':>8} | {'Time (s)':>10} | {'Speedup':>8} | {'Pi Estimate':>12}")
    print("-" * 50)

    base_time = None
    for n_threads in thread_counts:
        result = run_benchmark(n_threads, n_samples, "monte_carlo")
        if result:
            parts = result.split(",")
            elapsed = float(parts[1])
            pi = float(parts[2])

            if base_time is None:
                base_time = elapsed
                speedup = 1.0
            else:
                speedup = base_time / elapsed

            print(f"{n_threads:>8} | {elapsed:>10.4f} | {speedup:>8.2f}x | {pi:>12.8f}")

    # Benchmark 2: Pure element-wise operations
    print("\n" + "=" * 70)
    print("Benchmark 2: Element-wise Operations (100M elements)")
    print("  Tests: sqrt, exp, sin, power, sum")
    print("=" * 70)

    n_samples = 100_000_000
    print(f"\n{'Threads':>8} | {'Time (s)':>10} | {'Speedup':>8}")
    print("-" * 35)

    base_time = None
    for n_threads in thread_counts:
        result = run_benchmark(n_threads, n_samples, "element_wise")
        if result:
            parts = result.split(",")
            elapsed = float(parts[1])

            if base_time is None:
                base_time = elapsed
                speedup = 1.0
            else:
                speedup = base_time / elapsed

            print(f"{n_threads:>8} | {elapsed:>10.4f} | {speedup:>8.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
HPXPy uses HPX's parallel execution policies which automatically
distribute work across available threads. The speedup depends on:

1. Array size - larger arrays benefit more from parallelism
2. Operation type - compute-intensive operations scale better
3. Memory bandwidth - some operations are memory-bound

Note: Distributed parallelism (multiple processes/nodes) will be
available in Phase 3 with AGAS-backed distributed arrays.
""")


if __name__ == "__main__":
    main()
