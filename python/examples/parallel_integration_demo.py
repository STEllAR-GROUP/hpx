#!/usr/bin/env python3
"""
HPXPy Parallel Numerical Integration

This demo computes definite integrals using parallel evaluation,
demonstrating strong scaling for compute-intensive workloads.

We integrate: f(x) = sin(x) * exp(-x²) * cos(10x)
This is a complex oscillating function that requires many evaluations.

Usage:
    python parallel_integration_demo.py
"""

import subprocess
import sys
import os
import time
import numpy as np

WORKER_SCRIPT = '''
import time
import sys
import os
import numpy as np

os.environ['HPX_LOGLEVEL'] = '0'

n_threads = int(sys.argv[1])
n_points = int(sys.argv[2])

import hpxpy as hpx
hpx.init(num_threads=n_threads)

try:
    # Integration bounds
    a, b = -5.0, 5.0
    dx = (b - a) / n_points

    # Create evaluation points
    x_np = np.linspace(a + dx/2, b - dx/2, n_points)
    x = hpx.from_numpy(x_np)

    # Warm up
    _ = hpx.sin(x)

    # Time the integration
    start = time.perf_counter()

    # Evaluate: f(x) = sin(x) * exp(-x²) * cos(10x)
    # Multiple chained operations to be compute-intensive
    sin_x = hpx.sin(x)
    exp_neg_x2 = hpx.exp(-(x * x))
    cos_10x = hpx.cos(x * 10)

    f = sin_x * exp_neg_x2 * cos_10x

    # Riemann sum
    integral = float(hpx.sum(f)) * dx

    elapsed = time.perf_counter() - start

    print(f"{n_threads},{elapsed:.6f},{integral:.10f}")

finally:
    try:
        hpx.finalize()
    except:
        pass
'''


def run_integration(n_threads, n_points):
    """Run integration with specific thread count."""
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(script_dir)
    env["PYTHONPATH"] = f"{python_dir}:" + env.get("PYTHONPATH", "")

    venv_python = os.path.join(python_dir, ".venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    result = subprocess.run(
        [python_exe, "-c", WORKER_SCRIPT, str(n_threads), str(n_points)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr[:200]}")
        return None

    return result.stdout.strip()


def numpy_integration(n_points):
    """NumPy reference implementation."""
    a, b = -5.0, 5.0
    dx = (b - a) / n_points
    x = np.linspace(a + dx/2, b - dx/2, n_points)

    start = time.perf_counter()
    f = np.sin(x) * np.exp(-x**2) * np.cos(10*x)
    integral = np.sum(f) * dx
    elapsed = time.perf_counter() - start

    return elapsed, integral


def main():
    print("=" * 70)
    print("HPXPy Parallel Numerical Integration")
    print("Integrating: f(x) = sin(x) * exp(-x²) * cos(10x) over [-5, 5]")
    print("=" * 70)

    cpu_count = os.cpu_count() or 4

    thread_counts = [1, 2]
    if cpu_count >= 4:
        thread_counts.append(4)
    if cpu_count >= 8:
        thread_counts.append(8)

    # Test with large problem size for better scaling
    n_points = 50_000_000  # 50 million evaluation points

    print(f"\nSystem: {cpu_count} CPU cores available")
    print(f"Integration: {n_points:,} evaluation points")

    # NumPy baseline
    print("\n" + "-" * 70)
    print("NumPy Baseline")
    print("-" * 70)
    np_time, np_integral = numpy_integration(n_points)
    print(f"  Time: {np_time*1000:.2f} ms")
    print(f"  Result: {np_integral:.10f}")

    # HPXPy scaling test
    print("\n" + "=" * 70)
    print("HPXPy Strong Scaling Test")
    print("=" * 70)

    print(f"\n{'Threads':>8} | {'Time (ms)':>10} | {'Speedup':>8} | {'vs NumPy':>10} | {'Integral':>14}")
    print("-" * 70)

    base_time = None
    for n_threads in thread_counts:
        result = run_integration(n_threads, n_points)
        if result:
            parts = result.split(",")
            elapsed = float(parts[1])
            integral = float(parts[2])

            if base_time is None:
                base_time = elapsed
                speedup = 1.0
            else:
                speedup = base_time / elapsed

            vs_numpy = np_time / elapsed

            print(f"{n_threads:>8} | {elapsed*1000:>10.2f} | {speedup:>7.2f}x | {vs_numpy:>9.2f}x | {integral:>14.10f}")

    # Different problem sizes
    print("\n" + "=" * 70)
    print("Scaling with Problem Size (4 threads)")
    print("=" * 70)

    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    print(f"\n{'Points':>15} | {'NumPy (ms)':>12} | {'HPXPy (ms)':>12} | {'Speedup':>10}")
    print("-" * 60)

    for n_pts in sizes:
        np_t, _ = numpy_integration(n_pts)
        result = run_integration(4, n_pts)
        if result:
            parts = result.split(",")
            hpx_t = float(parts[1])
            speedup = np_t / hpx_t
            print(f"{n_pts:>15,} | {np_t*1000:>12.2f} | {hpx_t*1000:>12.2f} | {speedup:>9.2f}x")

    # Distributed projection
    print("\n" + "=" * 70)
    print("Distributed Computing Projection")
    print("=" * 70)
    print("""
Numerical integration is embarrassingly parallel - perfect for distribution:

1. No Communication Required:
   - Each locality evaluates f(x) on its portion independently
   - Only final sum needs global reduction
   - Perfect weak scaling expected

2. Distribution Strategy:
   - Block distribution: locality i evaluates x in [a + i*(b-a)/N, a + (i+1)*(b-a)/N]
   - Each locality computes partial sum
   - Single reduction at the end

3. Expected Distributed Performance:
   Localities | Points/Locality | Communication | Expected Speedup
   -----------|-----------------|---------------|------------------
        1     |   50,000,000    |      0        |      1x
        4     |   12,500,000    |   1 reduce    |     ~4x
       16     |    3,125,000    |   1 reduce    |    ~16x
       64     |      781,250    |   1 reduce    |    ~64x
      256     |      195,312    |   1 reduce    |   ~200x+

4. HPXPy Future API:
   ```python
   # Distribute work across all localities
   x = hpx.linspace(a, b, n_points, distribution='block')
   f = hpx.sin(x) * hpx.exp(-x*x) * hpx.cos(10*x)
   integral = hpx.sum(f) * dx  # Automatic distributed reduction
   ```
""")

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
