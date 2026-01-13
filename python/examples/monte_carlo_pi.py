#!/usr/bin/env python3
"""
Monte Carlo Pi Estimation using HPXPy

This example demonstrates HPXPy's parallel capabilities by estimating Pi
using the Monte Carlo method. Points are randomly sampled in a unit square,
and the ratio of points falling inside the inscribed quarter circle gives
an estimate of Pi/4.

This is a classic embarrassingly parallel problem that demonstrates:
- Random number generation
- Element-wise operations
- Reduction operations
- Operator overloading

Mathematical basis:
    Area of quarter circle with radius 1 = Pi/4
    Area of unit square = 1
    Ratio = (points in circle) / (total points) ≈ Pi/4
    Therefore: Pi ≈ 4 * (points in circle) / (total points)
"""

import time
import hpxpy as hpx


def monte_carlo_pi_numpy(n_samples: int) -> float:
    """Estimate Pi using NumPy (for comparison)."""
    import numpy as np

    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)

    inside = np.sum(x**2 + y**2 <= 1)
    return 4 * inside / n_samples


def monte_carlo_pi_hpxpy(n_samples: int) -> float:
    """Estimate Pi using HPXPy parallel operations."""
    # Generate random points in unit square
    x = hpx.random.uniform(0, 1, size=n_samples)
    y = hpx.random.uniform(0, 1, size=n_samples)

    # Check which points are inside the quarter circle
    # A point (x, y) is inside if x^2 + y^2 <= 1
    distances_squared = x**2 + y**2  # Using operator overloading

    # Count points inside (distance <= 1)
    inside_mask = distances_squared <= 1  # Comparison operator

    # Convert boolean mask to float array for summing
    # Note: We convert to numpy, then back to hpxpy for sum
    inside_float = hpx.from_numpy(inside_mask.to_numpy().astype(float), copy=True)
    inside_count = hpx.sum(inside_float)

    # Pi ≈ 4 * (inside / total)
    return 4 * inside_count / n_samples


def monte_carlo_pi_chunked(n_samples: int, chunk_size: int = 1_000_000) -> float:
    """Estimate Pi with memory-efficient chunked processing."""
    total_inside = 0
    n_processed = 0

    while n_processed < n_samples:
        current_chunk = min(chunk_size, n_samples - n_processed)

        x = hpx.random.uniform(0, 1, size=current_chunk)
        y = hpx.random.uniform(0, 1, size=current_chunk)

        distances_squared = x**2 + y**2
        inside_mask = distances_squared <= 1

        # Count inside points in this chunk
        inside_float = hpx.from_numpy(inside_mask.to_numpy().astype(float), copy=True)
        total_inside += hpx.sum(inside_float)
        n_processed += current_chunk

    return 4 * total_inside / n_samples


def benchmark():
    """Run benchmark comparing NumPy and HPXPy implementations."""
    import numpy as np

    sample_sizes = [100_000, 1_000_000, 10_000_000]

    print("=" * 70)
    print("Monte Carlo Pi Estimation Benchmark")
    print("=" * 70)
    print(f"{'Samples':>12} | {'Method':>10} | {'Pi Estimate':>12} | {'Error':>10} | {'Time (s)':>10}")
    print("-" * 70)

    for n in sample_sizes:
        # NumPy
        np.random.seed(42)
        start = time.perf_counter()
        pi_numpy = monte_carlo_pi_numpy(n)
        time_numpy = time.perf_counter() - start
        error_numpy = abs(pi_numpy - np.pi)

        # HPXPy
        hpx.random.seed(42)
        start = time.perf_counter()
        pi_hpxpy = monte_carlo_pi_hpxpy(n)
        time_hpxpy = time.perf_counter() - start
        error_hpxpy = abs(pi_hpxpy - np.pi)

        print(f"{n:>12,} | {'NumPy':>10} | {pi_numpy:>12.8f} | {error_numpy:>10.6f} | {time_numpy:>10.4f}")
        print(f"{n:>12,} | {'HPXPy':>10} | {pi_hpxpy:>12.8f} | {error_hpxpy:>10.6f} | {time_hpxpy:>10.4f}")
        print("-" * 70)

    print()
    print(f"True value of Pi: {np.pi:.15f}")


def main():
    """Main entry point."""
    import numpy as np

    print("Monte Carlo Pi Estimation")
    print("=" * 50)

    # Quick demo
    n_samples = 1_000_000

    print(f"\nEstimating Pi with {n_samples:,} samples...")

    hpx.random.seed(42)
    pi_estimate = monte_carlo_pi_hpxpy(n_samples)

    print(f"Pi estimate: {pi_estimate:.8f}")
    print(f"True value:  {np.pi:.8f}")
    print(f"Error:       {abs(pi_estimate - np.pi):.8f}")

    # Run benchmark
    print("\n")
    benchmark()


if __name__ == "__main__":
    hpx.init()
    try:
        main()
    finally:
        # Note: finalize may fail in some HPX configurations
        try:
            hpx.finalize()
        except RuntimeError:
            pass
