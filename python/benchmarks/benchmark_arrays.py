#!/usr/bin/env python3
# HPXPy Array Benchmarks
#
# SPDX-License-Identifier: BSL-1.0
#
# Run: python benchmarks/benchmark_arrays.py
#
# Benchmarks basic array operations and compares with NumPy.

"""HPXPy array operation benchmarks."""

import time
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    size: int
    hpxpy_time: float
    numpy_time: float

    @property
    def speedup(self) -> float:
        """HPXPy speedup vs NumPy (>1 means HPXPy is faster)."""
        if self.hpxpy_time == 0:
            return float('inf')
        return self.numpy_time / self.hpxpy_time

    def __str__(self) -> str:
        return (
            f"{self.name:20s} | {self.size:>12,} | "
            f"{self.hpxpy_time*1000:>10.3f} ms | "
            f"{self.numpy_time*1000:>10.3f} ms | "
            f"{self.speedup:>8.2f}x"
        )


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def benchmark_operation(
    name: str,
    size: int,
    hpxpy_op: Callable,
    numpy_op: Callable,
    warmup: int = 2,
    repeats: int = 5,
) -> BenchmarkResult:
    """Benchmark an operation in both HPXPy and NumPy."""
    # Warmup
    for _ in range(warmup):
        hpxpy_op()
        numpy_op()

    # Benchmark HPXPy
    hpxpy_times = []
    for _ in range(repeats):
        with timer() as t:
            hpxpy_op()
        hpxpy_times.append(t())

    # Benchmark NumPy
    numpy_times = []
    for _ in range(repeats):
        with timer() as t:
            numpy_op()
        numpy_times.append(t())

    return BenchmarkResult(
        name=name,
        size=size,
        hpxpy_time=min(hpxpy_times),
        numpy_time=min(numpy_times),
    )


def run_benchmarks():
    """Run all benchmarks."""
    try:
        import hpxpy as hpx
    except ImportError:
        print("ERROR: hpxpy not found. Make sure PYTHONPATH is set correctly.")
        sys.exit(1)

    hpx.init()
    print(f"HPXPy initialized with {hpx.num_threads()} threads\n")

    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    results = []

    print("=" * 80)
    print(f"{'Operation':20s} | {'Size':>12s} | {'HPXPy':>13s} | {'NumPy':>13s} | {'Speedup':>9s}")
    print("=" * 80)

    for size in sizes:
        # Create test arrays
        np_arr = np.arange(size, dtype=np.float64)
        hpx_arr = hpx.arange(size)

        # Benchmark: sum
        result = benchmark_operation(
            name="sum",
            size=size,
            hpxpy_op=lambda: hpx.sum(hpx_arr),
            numpy_op=lambda: np.sum(np_arr),
        )
        results.append(result)
        print(result)

        # Benchmark: element-wise multiply
        result = benchmark_operation(
            name="multiply (a * 2)",
            size=size,
            hpxpy_op=lambda: hpx_arr * 2,
            numpy_op=lambda: np_arr * 2,
        )
        results.append(result)
        print(result)

        # Benchmark: add arrays
        np_arr2 = np.ones(size, dtype=np.float64)
        hpx_arr2 = hpx.ones(size)

        result = benchmark_operation(
            name="add arrays",
            size=size,
            hpxpy_op=lambda: hpx_arr + hpx_arr2,
            numpy_op=lambda: np_arr + np_arr2,
        )
        results.append(result)
        print(result)

        # Benchmark: slicing
        result = benchmark_operation(
            name="slice [::2]",
            size=size,
            hpxpy_op=lambda: hpx_arr[::2],
            numpy_op=lambda: np_arr[::2],
        )
        results.append(result)
        print(result)

        # Benchmark: reshape
        if size >= 100:  # Need enough elements for 2D
            sqrt_size = int(size ** 0.5)
            reshape_size = sqrt_size * sqrt_size
            np_reshape_arr = np.arange(reshape_size, dtype=np.float64)
            hpx_reshape_arr = hpx.arange(reshape_size)

            result = benchmark_operation(
                name="reshape",
                size=reshape_size,
                hpxpy_op=lambda: hpx_reshape_arr.reshape((sqrt_size, sqrt_size)),
                numpy_op=lambda: np_reshape_arr.reshape((sqrt_size, sqrt_size)),
            )
            results.append(result)
            print(result)

        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_speedup = sum(r.speedup for r in results) / len(results)
    faster_count = sum(1 for r in results if r.speedup > 1)
    total_count = len(results)

    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"HPXPy faster in: {faster_count}/{total_count} benchmarks")
    print(f"Thread count: {hpx.num_threads()}")

    hpx.finalize()


if __name__ == "__main__":
    run_benchmarks()
