#!/usr/bin/env python3
"""
HPXPy K-Means Clustering Demo

This demo implements K-Means clustering using HPXPy, demonstrating:
1. Iterative MapReduce pattern (assign points -> update centroids)
2. Data-parallel operations across large datasets
3. How computation naturally partitions for distributed execution

K-Means is a classic distributed computing benchmark because:
- Data can be partitioned across localities (each owns a subset of points)
- Each iteration has a Map phase (local) and Reduce phase (global)
- Communication is minimal: only centroid updates need synchronization

Usage:
    python kmeans_clustering_demo.py
"""

import subprocess
import sys
import os
import time
import numpy as np

# Worker script for isolated HPX runtime
WORKER_SCRIPT = '''
import time
import sys
import os
import numpy as np

# Suppress all HPX output
os.environ['HPX_LOGLEVEL'] = '0'
os.environ['HPX_CONSOLE_OUTPUT'] = '0'

# Redirect stderr to suppress HPX config output
import io
sys.stderr = io.StringIO()

n_threads = int(sys.argv[1])
n_points = int(sys.argv[2])
n_clusters = int(sys.argv[3])
n_iterations = int(sys.argv[4])

import hpxpy as hpx
hpx.init(num_threads=n_threads)

try:
    # Generate clustered data (same seed for reproducibility)
    np.random.seed(42)

    # Create synthetic clusters
    points_per_cluster = n_points // n_clusters
    data_list = []
    for i in range(n_clusters):
        center = np.random.randn(2) * 10
        cluster_points = center + np.random.randn(points_per_cluster, 2)
        data_list.append(cluster_points)

    data_np = np.vstack(data_list).astype(np.float64)
    np.random.shuffle(data_np)

    # Extract x and y coordinates
    x_np = data_np[:, 0].copy()
    y_np = data_np[:, 1].copy()

    # Convert to HPXPy arrays
    x = hpx.from_numpy(x_np)
    y = hpx.from_numpy(y_np)

    # Initialize centroids (random points from data)
    np.random.seed(123)
    centroid_idx = np.random.choice(n_points, n_clusters, replace=False)
    centroids_x = x_np[centroid_idx].copy()
    centroids_y = y_np[centroid_idx].copy()

    # Warm up
    _ = hpx.sum(x)

    # Time K-Means iterations
    start = time.perf_counter()

    for iteration in range(n_iterations):
        # === MAP PHASE: Assign each point to nearest centroid ===
        # Compute distances to all centroids (vectorized)

        # For each centroid, compute squared distance from all points
        min_distances = None
        assignments = None

        for k in range(n_clusters):
            # Distance squared: (x - cx)^2 + (y - cy)^2
            dx = x - centroids_x[k]
            dy = y - centroids_y[k]
            dist_sq = dx * dx + dy * dy

            if min_distances is None:
                min_distances = dist_sq
                # Create assignment array (all zeros initially)
                assignments_np = np.zeros(n_points, dtype=np.float64)
            else:
                # Update assignments where this centroid is closer
                dist_sq_np = dist_sq.to_numpy()
                min_dist_np = min_distances.to_numpy()
                closer = dist_sq_np < min_dist_np
                assignments_np[closer] = k
                min_distances = hpx.from_numpy(np.minimum(min_dist_np, dist_sq_np))

        # === REDUCE PHASE: Update centroids ===
        # Compute new centroid positions as mean of assigned points

        for k in range(n_clusters):
            # Create mask for points in cluster k
            mask = (assignments_np == k).astype(np.float64)
            mask_hpx = hpx.from_numpy(mask)

            # Sum of coordinates for points in this cluster
            sum_x = float(hpx.sum(x * mask_hpx))
            sum_y = float(hpx.sum(y * mask_hpx))
            count = float(hpx.sum(mask_hpx))

            if count > 0:
                centroids_x[k] = sum_x / count
                centroids_y[k] = sum_y / count

    elapsed = time.perf_counter() - start

    # Compute final inertia (sum of squared distances to centroids)
    total_inertia = float(hpx.sum(min_distances))

    print(f"{n_threads},{elapsed:.6f},{total_inertia:.2f}")

finally:
    try:
        hpx.finalize()
    except:
        pass
'''


def run_kmeans(n_threads, n_points, n_clusters, n_iterations):
    """Run K-Means with specific configuration."""
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(script_dir)
    env["PYTHONPATH"] = f"{python_dir}:" + env.get("PYTHONPATH", "")
    env["HPX_LOGLEVEL"] = "0"

    venv_python = os.path.join(python_dir, ".venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    result = subprocess.run(
        [python_exe, "-c", WORKER_SCRIPT,
         str(n_threads), str(n_points), str(n_clusters), str(n_iterations)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env
    )

    if result.returncode != 0:
        # Check if we got valid output despite stderr noise
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            if ',' in line and line[0].isdigit():
                return line
        print(f"Error: {result.stderr[:200]}")
        return None

    # Extract the result line (last line with comma-separated values)
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        if ',' in line and line[0].isdigit():
            return line
    return result.stdout.strip()


def numpy_kmeans(data, n_clusters, n_iterations):
    """NumPy reference K-Means implementation."""
    n_points = len(data)
    x, y = data[:, 0], data[:, 1]

    # Initialize centroids
    np.random.seed(123)
    centroid_idx = np.random.choice(n_points, n_clusters, replace=False)
    centroids_x = data[centroid_idx, 0].copy()
    centroids_y = data[centroid_idx, 1].copy()

    start = time.perf_counter()

    for _ in range(n_iterations):
        # Compute distances to all centroids
        distances = np.zeros((n_points, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = (x - centroids_x[k])**2 + (y - centroids_y[k])**2

        # Assign to nearest centroid
        assignments = np.argmin(distances, axis=1)

        # Update centroids
        for k in range(n_clusters):
            mask = assignments == k
            if np.sum(mask) > 0:
                centroids_x[k] = np.mean(x[mask])
                centroids_y[k] = np.mean(y[mask])

    elapsed = time.perf_counter() - start

    # Compute inertia
    min_distances = np.min(distances, axis=1)
    inertia = np.sum(min_distances)

    return elapsed, inertia


def sklearn_kmeans(data, n_clusters, n_iterations):
    """Sklearn K-Means for comparison."""
    try:
        from sklearn.cluster import KMeans

        start = time.perf_counter()
        kmeans = KMeans(n_clusters=n_clusters, max_iter=n_iterations,
                       n_init=1, random_state=123, algorithm='lloyd')
        kmeans.fit(data)
        elapsed = time.perf_counter() - start

        return elapsed, kmeans.inertia_
    except ImportError:
        return None, None


def main():
    print("=" * 70)
    print("HPXPy K-Means Clustering Demo")
    print("Iterative MapReduce Pattern for Machine Learning")
    print("=" * 70)

    cpu_count = os.cpu_count() or 4

    # Configuration
    n_clusters = 10
    n_iterations = 20

    thread_counts = [1, 2]
    if cpu_count >= 4:
        thread_counts.append(4)
    if cpu_count >= 8:
        thread_counts.append(8)

    print(f"\nSystem: {cpu_count} CPU cores available")
    print(f"K-Means: {n_clusters} clusters, {n_iterations} iterations")

    # Generate test data
    print("\n" + "-" * 70)
    print("Generating synthetic clustered data...")
    print("-" * 70)

    n_points = 1_000_000
    np.random.seed(42)

    points_per_cluster = n_points // n_clusters
    data_list = []
    for i in range(n_clusters):
        center = np.random.randn(2) * 10
        cluster_points = center + np.random.randn(points_per_cluster, 2)
        data_list.append(cluster_points)

    data = np.vstack(data_list).astype(np.float64)
    np.random.shuffle(data)

    print(f"  Data points: {n_points:,}")
    print(f"  Dimensions: 2")
    print(f"  True clusters: {n_clusters}")

    # NumPy baseline
    print("\n" + "=" * 70)
    print("Baseline Comparison")
    print("=" * 70)

    np_time, np_inertia = numpy_kmeans(data, n_clusters, n_iterations)
    print(f"\nNumPy K-Means:")
    print(f"  Time: {np_time*1000:.2f} ms")
    print(f"  Inertia: {np_inertia:.2f}")

    sk_time, sk_inertia = sklearn_kmeans(data, n_clusters, n_iterations)
    if sk_time:
        print(f"\nScikit-learn K-Means:")
        print(f"  Time: {sk_time*1000:.2f} ms")
        print(f"  Inertia: {sk_inertia:.2f}")

    # HPXPy strong scaling
    print("\n" + "=" * 70)
    print("HPXPy Strong Scaling Test")
    print(f"  ({n_points:,} points, {n_clusters} clusters, {n_iterations} iterations)")
    print("=" * 70)

    print(f"\n{'Threads':>8} | {'Time (ms)':>10} | {'Speedup':>8} | {'vs NumPy':>10} | {'Inertia':>12}")
    print("-" * 65)

    base_time = None
    for n_threads in thread_counts:
        result = run_kmeans(n_threads, n_points, n_clusters, n_iterations)
        if result:
            parts = result.split(",")
            elapsed = float(parts[1])
            inertia = float(parts[2])

            if base_time is None:
                base_time = elapsed
                speedup = 1.0
            else:
                speedup = base_time / elapsed

            vs_numpy = np_time / elapsed

            print(f"{n_threads:>8} | {elapsed*1000:>10.2f} | {speedup:>7.2f}x | {vs_numpy:>9.2f}x | {inertia:>12.2f}")

    # Different data sizes
    print("\n" + "=" * 70)
    print("Scaling with Data Size (4 threads)")
    print("=" * 70)

    sizes = [100_000, 500_000, 1_000_000, 2_000_000]
    print(f"\n{'Points':>12} | {'NumPy (ms)':>12} | {'HPXPy (ms)':>12} | {'Speedup':>10}")
    print("-" * 55)

    for n_pts in sizes:
        # Generate data for this size
        np.random.seed(42)
        pts_per_cluster = n_pts // n_clusters
        test_data = []
        for i in range(n_clusters):
            center = np.random.randn(2) * 10
            cluster_points = center + np.random.randn(pts_per_cluster, 2)
            test_data.append(cluster_points)
        test_data = np.vstack(test_data).astype(np.float64)
        np.random.shuffle(test_data)

        np_t, _ = numpy_kmeans(test_data, n_clusters, n_iterations)
        result = run_kmeans(4, n_pts, n_clusters, n_iterations)

        if result:
            parts = result.split(",")
            hpx_t = float(parts[1])
            speedup = np_t / hpx_t
            print(f"{n_pts:>12,} | {np_t*1000:>12.2f} | {hpx_t*1000:>12.2f} | {speedup:>9.2f}x")

    # Distributed computing explanation
    print("\n" + "=" * 70)
    print("Distributed K-Means: How It Scales Across Nodes")
    print("=" * 70)

    print("""
K-Means has a natural MapReduce structure perfect for distribution:

ITERATION STRUCTURE:
┌─────────────────────────────────────────────────────────────────┐
│  MAP PHASE (Local - No Communication)                           │
│  ─────────────────────────────────────────                      │
│  Each locality processes its local data points:                 │
│  • Compute distance from each point to all K centroids          │
│  • Assign each point to nearest centroid                        │
│  • Compute local partial sums: Σx, Σy, count per cluster        │
│                                                                 │
│  Locality 0: points[0:N/4]     →  local_sums[0]                │
│  Locality 1: points[N/4:N/2]   →  local_sums[1]                │
│  Locality 2: points[N/2:3N/4]  →  local_sums[2]                │
│  Locality 3: points[3N/4:N]    →  local_sums[3]                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  REDUCE PHASE (Global - Minimal Communication)                  │
│  ──────────────────────────────────────────────                 │
│  All-reduce to combine partial sums:                            │
│  • global_sum_x[k] = Σ local_sum_x[k] for all localities       │
│  • global_sum_y[k] = Σ local_sum_y[k] for all localities       │
│  • global_count[k] = Σ local_count[k] for all localities       │
│                                                                 │
│  New centroid[k] = (global_sum_x[k], global_sum_y[k])          │
│                    / global_count[k]                            │
│                                                                 │
│  Communication: Only 3*K floats per locality per iteration!     │
└─────────────────────────────────────────────────────────────────┘

SCALING PROJECTION:
┌────────────┬─────────────┬───────────────┬──────────────────────┐
│ Localities │ Points/Node │ Communication │ Expected Speedup     │
├────────────┼─────────────┼───────────────┼──────────────────────┤
│     1      │  1,000,000  │      0        │        1x            │
│     4      │    250,000  │  120 floats   │       ~4x            │
│    16      │     62,500  │  480 floats   │      ~16x            │
│    64      │     15,625  │ 1920 floats   │      ~60x            │
│   256      │      3,906  │ 7680 floats   │     ~200x            │
└────────────┴─────────────┴───────────────┴──────────────────────┘

Communication overhead is O(K) per iteration, independent of data size!
This makes K-Means near-perfectly scalable for large datasets.

HPXPy FUTURE API:
```python
import hpxpy as hpx

# Distribute data across all localities
x = hpx.from_numpy(data[:, 0], distribution='block')
y = hpx.from_numpy(data[:, 1], distribution='block')

for iteration in range(max_iter):
    # MAP: Each locality computes local assignments (no communication)
    for k in range(n_clusters):
        dist_sq = (x - centroids[k, 0])**2 + (y - centroids[k, 1])**2
        # ... assign points locally

    # REDUCE: Global reduction to update centroids
    # HPX handles the all-reduce automatically
    for k in range(n_clusters):
        centroids[k, 0] = hpx.sum(x * mask[k]) / hpx.sum(mask[k])
        centroids[k, 1] = hpx.sum(y * mask[k]) / hpx.sum(mask[k])
```
""")

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
