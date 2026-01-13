# HPXPy

**Parallel NumPy Arrays for High Performance Computing**

HPXPy brings the power of [HPX](https://hpx-docs.stellar-group.org/) to Python, providing NumPy-compatible arrays with automatic parallelization, distributed computing, and GPU acceleration.

```{note}
HPXPy is part of the [HPX ecosystem](https://github.com/STEllAR-GROUP/hpx) - a C++ Standard Library for Parallelism and Concurrency.
```

## Key Features

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Parallel Algorithms
:text-align: center

Automatic parallelization of array operations using HPX's task-based runtime.

```python
result = hpx.reduce(arr, policy=hpx.par)
```
:::

:::{grid-item-card} Distributed Computing
:text-align: center

Scale across multiple nodes with distributed arrays and collective operations.

```python
arr = hpx.distributed_zeros(size, localities=4)
```
:::

:::{grid-item-card} GPU Acceleration
:text-align: center

CUDA and SYCL support via HPX executors for NVIDIA, Intel, AMD, and Apple GPUs.

```python
arr = hpx.zeros(1000000, device='auto')
```
:::

::::

## Quick Install

```bash
pip install hpxpy
```

Or build from source with HPX:

```bash
cd hpx/python
pip install -e .
```

## Quick Example

```python
import hpxpy as hpx

# Initialize HPX runtime
hpx.init()

# Create parallel array
arr = hpx.arange(1_000_000)

# Parallel reduction - automatically uses all CPU cores
total = hpx.reduce(arr)
print(f"Sum: {total}")

# GPU acceleration (if available)
if hpx.gpu.is_available():
    gpu_arr = hpx.zeros(1_000_000, device='gpu')
    print(f"GPU: {hpx.gpu.get_device(0).name}")

# Clean shutdown
hpx.finalize()
```

## Documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fas}`rocket` Getting Started
:link: getting_started/index
:link-type: doc

Installation guide, quickstart tutorial, and core concepts.
+++
[Get Started →](getting_started/index)
:::

:::{grid-item-card} {fas}`graduation-cap` Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step Jupyter notebooks from basics to advanced topics.
+++
[View Tutorials →](tutorials/index)
:::

:::{grid-item-card} {fas}`book` User Guide
:link: user_guide/index
:link-type: doc

In-depth documentation for all HPXPy features.
+++
[Read Guide →](user_guide/index)
:::

:::{grid-item-card} {fas}`code` API Reference
:link: api/index
:link-type: doc

Complete API documentation with examples.
+++
[Browse API →](api/index)
:::

::::

## HPX Ecosystem

HPXPy is built on top of HPX, a C++ Standard Library for Parallelism and Concurrency:

- **[HPX Documentation](https://hpx-docs.stellar-group.org/)** - Complete HPX C++ documentation
- **[HPX GitHub](https://github.com/STEllAR-GROUP/hpx)** - Source code and issue tracker
- **[STEllAR Group](https://stellar-group.org/)** - Research group behind HPX

## License

HPXPy is distributed under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt).

---

```{toctree}
:maxdepth: 2
:hidden:

getting_started/index
tutorials/index
user_guide/index
api/index
development/index
```
