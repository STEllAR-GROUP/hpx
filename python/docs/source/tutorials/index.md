# Tutorials

Interactive Jupyter notebooks to learn HPXPy step by step.

## Learning Path

::::{grid} 1
:gutter: 3

:::{grid-item-card} 1. Getting Started
:link: 01_getting_started
:link-type: doc

**Level:** Beginner | **Time:** 15 min

Learn the basics: HPX runtime, array creation, and simple operations.

Topics covered:
- Initializing HPX runtime
- Creating arrays (zeros, ones, arange)
- Basic array operations
- NumPy interoperability
:::

:::{grid-item-card} 2. Parallel Algorithms
:link: 02_parallel_algorithms
:link-type: doc

**Level:** Beginner | **Time:** 20 min

Master HPXPy's parallel algorithms for fast array processing.

Topics covered:
- reduce, transform, for_each
- Custom reduction operations
- Performance comparison with NumPy
:::

:::{grid-item-card} 3. Distributed Computing
:link: 03_distributed_computing
:link-type: doc

**Level:** Advanced | **Time:** 30 min

Scale across multiple nodes with distributed arrays.

Topics covered:
- Multi-locality concepts
- Distributed array creation
- Collective operations
- Launching multi-node jobs
:::

:::{grid-item-card} 4. GPU Acceleration
:link: 05_gpu_acceleration
:link-type: doc

**Level:** Advanced | **Time:** 30 min

Accelerate computations with CUDA and SYCL GPUs.

Topics covered:
- Device detection and selection
- GPU array creation
- Host-device transfers
- Async operations with futures
:::

::::

## Running the Tutorials

### Option 1: JupyterLab

```bash
cd hpx/python/tutorials
pip install jupyterlab
jupyter lab
```

### Option 2: VS Code

Open the tutorials folder in VS Code with the Jupyter extension installed.

### Option 3: Google Colab

Upload notebooks to Google Colab (note: HPX must be installed in the Colab environment).

## Prerequisites

Before starting the tutorials, ensure you have:

1. HPXPy installed (see [Installation](../getting_started/index.md))
2. Jupyter or JupyterLab installed
3. NumPy and Matplotlib for visualization

```bash
pip install jupyterlab numpy matplotlib
```

## Related Resources

- **[HPX Tutorials](https://hpx-docs.stellar-group.org/latest/html/manual/getting_started.html)** - HPX C++ tutorials
- **[NumPy Tutorials](https://numpy.org/doc/stable/user/absolute_beginners.html)** - NumPy basics

```{toctree}
:maxdepth: 1
:hidden:

01_getting_started
02_parallel_algorithms
03_distributed_computing
05_gpu_acceleration
```
