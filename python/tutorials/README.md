# HPXPy Tutorials

Interactive Jupyter notebooks for learning HPXPy.

## Tutorials

| Notebook | Description |
|----------|-------------|
| [01_getting_started.ipynb](01_getting_started.ipynb) | Runtime initialization, array creation, basic operations |
| [02_parallel_algorithms.ipynb](02_parallel_algorithms.ipynb) | Math functions, sorting, scans, random numbers |
| [03_distributed_computing.ipynb](03_distributed_computing.ipynb) | Collectives, distributed arrays, multi-locality |

## Prerequisites

1. HPXPy installed and built
2. JupyterLab or Jupyter Notebook

## Running the Tutorials

```bash
cd /path/to/hpx/python

# Activate virtual environment
source .venv/bin/activate

# Install Jupyter if needed
pip install jupyterlab

# Start JupyterLab
jupyter lab tutorials/
```

## Tutorial Order

The tutorials build on each other:

1. **Getting Started** - Learn the basics of HPXPy
2. **Parallel Algorithms** - Explore parallel operations
3. **Distributed Computing** - Scale to multiple processes

## Tips

- Run cells in order (some depend on previous cells)
- The HPX runtime must be initialized before using HPXPy functions
- Use `hpx.finalize()` or the `with hpx.runtime():` context manager
- Each notebook handles its own runtime lifecycle
