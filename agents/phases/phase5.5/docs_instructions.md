# HPXPy Documentation Instructions

## Overview

This guide provides step-by-step instructions for setting up, building, and maintaining the HPXPy documentation using Sphinx with the PyData theme.

## Prerequisites

- Python 3.9+
- HPXPy source code cloned
- Basic familiarity with Markdown/reStructuredText

## Quick Start

```bash
# From the repository root
cd python

# Install documentation dependencies
pip install -r docs/requirements.txt

# Build the documentation
cd docs
make html

# Open in browser
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

## Directory Structure to Create

```
python/
└── docs/
    ├── Makefile                 # Build automation
    ├── make.bat                 # Windows build script
    ├── requirements.txt         # Doc dependencies
    ├── source/
    │   ├── conf.py             # Sphinx configuration
    │   ├── index.md            # Landing page
    │   ├── _static/            # Static assets (CSS, images)
    │   │   ├── custom.css
    │   │   └── logo.svg
    │   ├── _templates/         # Custom templates
    │   ├── getting_started/
    │   ├── tutorials/
    │   ├── user_guide/
    │   ├── api/
    │   └── development/
    └── build/                   # Generated output (gitignored)
```

## Step 1: Initialize Documentation Structure

### Create the docs directory
```bash
mkdir -p python/docs/source/{_static,_templates,getting_started,tutorials,user_guide,api,development}
```

### Create requirements.txt
```
# python/docs/requirements.txt

# Core Sphinx
sphinx>=7.0
sphinx-autobuild>=2024.0

# Theme
pydata-sphinx-theme>=0.14

# Markdown support
myst-parser>=2.0
myst-nb>=1.0  # For notebook rendering

# Extensions
sphinx-copybutton>=0.5
sphinx-design>=0.5
numpydoc>=1.6

# Optional: C++ documentation
# breathe>=4.35  # Uncomment if documenting C++ API

# Optional: Link checking
# sphinx-linkcheck>=0.1
```

## Step 2: Create Sphinx Configuration

### conf.py
```python
# python/docs/source/conf.py
"""Sphinx configuration for HPXPy documentation."""

import os
import sys

# Add hpxpy to path for autodoc
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "HPXPy"
copyright = "2024, STEllAR Group"
author = "STEllAR Group"
version = "0.1"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    # Core Sphinx
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",

    # Markdown
    "myst_parser",
    "myst_nb",

    # UI enhancements
    "sphinx_copybutton",
    "sphinx_design",
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master doc
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

# Notebook execution
nb_execution_mode = "off"  # Don't execute notebooks during build
nb_execution_timeout = 300

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output configuration -----------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "HPXPy",
    },
    "github_url": "https://github.com/STEllAR-GROUP/hpx",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_depth": 3,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "STEllAR-GROUP",
    "github_repo": "hpx",
    "github_version": "master",
    "doc_path": "python/docs/source",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Copy button configuration -----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True

# -- Warnings as errors (for CI) ---------------------------------------------
# Uncomment for strict builds
# nitpicky = True
# nitpick_ignore = []
```

### Create Makefile
```makefile
# python/docs/Makefile

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml

clean:
	rm -rf $(BUILDDIR)/*

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
```

## Step 3: Create Landing Page

### index.md
```markdown
# HPXPy

**Parallel NumPy Arrays for High Performance Computing**

HPXPy brings the power of [HPX](https://hpx.stellar-group.org/) to Python,
providing NumPy-compatible arrays with:

- **Parallel algorithms** - Automatic parallelization of array operations
- **Distributed computing** - Scale across multiple nodes seamlessly
- **GPU acceleration** - CUDA and SYCL support via HPX executors
- **Async execution** - Non-blocking operations with futures

## Quick Install

```bash
pip install hpxpy
```

## Quick Example

```python
import hpxpy as hpx

# Initialize HPX runtime
hpx.init()

# Create parallel array
arr = hpx.arange(1_000_000)

# Parallel reduction
total = hpx.reduce(arr)

# GPU acceleration (if available)
gpu_arr = hpx.zeros(1_000_000, device='auto')

# Clean shutdown
hpx.finalize()
```

## Documentation

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Getting Started
:link: getting_started/index
:link-type: doc

Installation, quickstart, and core concepts.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step Jupyter notebooks.
:::

:::{grid-item-card} User Guide
:link: user_guide/index
:link-type: doc

In-depth feature documentation.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Complete API documentation.
:::

:::{grid-item-card} Development
:link: development/index
:link-type: doc

Contributing and building from source.
:::

::::

## Table of Contents

```{toctree}
:maxdepth: 2
:hidden:

getting_started/index
tutorials/index
user_guide/index
api/index
development/index
changelog
```
```

## Step 4: Link Tutorial Notebooks

Create symlinks or copy notebooks to docs:

```bash
# Option 1: Symlinks (recommended for development)
cd python/docs/source/tutorials
ln -s ../../../tutorials/01_getting_started.ipynb
ln -s ../../../tutorials/02_parallel_algorithms.ipynb
ln -s ../../../tutorials/03_execution_policies.ipynb
ln -s ../../../tutorials/04_distributed_computing.ipynb
ln -s ../../../tutorials/05_gpu_acceleration.ipynb

# Option 2: Copy during build (better for reproducibility)
# Add to Makefile or conf.py
```

### tutorials/index.md
```markdown
# Tutorials

Interactive Jupyter notebooks to learn HPXPy step by step.

| Tutorial | Topics | Level |
|----------|--------|-------|
| {doc}`01_getting_started` | Runtime, arrays, basic operations | Beginner |
| {doc}`02_parallel_algorithms` | reduce, transform, for_each | Beginner |
| {doc}`03_execution_policies` | seq, par, par_unseq, executors | Intermediate |
| {doc}`04_distributed_computing` | Multi-locality, collectives | Advanced |
| {doc}`05_gpu_acceleration` | CUDA, SYCL, device selection | Advanced |

```{toctree}
:maxdepth: 1
:hidden:

01_getting_started
02_parallel_algorithms
03_execution_policies
04_distributed_computing
05_gpu_acceleration
```
```

## Step 5: Generate API Reference

### api/index.md
```markdown
# API Reference

Complete reference for all HPXPy modules.

## Core Module

```{eval-rst}
.. currentmodule:: hpxpy

.. autosummary::
   :toctree: generated
   :recursive:

   init
   finalize
   is_initialized
   array
   zeros
   ones
   arange
   reduce
   transform
   for_each
```

## GPU Module

```{eval-rst}
.. currentmodule:: hpxpy.gpu

.. autosummary::
   :toctree: generated
   :recursive:

   is_available
   device_count
   get_devices
   zeros
   ones
   from_numpy
```

## SYCL Module

```{eval-rst}
.. currentmodule:: hpxpy.sycl

.. autosummary::
   :toctree: generated
   :recursive:

   is_available
   device_count
   get_devices
   zeros
   ones
   from_numpy
```
```

## Step 6: Set Up Read the Docs

### .readthedocs.yaml (in repo root)
```yaml
# .readthedocs.yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: python/docs/source/conf.py
  fail_on_warning: false

python:
  install:
    - requirements: python/docs/requirements.txt
    - method: pip
      path: python
      extra_requirements:
        - docs
```

## Step 7: Add to CI

### GitHub Actions (.github/workflows/docs.yml)
```yaml
name: Documentation

on:
  push:
    branches: [master]
    paths:
      - 'python/docs/**'
      - 'python/hpxpy/**'
      - '.github/workflows/docs.yml'
  pull_request:
    paths:
      - 'python/docs/**'
      - 'python/hpxpy/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r python/docs/requirements.txt
          pip install -e python/

      - name: Build documentation
        run: |
          cd python/docs
          make html

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: docs
          path: python/docs/build/html/
```

## Common Tasks

### Build documentation locally
```bash
cd python/docs
make html
```

### Live reload during writing
```bash
cd python/docs
make livehtml
# Opens browser at http://127.0.0.1:8000
```

### Check for broken links
```bash
cd python/docs
make linkcheck
```

### Clean and rebuild
```bash
cd python/docs
make clean html
```

### Build PDF (requires LaTeX)
```bash
cd python/docs
make latexpdf
```

## Docstring Style Guide

Use NumPy-style docstrings for all public APIs:

```python
def parallel_reduce(arr, op=None, init=None, policy=None):
    """Reduce array elements using a parallel algorithm.

    Performs a parallel reduction across all elements of the input array
    using the specified binary operation.

    Parameters
    ----------
    arr : array_like
        Input array to reduce.
    op : callable, optional
        Binary operation for reduction. Default is addition.
        Must be associative for parallel execution.
    init : scalar, optional
        Initial value for the reduction. Default depends on `op`.
    policy : ExecutionPolicy, optional
        Execution policy controlling parallelism.
        Default is `par` (parallel execution).

    Returns
    -------
    scalar
        Result of the reduction.

    Raises
    ------
    ValueError
        If `arr` is empty and no `init` is provided.
    RuntimeError
        If HPX runtime is not initialized.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.arange(100)
    >>> hpx.reduce(arr)
    4950
    >>> hpx.reduce(arr, op=lambda a, b: a * b, init=1)
    0
    >>> hpx.finalize()

    See Also
    --------
    transform : Apply operation to each element.
    for_each : Apply function for side effects.

    Notes
    -----
    The reduction is performed in parallel using HPX's parallel algorithms.
    The operation must be associative to guarantee correct results.
    """
```

## Troubleshooting

### "Module hpxpy not found" during build
Ensure HPXPy is installed or added to `sys.path` in `conf.py`:
```python
sys.path.insert(0, os.path.abspath("../../"))
```

### Notebooks not rendering
1. Check myst-nb is installed
2. Verify notebook paths in toctree
3. Check `nb_execution_mode` setting

### Theme not applying
1. Verify pydata-sphinx-theme is installed
2. Check `html_theme` spelling in conf.py

### Autodoc not finding functions
1. Check module is importable
2. Verify `__all__` exports in module
3. Check for import errors in module

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [PyData Theme Guide](https://pydata-sphinx-theme.readthedocs.io/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [MyST-NB (Notebooks)](https://myst-nb.readthedocs.io/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
