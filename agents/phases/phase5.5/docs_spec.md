# HPXPy Documentation Specification

## Overview

This specification defines the documentation system for HPXPy, a Python wrapper for HPX that provides NumPy-compatible parallel arrays with GPU acceleration.

**Documentation Framework:** Sphinx with PyData theme
**Markup Language:** MyST Markdown (with RST support where needed)
**Hosting:** Read the Docs (recommended) or GitHub Pages

## Documentation Goals

1. **Accessible** - Easy to navigate for beginners and experts
2. **Complete** - Cover all public APIs with examples
3. **Current** - Auto-generated from docstrings, always up-to-date
4. **Searchable** - Full-text search across all content
5. **Notebook-friendly** - Render tutorial notebooks inline

## Site Structure

```
docs/
├── index.md                    # Landing page
├── getting_started/
│   ├── index.md               # Getting started overview
│   ├── installation.md        # Installation guide
│   ├── quickstart.md          # 5-minute quickstart
│   └── concepts.md            # Core concepts (HPX, parallelism)
├── tutorials/
│   ├── index.md               # Tutorials overview
│   ├── 01_getting_started.ipynb
│   ├── 02_parallel_algorithms.ipynb
│   ├── 03_execution_policies.ipynb
│   ├── 04_distributed_computing.ipynb
│   └── 05_gpu_acceleration.ipynb
├── user_guide/
│   ├── index.md               # User guide overview
│   ├── arrays.md              # Working with arrays
│   ├── parallel_algorithms.md # Parallel algorithms deep-dive
│   ├── execution_policies.md  # Execution policy guide
│   ├── distributed.md         # Distributed computing guide
│   ├── gpu.md                 # GPU acceleration guide
│   └── best_practices.md      # Performance tips
├── api/
│   ├── index.md               # API reference overview
│   ├── hpxpy.md               # Main module API
│   ├── hpxpy.gpu.md           # GPU module API
│   ├── hpxpy.sycl.md          # SYCL module API
│   ├── hpxpy.launcher.md      # Launcher module API
│   └── cpp/                   # C++ API (Doxygen/Breathe)
│       ├── index.md
│       └── ...
├── development/
│   ├── index.md               # Developer guide overview
│   ├── contributing.md        # How to contribute
│   ├── architecture.md        # Codebase architecture
│   ├── building.md            # Building from source
│   └── testing.md             # Running tests
├── changelog.md               # Version history
└── faq.md                     # Frequently asked questions
```

## Content Requirements

### Landing Page (index.md)

- HPXPy logo/banner
- One-line description
- Key features (4-6 bullet points)
- Quick install command
- Simple code example
- Links to Getting Started, Tutorials, API Reference
- Badges (PyPI version, build status, license)

### Getting Started Section

#### installation.md
- Prerequisites (Python version, HPX requirement)
- pip installation (when available)
- conda installation (when available)
- Building from source
  - CMake configuration options
  - CUDA/SYCL optional dependencies
- Verifying installation
- Troubleshooting common issues

#### quickstart.md
- Import and initialize HPX runtime
- Create arrays
- Run parallel algorithm
- Clean shutdown
- Complete working example (<20 lines)

#### concepts.md
- What is HPX? (brief, link to HPX docs)
- Task-based parallelism
- Futures and async execution
- Localities and distributed computing
- Executors and execution policies

### Tutorials Section

- Overview page with learning path
- Each notebook rendered with nbsphinx
- Prerequisites listed for each tutorial
- Estimated completion time
- Difficulty level indicator

### User Guide Section

Deep-dive documentation for each feature area:

#### arrays.md
- Array creation functions
- Data types supported
- Memory layout
- Interoperability with NumPy
- Array properties and methods

#### parallel_algorithms.md
- Available algorithms (reduce, transform, etc.)
- When to use parallel vs sequential
- Custom operations with lambdas
- Performance considerations

#### execution_policies.md
- `seq`, `par`, `par_unseq` explained
- Choosing the right policy
- Custom executors
- HPX thread pools

#### distributed.md
- Multi-locality concepts
- Distributed arrays
- Collective operations
- Launching multi-node jobs
- AGAS and global addressing

#### gpu.md
- CUDA backend
- SYCL backend
- Device selection (`device='auto'`)
- Async GPU operations
- Memory management
- Performance tips

### API Reference Section

Auto-generated from docstrings using sphinx-autodoc:

#### Structure per module
- Module overview
- Classes with methods
- Functions with signatures
- Type hints displayed
- Examples in docstrings rendered
- Cross-references to related functions

#### Docstring Requirements
All public functions/classes must have:
```python
def function_name(param1: type, param2: type = default) -> return_type:
    """Short one-line description.

    Longer description if needed, explaining behavior,
    edge cases, and implementation notes.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is `default`.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ExceptionType
        When this exception is raised.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> result = hpx.function_name(value)
    >>> hpx.finalize()

    See Also
    --------
    related_function : Brief description.

    Notes
    -----
    Additional implementation notes if needed.
    """
```

### Development Section

#### contributing.md
- Code of conduct
- How to report bugs
- How to request features
- Pull request process
- Code style (C++, Python)
- Commit message format

#### architecture.md
- Directory structure
- Module organization
- C++ binding structure
- Build system overview

#### building.md
- Development environment setup
- CMake options reference
- Debug vs Release builds
- Running specific tests

### Changelog

- Follow [Keep a Changelog](https://keepachangelog.com/) format
- Sections: Added, Changed, Deprecated, Removed, Fixed, Security
- Link to GitHub releases/PRs

## Theme and Styling

### PyData Sphinx Theme Configuration

```python
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "HPXPy",
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
    "github_url": "https://github.com/STEllAR-GROUP/hpx",
    "navbar_end": ["theme-switcher", "navbar-icon-links", "search-field"],
    "navigation_depth": 3,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "pygment_light_style": "default",
    "pygment_dark_style": "monokai",
}
```

### Color Scheme
- Primary: HPX blue (#1a73e8 or similar)
- Follow PyData theme defaults for consistency with ecosystem

### Code Blocks
- Syntax highlighting for Python, C++, Bash
- Copy button on code blocks
- Line numbers for longer examples

## Sphinx Extensions

Required extensions:
```python
extensions = [
    # Core
    "sphinx.ext.autodoc",           # Auto-generate API docs
    "sphinx.ext.autosummary",       # Generate summary tables
    "sphinx.ext.viewcode",          # Link to source code
    "sphinx.ext.intersphinx",       # Link to other projects

    # MyST (Markdown support)
    "myst_parser",                  # Markdown parsing
    "myst_nb",                      # Notebook support (or nbsphinx)

    # Documentation quality
    "sphinx.ext.napoleon",          # NumPy/Google docstring support
    "sphinx.ext.doctest",           # Test examples in docs
    "sphinx_copybutton",            # Copy button for code blocks

    # C++ documentation (optional)
    "breathe",                      # Doxygen integration

    # Search
    "sphinx_design",                # Cards, tabs, grids
]
```

## Intersphinx Links

Link to related project documentation:
```python
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "hpx": ("https://hpx-docs.stellar-group.org/latest/html", None),
}
```

## Build Configuration

### Minimum Requirements
- Python 3.9+
- Sphinx 7.0+
- PyData Sphinx Theme 0.14+

### Build Commands
```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
cd docs
make html

# Build and serve locally with auto-reload
sphinx-autobuild source build/html

# Check for broken links
make linkcheck

# Run doctests
make doctest
```

## Versioning

- Document current development version
- Support version switcher for releases
- Archive old versions on Read the Docs

## Accessibility

- Alt text for all images
- Descriptive link text
- Sufficient color contrast
- Keyboard navigation support
- Screen reader compatibility

## Search Engine Optimization

- Descriptive page titles
- Meta descriptions
- Canonical URLs
- Sitemap generation
- robots.txt configuration

## Quality Assurance

### Documentation CI Checks
- Build must succeed without warnings (treat warnings as errors)
- All internal links must resolve
- All code examples must be syntactically valid
- Spelling check (optional)

### Review Checklist
- [ ] All new public APIs documented
- [ ] Examples tested and working
- [ ] Cross-references added
- [ ] Changelog updated
- [ ] Screenshots current (if applicable)

## Hosting Options

### Read the Docs (Recommended)
- Free for open source
- Automatic builds on push
- Version hosting
- Search functionality
- Custom domain support

### GitHub Pages
- Free hosting
- GitHub Actions for builds
- Manual version management

## Timeline

1. **Setup** - Configure Sphinx, theme, extensions
2. **Structure** - Create directory structure and stub pages
3. **Content** - Write Getting Started and User Guide
4. **API Docs** - Ensure all docstrings complete, generate API reference
5. **Tutorials** - Integrate existing notebooks
6. **Review** - Technical review and editing
7. **Deploy** - Set up hosting and CI
