# HPXPy Documentation Configuration
# Sphinx configuration file

import os
import sys
from datetime import datetime

# Add hpxpy to path for autodoc (when available)
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "HPXPy"
copyright = f"{datetime.now().year}, STEllAR Group"
author = "STEllAR Group"
version = "0.1"
release = "0.1.0-dev"

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

    # Notebooks
    "nbsphinx",

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
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# Templates
templates_path = ["_templates"]

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

myst_heading_anchors = 3

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True

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

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output configuration -----------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# GitHub context for "Edit on GitHub" links
html_context = {
    "display_github": True,
    "github_user": "STEllAR-GROUP",
    "github_repo": "hpx",
    "github_version": "master",
    "conf_py_path": "/python/docs/source/",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Favicon and logo
# html_favicon = "_static/favicon.ico"
# html_logo = "_static/logo.svg"

html_title = "HPXPy Documentation"
html_short_title = "HPXPy"

# -- Copy button configuration -----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True

# -- LaTeX output (for PDF) --------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
}

# -- Warnings ----------------------------------------------------------------
# Treat warnings as errors in CI (uncomment for strict builds)
# nitpicky = True

suppress_warnings = ["myst.header"]
