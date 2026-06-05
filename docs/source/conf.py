# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "tmg-hmc"
copyright = "2026, Erik A. Bensen, Mikael Kuusela"
author = "Erik A. Bensen, Mikael Kuusela"
release = "1.0.4"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # pulls docstrings automatically
    "sphinx.ext.napoleon",  # parses NumPy-style docstrings
    "sphinx.ext.mathjax",  # renders LaTeX math
    "sphinx.ext.viewcode",  # adds links to source code
    "sphinx_autodoc_typehints",  # pulls type annotations into docs
    "nbsphinx",  # renders Jupyter notebooks
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# Napoleon settings - tell Sphinx your docstrings are NumPy style
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc settings
autodoc_member_order = "bysource"  # document members in source order
autodoc_typehints = "description"  # put type hints in the description

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during the build process.

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
        "packages": {"[+]": ["boldsymbol"]},
    },
    "loader": {"load": ["[tex]/boldsymbol"]},
}

html_theme_options = {
    "navigation_depth": 4,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
