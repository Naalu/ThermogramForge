# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

import sphinx_rtd_theme  # type: ignore

# -- Path setup --------------------------------------------------------------
# Paths to the source code
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../packages"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ThermogramForge"
copyright = "2025, Karl Reger"
author = "Karl Reger"
version = "0.10"
release = "0.10"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx extension modules
extensions = [
    "sphinx.ext.autodoc",  # Pull documentation from docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx_rtd_theme",  # Read the Docs theme
    "myst_parser",  # For parsing Markdown files
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Patterns that are excluded from any directory (files that are not included in the documentation)
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for autodoc extension -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc

# The suffix of source filenames
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# includining the theme so the import works
my_theme = sphinx_rtd_theme

# Theme configuration
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
myst_enable_extensions = ["colon_fence"]
