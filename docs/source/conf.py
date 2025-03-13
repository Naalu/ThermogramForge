# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# Project information
project = "Thermogram Analysis"
copyright = "2025, Karl Reger"
author = "Karl Reger"
version = "0.1.0"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
