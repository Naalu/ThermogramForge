import os
import sys
from datetime import date

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ThermogramForge"
copyright = f"{date.today().year}, Chris Reger"
author = "Chris Reger"

# Get version from __init__.py or another source if available
# For now, setting a placeholder
release = "0.1.0"
version = "0.1.0"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.intersphinx",  # Link to other projects' docs
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx_copybutton",  # Add copy button to code blocks
    # 'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings (Uncomment if needed)
    # 'myst_parser',           # Support for Markdown (Uncomment if needed)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The master toctree document.
master_doc = (
    "index"  # Changed from root_doc for older Sphinx versions compatibility if needed
)

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_css_files = ['custom.css'] # Example if you add custom CSS

# Theme options
html_theme_options = {
    "github_url": "https://github.com/Naalu/ThermogramForge/",
    "logo": {
        # "image_light": "_static/logo-light.png", # Example: Add light mode logo
        # "image_dark": "_static/logo-dark.png",   # Example: Add dark mode logo
        "text": "ThermogramForge",  # Show project name if no logo
    },
    "use_edit_page_button": True,
    # Add other options here from PyData theme docs as needed
    # e.g., "show_toc_level": 2, "navbar_align": "left", etc.
}

# Configure the edit page button
html_context = {
    "github_user": "Naalu",
    "github_repo": "ThermogramForge",
    "github_version": "documentation",  # Or main/master depending on your branch
    "doc_path": "docs",
}

# -- Options for intersphinx extension ---------------------------------------
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "dash": ("https://dash.plotly.com/", None),
    # "dash_bootstrap_components": (
    #     "https://dash-bootstrap-components.opensource.faculty.ai/", # Temporarily removed
    #     None,
    # ),
    # Add others as needed
}

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = "bysource"  # Keep source order
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- Options for Napoleon extension (if used) --------------------------------
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
# napoleon_preprocess_types = False
# napoleon_type_aliases = None
# napoleon_attr_annotations = True
