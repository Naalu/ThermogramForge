Installation
===========

Basic Requirements
-----------

* Python 3.10+
* uv package manager

Basic Setup
-----

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/Naalu/ThermogramForge.git
    cd ThermogramForge

    # Create a virtual environment with uv
    uv venv

    # Install the package and its dependencies
    uv sync

R Integration (Optional but Recommended)
----------------------------------------

For optimal performance and accuracy in baseline subtraction, ThermogramForge can optionally use R's `smooth.spline` function directly through the `rpy2` package. This integration provides exact equivalence to the original R implementation.

Requirements
------------

* R 4.0.0+ installed and in PATH
* R `stats` package (included with base R)
* rpy2 Python package

Setup
-----

1. Install R from <https://cran.r-project.org/>

2. Install ThermogramForge with R integration:

   .. code-block:: bash

       # Using uv
       uv pip install -e ".[r-integration]"
       
       # Or using pip
       pip install -e ".[r-integration]"

3. Verify your R environment:

   .. code-block:: bash

       python scripts/check_r_environment.py

   This script checks if R is properly installed and accessible from Python.

Usage
-----

When the R integration is available, ThermogramForge will automatically use it for the most accurate spline fitting. If R is not available, it will fall back to the Python implementation.

To explicitly control which implementation to use:

.. code-block:: python

    from thermogram_baseline.spline_fitter import SplineFitter
    
    # Create a fitter
    fitter = SplineFitter()
    
    # Use R if available (default)
    spline_r = fitter.fit_with_gcv(x, y, use_r=True)
    
    # Force Python implementation
    spline_py = fitter.fit_with_gcv(x, y, use_r=False)
