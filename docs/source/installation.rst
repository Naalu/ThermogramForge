Installation
===========

Basic Requirements
-----------------

* Python 3.10+
* uv package manager (recommended) or pip

Basic Setup
----------

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/Naalu/ThermogramForge.git
    cd ThermogramForge

    # Create a virtual environment with uv (recommended)
    uv venv
    uv pip install -e .

    # Or using standard tools
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e .

R Integration (Optional but Recommended)
----------------------------------------

For optimal performance and accuracy in baseline subtraction, ThermogramForge can optionally use R's `smooth.spline` function directly through the `rpy2` package. This integration provides exact equivalence to the original R implementation.

Requirements
~~~~~~~~~~~

* R 4.0.0+ installed and in PATH
* R `stats` package (included with base R)
* rpy2 Python package

Setup
~~~~~

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

Development Installation
-----------------------

For development, install additional dependencies:

.. code-block:: bash

    # Using uv
    uv pip install -e ".[dev]"

    # Or using pip
    pip install -e ".[dev]"

This installs testing, documentation, and code quality tools.

Web Application
--------------

To run the web application:

.. code-block:: bash

    # Using Python module
    python -m thermogram_app.app

    # Navigate to http://127.0.0.1:8050/ in your browser

Standalone Executables
---------------------

Standalone executables can be built for Windows and macOS:

.. code-block:: bash

    # Build for Windows
    python scripts/build_windows.py

    # Build for macOS
    python scripts/build_macos.py

The executables will be available in the `dist` directory.

Troubleshooting
--------------

If you encounter issues during installation:

1. Ensure Python 3.10+ is installed and active
2. For R integration issues, see the R Integration documentation
3. For rpy2 installation issues, ensure R is installed first
4. On Windows, you may need to set the R_HOME environment variable

For detailed logs and help with troubleshooting, run:

.. code-block:: bash

    # Enable verbose output
    export THERMOGRAM_FORGE_VERBOSE=1

    # Run the installation
    pip install -e . -v
