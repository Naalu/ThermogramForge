R Integration Details
===================

The R integration in ThermogramForge provides exact compatibility with the original R implementation, particularly for spline fitting functionality.

How It Works
-----------

ThermogramForge can directly use R's ``smooth.spline`` function through the ``rpy2`` package. This ensures the highest possible fidelity to the original R implementation.

The integration is implemented in the ``SplineFitter`` class in ``thermogram_baseline/spline_fitter.py``. When you call ``fit_with_gcv()``, it will:

1. Check if ``rpy2`` is available and R is properly installed
2. If ``use_r=True`` (default) and R is available, use R's ``smooth.spline`` function
3. Otherwise, fall back to the Python implementation

The R implementation is wrapped in a ``RSpline`` class that provides a compatible interface with SciPy's ``UnivariateSpline``.

Debugging R Integration Issues
----------------------------

If you're experiencing problems with the R integration, here are some steps to diagnose and fix them:

Verify R Environment
~~~~~~~~~~~~~~~~~~~

Run the environment checker script:

.. code-block:: bash

    python scripts/check_r_environment.py

This script checks:

- If R is installed and accessible
- If required R packages are available
- If ``rpy2`` is installed and working
- If a simple integration test passes

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

R not found in PATH
^^^^^^^^^^^^^^^^^^^

**Symptom:** Error message like ``R: command not found`` or ``ExecutableNotFound``

**Solution:**

- Ensure R is installed
- Add R to your PATH environment variable
- On Windows, restart your terminal after installation

rpy2 installation fails
^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Errors during ``pip install rpy2``

**Solution:**

- Ensure R is installed before installing rpy2
- On Windows, you may need to set ``R_HOME`` environment variable
- On macOS, install R using Homebrew (``brew install r``) for better compatibility
- Check rpy2 documentation for platform-specific instructions

R package loading errors
^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Errors like ``Error: package 'stats' not found``

**Solution:**

- Start R and run ``install.packages("stats")`` if needed
- Check if your R installation is complete and not corrupted

Environment Variables
-------------------

You can set the following environment variables:

- ``THERMOGRAM_FORGE_USE_R``: Set to "0" to disable R integration by default
- ``THERMOGRAM_FORGE_VERBOSE``: Set to "1" to enable verbose logging

Example:

.. code-block:: bash

    # Disable R integration
    export THERMOGRAM_FORGE_USE_R=0

    # Enable verbose logging
    export THERMOGRAM_FORGE_VERBOSE=1

    # Run your script
    python my_script.py

Python Implementation Notes
------------------------

When R integration is not available, ThermogramForge uses a Python implementation that approximates R's behavior:

- Uses SciPy's ``UnivariateSpline`` with custom parameter conversion
- Implements Generalized Cross-Validation (GCV) for automatic smoothing parameter selection
- Includes heuristics to map between R's ``spar`` parameter and SciPy's ``s`` parameter

For the most accurate results that exactly match the R implementation, we recommend using the R integration option when possible.
