# R Integration in ThermogramForge

This document explains the R integration in ThermogramForge, including how it works, why it's useful, and how to debug issues.

## Overview

The `ThermogramForge` package implements Python versions of various R functions from the original `ThermogramBaseline` R package. One of the most critical functions is R's `smooth.spline`, which is used for baseline subtraction in thermogram data.

Spline fitting with automatic smoothing parameter selection is a complex statistical procedure, and the exact implementation details can vary significantly between R and Python libraries. To ensure maximum compatibility with the original R code, we've implemented:

1. A pure Python implementation that approximates R's behavior
2. A direct R integration using `rpy2` that calls R's `smooth.spline` function directly

## Why Use R Integration?

The R integration provides several benefits:

- **Exact compatibility** with the original R implementation
- **Validation tool** for our Python implementation
- **Fallback option** when the Python implementation doesn't match R's behavior precisely

For production use, we recommend using the R integration when possible, as it ensures the most accurate results that match the original R package.

## How It Works

The integration is implemented in the `SplineFitter` class in `thermogram_baseline/spline_fitter.py`. When you call `fit_with_gcv()`, it will:

1. Check if `rpy2` is available and R is properly installed
2. If `use_r=True` (default) and R is available, use R's `smooth.spline` function
3. Otherwise, fall back to the Python implementation

The R implementation is wrapped in a `RSpline` class that provides a compatible interface with SciPy's `UnivariateSpline`.

## Debugging R Integration Issues

If you're experiencing problems with the R integration, here are some steps to diagnose and fix them:

### 1. Verify R Environment

Run the environment checker script:

```bash
python scripts/check_r_environment.py
```

This script checks:

- If R is installed and accessible
- If required R packages are available
- If `rpy2` is installed and working
- If a simple integration test passes

### 2. Common Issues and Solutions

#### R not found in PATH

**Symptom:** Error message like `R: command not found` or `ExecutableNotFound`

**Solution:**

- Ensure R is installed
- Add R to your PATH environment variable
- On Windows, restart your terminal after installation

#### rpy2 installation fails

**Symptom:** Errors during `pip install rpy2`

**Solution:**

- Ensure R is installed before installing rpy2
- On Windows, you may need to set `R_HOME` environment variable
- On macOS, install R using Homebrew (`brew install r`) for better compatibility
- Check rpy2 documentation for platform-specific instructions

#### R package loading errors

**Symptom:** Errors like `Error: package 'stats' not found`

**Solution:**

- Start R and run `install.packages("stats")` if needed
- Check if your R installation is complete and not corrupted

### 3. Enable Verbose Logging

To get more detailed information about what's happening during spline fitting:

```python
from thermogram_baseline.spline_fitter import SplineFitter

# Create a fitter with verbose logging
fitter = SplineFitter(verbose=True)

# Fit with verbose output
spline = fitter.fit_with_gcv(x, y)
```

### 4. Comparing R and Python Implementations

For diagnosing differences between R and Python implementations:

```python
from thermogram_baseline.spline_fitter import SplineFitter
import numpy as np
import matplotlib.pyplot as plt

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)

# Create fitter
fitter = SplineFitter(verbose=True)

# Fit with both implementations
spline_r = fitter.fit_with_gcv(x, y, use_r=True)
spline_py = fitter.fit_with_gcv(x, y, use_r=False)

# Compare fitted values
fitted_r = spline_r(x)
fitted_py = spline_py(x)
diff = fitted_r - fitted_py

# Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.scatter(x, y, alpha=0.5, label='Data')
plt.plot(x, fitted_r, 'r-', label='R fit')
plt.plot(x, fitted_py, 'b--', label='Python fit')
plt.legend()
plt.title('Comparison of R and Python Spline Fits')

plt.subplot(2, 1, 2)
plt.plot(x, diff, 'g-')
plt.axhline(y=0, color='k', linestyle=':')
plt.title(f'Difference (R - Python), Max: {np.max(np.abs(diff)):.6f}')

plt.tight_layout()
plt.show()
```

## Configuration Options

The R integration can be controlled with the following options:

### In Python code

```python
from thermogram_baseline.spline_fitter import SplineFitter

# Always use R if available
fitter = SplineFitter()
spline = fitter.fit_with_gcv(x, y, use_r=True)  # Default

# Force Python implementation even if R is available
spline = fitter.fit_with_gcv(x, y, use_r=False)

# Enable verbose logging
fitter = SplineFitter(verbose=True)
```

### Environment Variables

You can set the following environment variables:

- `THERMOGRAM_FORGE_USE_R`: Set to "0" to disable R integration by default
- `THERMOGRAM_FORGE_VERBOSE`: Set to "1" to enable verbose logging

Example:

```bash
# Disable R integration
export THERMOGRAM_FORGE_USE_R=0

# Enable verbose logging
export THERMOGRAM_FORGE_VERBOSE=1

# Run your script
python my_script.py
```

## Python Implementation Notes

The Python implementation of spline fitting in ThermogramForge:

1. Uses SciPy's `UnivariateSpline` with custom parameter conversion
2. Implements Generalized Cross-Validation (GCV) for automatic smoothing parameter selection
3. Includes heuristics to map between R's `spar` parameter and SciPy's `s` parameter

The key challenge is that R's `smooth.spline` and SciPy's `UnivariateSpline` use different internal algorithms and parameterizations, making exact replication difficult. The current implementation provides a close approximation but may not match R exactly for all datasets.

For the most accurate results, we recommend using the R integration option.
