# ThermogramForge

A Python implementation of thermogram analysis tools for thermal liquid biopsy (TLB) data.

## Overview

ThermogramForge provides comprehensive tools for analyzing thermogram data, focusing on:

1. **Baseline subtraction** - Automatic detection of baseline endpoints and baseline removal
2. **Peak analysis** - Detection and characterization of peaks in thermogram data
3. **Thermogram metrics** - Calculation of various metrics for thermogram characterization

The package has two main components:

- **thermogram_baseline**: For baseline subtraction and preprocessing
- **tlbparam**: For calculating metrics from thermogram data

This project is designed to be compatible with the original R implementation while offering improved performance and usability through Python's ecosystem.

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ThermogramForge.git
cd ThermogramForge

# Create a virtual environment with uv (recommended)
uv venv

# Install the package and its dependencies
uv pip install -e .
```

### Installation with R Integration (Recommended)

For optimal baseline subtraction that precisely matches the original R implementation's behavior:

```bash
# Install with R integration support
uv pip install -e ".[r-integration]"
```

This requires:

- R (4.0.0+) installed on your system
- The following R packages: `smooth.spline`, `stats`

You'll also need the `rpy2` Python package which will be installed automatically with the r-integration option.

## Quick Start

### Basic Workflow

```python
import polars as pl
import numpy as np
from thermogram_baseline.endpoint_detection import detect_endpoints
from thermogram_baseline.baseline import subtract_baseline
from thermogram_baseline.interpolation import interpolate_thermogram
from tlbparam.peak_detection import PeakDetector

# Load thermogram data
data = pl.read_csv("your_thermogram_data.csv")

# 1. Detect baseline endpoints
endpoints = detect_endpoints(data)
print(f"Detected endpoints: Lower={endpoints.lower:.2f}, Upper={endpoints.upper:.2f}")

# 2. Subtract baseline
baseline_subtracted = subtract_baseline(data, endpoints.lower, endpoints.upper)

# 3. Interpolate to a uniform temperature grid
interpolated = interpolate_thermogram(baseline_subtracted)

# 4. Detect peaks and calculate metrics
detector = PeakDetector()
peaks = detector.detect_peaks(interpolated)

# 5. Display results
for peak_name, peak_info in peaks.items():
    if peak_name != "FWHM":
        print(f"{peak_name}: Height={peak_info['peak_height']:.4f}, Temperature={peak_info['peak_temp']:.2f}°C")
    else:
        print(f"FWHM: {peak_info['value']:.4f}°C")

# 6. Save results
baseline_subtracted.write_csv("baseline_subtracted.csv")
```

### Customizing Baseline Subtraction

```python
from thermogram_baseline.spline_fitter import SplineFitter
from thermogram_baseline.baseline import subtract_baseline_with_custom_spline

# Create a custom spline fitter
fitter = SplineFitter(use_r=True)  # Use R for exact compatibility

# Customize baseline subtraction
baseline_subtracted = subtract_baseline_with_custom_spline(
    data, 
    lower_temp=55.0, 
    upper_temp=85.0,
    spline_fitter=fitter,
    spar=0.5  # Control smoothing parameter
)
```

### Web Application

ThermogramForge includes a web application for interactive data analysis:

```bash
# Run the web application
python -m thermogram_app.app
```

Then navigate to <http://127.0.0.1:8050/> in your browser.

## Documentation

For more detailed documentation:

- [API Reference](docs/source/api.rst)
- [User Guide](docs/source/usage.rst)
- [Installation](docs/source/installation.rst)

## Development

To set up a development environment:

```bash
# Create a virtual environment with uv
uv venv

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Check code formatting and linting
ruff check .
black --check .
mypy .
```

## Project Configuration

This project uses:

- `pyproject.toml` as the primary configuration file for Python tools and dependencies
- `.editorconfig` for editor-specific settings
- `.lintr` for R-specific linting configurations (used with R integration code)

Requirements files (`requirements.txt` and `requirements-dev.txt`) are generated from `pyproject.toml`
using the script at `scripts/generate_requirements.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
