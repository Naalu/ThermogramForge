# Thermogram Baseline Package

## Overview

The `thermogram_baseline` package provides tools for processing and analyzing thermogram data, with a focus on baseline detection, subtraction, and signal analysis. It is designed to be efficient, flexible, and easy to use, with comprehensive error handling and visualization capabilities.

## Features

- **Endpoint Detection**: Automatically identify optimal endpoints for baseline subtraction
- **Baseline Subtraction**: Remove baselines from thermograms using various methods
- **Interpolation**: Interpolate thermogram data onto uniform temperature grids
- **Signal Detection**: Distinguish meaningful signals from noise
- **Batch Processing**: Process multiple thermograms efficiently
- **Visualization**: Create interactive plots for data analysis and presentation

## Installation

```bash
# From PyPI (once published)
pip install thermogram_baseline

# From source
pip install -e /path/to/thermogram_baseline
```

## Quick Start

```python

import polars as pl
from thermogram_baseline import auto_baseline, plot_baseline_result

# Load thermogram data
data = pl.read_csv("thermogram_data.csv")

# Process a single thermogram
result = auto_baseline(
    data=data,
    window_size=90,
    exclusion_lower=60.0,
    exclusion_upper=80.0,
    verbose=True
)

# Create interactive visualization
fig = plot_baseline_result(result.baseline_result)
```

## Documentation

For detailed documentation, please refer to:

- [User Guide](../../docs/user/thermogram_baseline_guide.md)
- [API Reference](../../docs/api/thermogram_basline_api.md)
- [Developer Guide](../../docs/developer/thermogram_baseline_dev_guide.md)

## License

MIT
