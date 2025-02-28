# Thermogram Baseline User Guide

## Introduction

The `thermogram_baseline` package provides tools for processing thermogram data, with a focus on baseline detection, subtraction, and subsequent analysis. This guide walks through common usage scenarios and best practices.

## Basic Concepts

### What is a Thermogram?

A thermogram represents the excess heat capacity (dCp) of a sample as a function of temperature. These measurements capture thermal transitions that occur in biomolecules like proteins and nucleic acids as they unfold or undergo structural changes.

### Baseline Subtraction

Baseline subtraction is the process of removing the background signal from a thermogram, leaving only the relevant thermal transitions. The process involves:

1. **Endpoint Detection**: Identifying suitable points for defining the baseline
2. **Baseline Creation**: Fitting a baseline using these endpoints
3. **Subtraction**: Removing the baseline from the original data

### Workflow

The typical workflow is:

1. Load thermogram data
2. Detect endpoints for baseline subtraction
3. Subtract baseline
4. Interpolate to a uniform temperature grid
5. Analyze and visualize results

## Basic Usage

### Processing a Single Thermogram

The simplest way to process a thermogram is using the `auto_baseline` function:

```python
import polars as pl
from thermogram_baseline import auto_baseline, plot_baseline_result

# Load data
data = pl.read_csv("thermogram.csv")

# Process thermogram
result = auto_baseline(
    data=data,
    window_size=90,
    exclusion_lower=60.0,
    exclusion_upper=80.0,
    verbose=True
)

# Visualize result
fig = plot_baseline_result(result.baseline_result)
```

### Using Step-by-Step Processing

For more control, you can use the individual processing functions:

```python
import polars as pl
import numpy as np
from thermogram_baseline import (
    ThermogramData,
    detect_endpoints,
    subtract_baseline,
    interpolate_sample,
    plot_baseline_result
)

# Load data
df = pl.read_csv("thermogram.csv")
data = ThermogramData.from_dataframe(df)

# Detect endpoints
endpoints = detect_endpoints(
    data=data,
    window_size=90,
    exclusion_lower=60.0,
    exclusion_upper=80.0,
    point_selection="innermost"
)

# Subtract baseline
baseline_result = subtract_baseline(
    data=data,
    lower_temp=endpoints.lower,
    upper_temp=endpoints.upper
)

# Interpolate to uniform grid
interpolated_result = interpolate_sample(
    data=baseline_result,
    grid_temp=np.arange(45, 90.1, 0.1)
)

# Visualize results
fig1 = plot_baseline_result(baseline_result)
fig2 = plot_interpolated_result(interpolated_result)
```

## Advanced Usage

### Batch Processing

Processing multiple thermograms can be done efficiently:

```python
import polars as pl
from thermogram_baseline import process_multiple, plot_multiple_thermograms, create_heatmap

# Load multiple thermograms
thermograms = {
    f"sample_{i}": pl.read_csv(f"sample_{i}.csv")
    for i in range(1, 11)
}

# Batch processing with parallel execution
batch_result = process_multiple(
    thermograms,
    window_size=90,
    exclusion_lower=60.0,
    exclusion_upper=80.0,
    verbose=True,
    max_workers=4  # Use 4 parallel workers
)

# Visualize results
comparison_fig = plot_multiple_thermograms(batch_result)
heatmap_fig = create_heatmap(batch_result)

# Save results to file
from pathlib import Path
process_multiple(
    thermograms,
    output_file=Path("processed_thermograms.csv")
)
```

### Signal Detection

Determine if a thermogram contains meaningful signal or just noise:

```python
from thermogram_baseline import detect_signal

# Detect signal using different methods
result_peaks = detect_signal(data, method="peaks", verbose=True)
result_arima = detect_signal(data, method="arima", verbose=True)
result_adf = detect_signal(data, method="adf", verbose=True)

print(f"Is signal (peaks method): {result_peaks.is_signal}, confidence: {result_peaks.confidence:.2f}")
print(f"Is signal (ARIMA method): {result_arima.is_signal}, confidence: {result_arima.confidence:.2f}")
print(f"Is signal (ADF method): {result_adf.is_signal}, confidence: {result_adf.confidence:.2f}")
```

### Customizing Visualization

Create custom plots for your specific needs:

```python
from thermogram_baseline import plot_multiple_thermograms

# Custom multi-thermogram plot
fig = plot_multiple_thermograms(
    batch_result,
    title="Comparison of Treatment Groups",
    sample_ids=["control_1", "treatment_1", "treatment_2"],
    colormap="viridis",
    width=1000,
    height=600,
    normalize=True  # Normalize values for better comparison
)

# Custom heatmap
heatmap = create_heatmap(
    batch_result,
    temp_range=(60, 80),  # Focus on relevant temperature range
    sample_order=sorted(batch_result.results.keys()),  # Sort samples alphabetically
    colorscale="Plasma",
    width=1200,
    height=800
)
```

## Best Practices

### Choosing Exclusion Zones

The exclusion zone parameters (`exclusion_lower` and `exclusion_upper`) should be set to encompass the region where thermal transitions are expected:

- For protein thermograms, values around 60-80°C are typical
- For nucleic acid thermograms, values may need to be adjusted based on expected transition temperatures

### Window Size Selection

The `window_size parameter` controls how many data points are considered when looking for regions of low variance:

- Larger window sizes tend to be more stable but less sensitive
- Smaller window sizes can detect more subtle regions but may be more affected by noise
- Start with window sizes between 50-100 points and adjust based on results

### Point Selection Methods

Three methods are available for selecting endpoints from windows with minimal variance:

- **innermost**: Selects points closest to the center of the data (most conservative)
- **outmost**: Selects points farthest from the center (more aggressive baseline subtraction)
- **mid**: Selects the middle point of each window (balanced approach)

For most applications, "innermost" provides the safest results with minimal risk of distorting transitions.

### Parallel Processing

When processing large batches of thermograms:

- Set `max_workers` based on your CPU cores (typically number of cores - 1)
- For very large datasets, consider processing in smaller batches to manage memory usage
- Enable `verbose=True` to monitor progress

## Troubleshooting

### Common Issues

**Problem**: Baseline subtraction removes part of the signal

**Solution**:

- Adjust exclusion zones to better isolate the transition region
- Try a different point selection method, such as "innermost"
- Manually specify endpoints if automatic detection is not optimal

**Problem**: Noisy baseline after subtraction

**Solution**:

- Increase window size to find more stable regions
- Apply smoothing to the data before processing
- Check for instrument issues in the original data

**Problem**: Parallel processing fails or hangs

**Solution**:

- Reduce `max_workers` value
- Check for corrupt or unusually large input files
- Process in smaller batches

## Next Steps

After baseline processing, you may want to:

- Calculate thermogram parameters using the `tlbparam package`
- Perform statistical analysis on sample groups
- Export results for publication or further analysis
