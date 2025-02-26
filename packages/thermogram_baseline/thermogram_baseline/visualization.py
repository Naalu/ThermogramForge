"""Visualization utilities for thermogram data."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from .types import (
    BaselineResult,
    BatchProcessingResult,
    InterpolatedResult,
    ThermogramData,
)


def plot_thermogram(
    data: Union[ThermogramData, pl.DataFrame],
    title: Optional[str] = "Thermogram",
    show_grid: bool = True,
    line_color: Optional[str] = None,
    marker_size: int = 3,
    width: int = 800,
    height: int = 500,
) -> dict:
    """
    Create an interactive plot of thermogram data.

    Args:
        data: Thermogram data to plot
        title: Plot title
        show_grid: Whether to show grid lines
        line_color: Line color (None for auto)
        marker_size: Size of markers
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly figure dictionary
    """
    # Convert input to ThermogramData if it's a DataFrame
    if isinstance(data, pl.DataFrame):
        therm_data = ThermogramData.from_dataframe(data)
    else:
        therm_data = data

    # Create figure
    fig = {
        "data": [
            {
                "x": therm_data.temperature.tolist(),
                "y": therm_data.dcp.tolist(),
                "mode": "lines+markers",
                "marker": {"size": marker_size},
                "name": "Thermogram",
                "line": {"color": line_color},
            }
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {
                "title": "Temperature (°C)",
                "showgrid": show_grid,
                "zeroline": True,
            },
            "yaxis": {
                "title": "Excess Heat Capacity (dCp)",
                "showgrid": show_grid,
                "zeroline": True,
            },
            "width": width,
            "height": height,
            "hovermode": "closest",
        },
    }

    return fig


def plot_baseline_result(
    result: BaselineResult,
    title: Optional[str] = "Baseline Subtraction",
    show_endpoints: bool = True,
    show_original: bool = True,
    show_baseline: bool = True,
    show_subtracted: bool = True,
    width: int = 800,
    height: int = 500,
) -> dict:
    """
    Create an interactive plot of baseline subtraction result.

    Args:
        result: Baseline subtraction result to plot
        title: Plot title
        show_endpoints: Whether to show detected endpoints
        show_original: Whether to show original data
        show_baseline: Whether to show baseline
        show_subtracted: Whether to show subtracted data
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly figure dictionary
    """
    # Create figure
    fig = {
        "data": [],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": "Temperature (°C)", "showgrid": True, "zeroline": True},
            "yaxis": {
                "title": "Excess Heat Capacity (dCp)",
                "showgrid": True,
                "zeroline": True,
            },
            "width": width,
            "height": height,
            "hovermode": "closest",
            "legend": {"x": 1.05, "y": 1, "xanchor": "left"},
        },
    }

    # Add original data
    if show_original:
        fig["data"].append(
            {
                "x": result.original.temperature.tolist(),
                "y": result.original.dcp.tolist(),
                "mode": "lines",
                "name": "Original",
                "line": {"color": "blue"},
            }
        )

    # Add baseline
    if show_baseline:
        fig["data"].append(
            {
                "x": result.baseline.temperature.tolist(),
                "y": result.baseline.dcp.tolist(),
                "mode": "lines",
                "name": "Baseline",
                "line": {"color": "red", "dash": "dash"},
            }
        )

    # Add subtracted data
    if show_subtracted:
        fig["data"].append(
            {
                "x": result.subtracted.temperature.tolist(),
                "y": result.subtracted.dcp.tolist(),
                "mode": "lines",
                "name": "Subtracted",
                "line": {"color": "green"},
            }
        )

    # Add endpoints
    if show_endpoints:
        lower_temp = result.endpoints.lower
        upper_temp = result.endpoints.upper

        # Find original data points at endpoints
        lower_idx = np.abs(result.original.temperature - lower_temp).argmin()
        upper_idx = np.abs(result.original.temperature - upper_temp).argmin()

        lower_dcp = result.original.dcp[lower_idx]
        upper_dcp = result.original.dcp[upper_idx]

        fig["data"].append(
            {
                "x": [lower_temp, upper_temp],
                "y": [lower_dcp, upper_dcp],
                "mode": "markers",
                "name": "Endpoints",
                "marker": {"size": 10, "color": "red", "symbol": "circle"},
            }
        )

    return fig


def plot_interpolated_result(
    result: InterpolatedResult,
    title: Optional[str] = "Interpolated Thermogram",
    show_original: bool = True,
    width: int = 800,
    height: int = 500,
) -> dict:
    """
    Create an interactive plot of interpolated result.

    Args:
        result: Interpolated result to plot
        title: Plot title
        show_original: Whether to show original data (if available)
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly figure dictionary
    """
    # Create figure
    fig = {
        "data": [],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": "Temperature (°C)", "showgrid": True, "zeroline": True},
            "yaxis": {
                "title": "Excess Heat Capacity (dCp)",
                "showgrid": True,
                "zeroline": True,
            },
            "width": width,
            "height": height,
            "hovermode": "closest",
        },
    }

    # Add interpolated data
    fig["data"].append(
        {
            "x": result.data.temperature.tolist(),
            "y": result.data.dcp.tolist(),
            "mode": "lines",
            "name": "Interpolated",
            "line": {"color": "green"},
        }
    )

    # Add original data if available and requested
    if show_original and result.original_data is not None:
        fig["data"].append(
            {
                "x": result.original_data.temperature.tolist(),
                "y": result.original_data.dcp.tolist(),
                "mode": "markers",
                "name": "Original",
                "marker": {"size": 5, "color": "blue"},
            }
        )

    return fig


def plot_multiple_thermograms(
    data: Union[
        BatchProcessingResult, Dict[str, ThermogramData], Dict[str, InterpolatedResult]
    ],
    title: Optional[str] = "Multiple Thermograms",
    max_samples: Optional[int] = None,
    sample_ids: Optional[List[str]] = None,
    width: int = 800,
    height: int = 500,
    colormap: Optional[str] = None,
    normalize: bool = False,
) -> dict:
    """
    Create an interactive plot of multiple thermograms.

    Args:
        data: Multiple thermograms to plot
        title: Plot title
        max_samples: Maximum number of samples to include (None for all)
        sample_ids: Specific sample IDs to include (None for all)
        width: Plot width in pixels
        height: Plot height in pixels
        colormap: Colormap name (None for default)
        normalize: Whether to normalize data to 0-1 range

    Returns:
        Plotly figure dictionary
    """
    # Extract sample data
    samples = {}

    if isinstance(data, BatchProcessingResult):
        # Extract from BatchProcessingResult
        for sample_id, result in data.results.items():
            samples[sample_id] = result.data

    elif isinstance(data, dict):
        # Process dictionary
        for sample_id, item in data.items():
            if isinstance(item, ThermogramData):
                samples[sample_id] = item
            elif isinstance(item, InterpolatedResult):
                samples[sample_id] = item.data
            else:
                raise ValueError(
                    f"Unsupported data type for sample {sample_id}: {type(item)}"
                )

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Filter samples if needed
    if sample_ids is not None:
        samples = {k: v for k, v in samples.items() if k in sample_ids}

    if max_samples is not None and max_samples < len(samples):
        samples = dict(list(samples.items())[:max_samples])

    # Create figure
    fig = {
        "data": [],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": "Temperature (°C)", "showgrid": True, "zeroline": True},
            "yaxis": {
                "title": "Excess Heat Capacity (dCp)",
                "showgrid": True,
                "zeroline": True,
            },
            "width": width,
            "height": height,
            "hovermode": "closest",
            "legend": {"x": 1.05, "y": 1, "xanchor": "left"},
        },
    }

    # Generate colors if a colormap is specified
    colors = _generate_colors(len(samples), colormap) if colormap else None

    # Add each sample
    for i, (sample_id, sample_data) in enumerate(samples.items()):
        y_values = sample_data.dcp

        # Normalize if requested
        if normalize:
            y_min = np.min(y_values)
            y_max = np.max(y_values)
            if y_max > y_min:
                y_values = (y_values - y_min) / (y_max - y_min)

        fig["data"].append(
            {
                "x": sample_data.temperature.tolist(),
                "y": y_values.tolist(),
                "mode": "lines",
                "name": sample_id,
                "line": {"color": colors[i] if colors else None},
            }
        )

    return fig


def _generate_colors(n: int, colormap: str) -> List[str]:
    """
    Generate n colors from a colormap.

    Args:
        n: Number of colors to generate
        colormap: Colormap name

    Returns:
        List of color strings
    """
    # Simple colormap implementation
    if colormap == "rainbow":
        colors = [
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "indigo",
            "violet",
            "purple",
            "pink",
            "brown",
            "gray",
            "black",
        ]
    elif colormap == "viridis":
        colors = [
            "#440154",
            "#482878",
            "#3e4989",
            "#31688e",
            "#26828e",
            "#1f9e89",
            "#35b779",
            "#6ece58",
            "#b5de2b",
            "#fde725",
        ]
    elif colormap == "plasma":
        colors = [
            "#0d0887",
            "#46039f",
            "#7201a8",
            "#9c179e",
            "#bd3786",
            "#d8576b",
            "#ed7953",
            "#fb9f3a",
            "#fdca26",
            "#f0f921",
        ]
    else:
        # Default colors
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    # Repeat colors if needed
    return [colors[i % len(colors)] for i in range(n)]


def create_heatmap(
    batch_result: BatchProcessingResult,
    temp_range: Optional[Tuple[float, float]] = None,
    sample_order: Optional[List[str]] = None,
    title: str = "Thermogram Heatmap",
    width: int = 800,
    height: int = 600,
    colorscale: str = "Viridis",
) -> dict:
    """
    Create a heatmap visualization of multiple thermograms.

    Args:
        batch_result: Batch processing result with multiple thermograms
        temp_range: Optional temperature range to include (min, max)
        sample_order: Optional list of sample IDs to define order
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        colorscale: Colormap for the heatmap

    Returns:
        Plotly figure dictionary
    """
    # Extract grid temperatures
    temps = batch_result.grid_temp

    # Filter temperature range if specified
    if temp_range is not None:
        temp_mask = (temps >= temp_range[0]) & (temps <= temp_range[1])
        temps = temps[temp_mask]
    else:
        temp_mask = np.ones_like(temps, dtype=bool)

    # Get sample IDs in desired order
    if sample_order is None:
        sample_ids = list(batch_result.results.keys())
    else:
        sample_ids = [sid for sid in sample_order if sid in batch_result.results]

    # Create data matrix
    data_matrix = np.zeros((len(sample_ids), np.sum(temp_mask)))

    for i, sample_id in enumerate(sample_ids):
        result = batch_result.results[sample_id]
        data_matrix[i, :] = result.data.dcp[temp_mask]

    # Create figure
    fig = {
        "data": [
            {
                "type": "heatmap",
                "z": data_matrix.tolist(),
                "x": temps.tolist(),
                "y": sample_ids,
                "colorscale": colorscale,
                "colorbar": {"title": "dCp"},
                "hovertemplate": "Sample: %{y}<br>Temperature: %{x:.1f}°C<br>dCp: %{z:.4f}<extra></extra>",
            }
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": "Temperature (°C)"},
            "yaxis": {"title": "Sample ID", "autorange": "reversed"},
            "width": width,
            "height": height,
        },
    }

    return fig
