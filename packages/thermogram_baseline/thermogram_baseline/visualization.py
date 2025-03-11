"""Visualization utilities for thermogram data."""

from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Creates an interactive plot of thermogram data.

    Generates a Plotly figure showing temperature vs excess heat capacity with
    customizable appearance options.

    Args:
        data: Input thermogram data as ThermogramData or polars DataFrame
        title: Plot title. Defaults to "Thermogram".
        show_grid: Whether to show grid lines. Defaults to True.
        line_color: Line color (None for auto). Defaults to None.
        marker_size: Size of data point markers. Defaults to 3.
        width: Plot width in pixels. Defaults to 800.
        height: Plot height in pixels. Defaults to 500.

    Returns:
        dict: Plotly figure dictionary with plot configuration.

    Examples:
        >>> # Basic plot from ThermogramData
        >>> fig = plot_thermogram(data)
        >>>
        >>> # Customized plot from DataFrame
        >>> fig = plot_thermogram(
        ...     data_df,
        ...     title="Sample A Thermogram",
        ...     line_color="blue",
        ...     marker_size=5
        ... )
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
    """Creates an interactive plot of baseline subtraction results.

    Visualizes original data, baseline, and corrected data with optional endpoints.

    Args:
        result: Baseline subtraction result to plot
        title: Plot title. Defaults to "Baseline Subtraction".
        show_endpoints: Show detected baseline endpoints. Defaults to True.
        show_original: Show original thermogram data. Defaults to True.
        show_baseline: Show calculated baseline curve. Defaults to True.
        show_subtracted: Show baseline-subtracted data. Defaults to True.
        width: Plot width in pixels. Defaults to 800.
        height: Plot height in pixels. Defaults to 500.

    Returns:
        dict: Plotly figure dictionary with multi-trace plot.

    Examples:
        >>> # Show all components
        >>> fig = plot_baseline_result(baseline_result)
        >>>
        >>> # Show only original and subtracted data
        >>> fig = plot_baseline_result(
        ...     baseline_result,
        ...     show_baseline=False,
        ...     show_endpoints=False
        ... )
    """
    fig: Dict[str, Any] = {
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
    """Creates an interactive plot comparing original and interpolated thermogram data.

    Visualizes the interpolated data on a uniform temperature grid alongside the
    original data points if available.

    Args:
        result: Interpolated result to plot
        title: Plot title. Defaults to "Interpolated Thermogram".
        show_original: Whether to show original data points. Defaults to True.
        width: Plot width in pixels. Defaults to 800.
        height: Plot height in pixels. Defaults to 500.

    Returns:
        dict: Plotly figure dictionary with plot configuration:
            - Interpolated data as continuous line
            - Original data as scatter points (if available and requested)
            - Axes labels and title
            - Interactive hover information

    Examples:
        >>> # Basic plot with original data
        >>> fig = plot_interpolated_result(interp_result)
        >>>
        >>> # Show only interpolated data
        >>> fig = plot_interpolated_result(
        ...     interp_result,
        ...     title="Sample A - Processed",
        ...     show_original=False
        ... )
    """
    # Create figure
    fig: Dict[str, Any] = {
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
    """Creates an interactive plot comparing multiple thermograms.

    Generates a multi-line plot showing multiple thermograms with customizable
    appearance and optional data normalization.

    Args:
        data: Multiple thermograms to plot in one of these formats:
            - BatchProcessingResult: Result from batch processing
            - Dict[str, ThermogramData]: Mapping of IDs to raw data
            - Dict[str, InterpolatedResult]: Mapping of IDs to processed data
        title: Plot title. Defaults to "Multiple Thermograms".
        max_samples: Maximum number of samples to include. Defaults to None (all).
        sample_ids: Specific sample IDs to include. Defaults to None (all).
        width: Plot width in pixels. Defaults to 800.
        height: Plot height in pixels. Defaults to 500.
        colormap: Name of colormap for lines. Options:
            - "rainbow": Basic spectral colors
            - "viridis": Perceptually uniform sequential
            - "plasma": High-contrast sequential
            Defaults to None (default color cycle).
        normalize: Whether to normalize dCp values to 0-1. Defaults to False.

    Returns:
        dict: Plotly figure dictionary with plot configuration:
            - Multiple line traces (one per sample)
            - Legend identifying samples
            - Axes labels and title
            - Interactive hover information

    Raises:
        ValueError: If data format is not supported or samples have invalid type

    Examples:
        >>> # Basic plot of all samples
        >>> fig = plot_multiple_thermograms(batch_result)
        >>>
        >>> # Customized plot with subset of samples
        >>> fig = plot_multiple_thermograms(
        ...     data=sample_dict,
        ...     sample_ids=['A', 'B', 'C'],
        ...     colormap='viridis',
        ...     normalize=True
        ... )
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
    fig: Dict[str, Any] = {
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
    """Creates a heatmap visualization of multiple thermograms.

    Generates a 2D heatmap showing heat capacity values across temperature range
    for multiple samples, with customizable temperature range and sample ordering.

    Args:
        batch_result: Batch processing result containing multiple thermograms
        temp_range: Optional (min_temp, max_temp) to display. Defaults to None.
        sample_order: Optional list of sample IDs to control order. Defaults to None.
        title: Plot title. Defaults to "Thermogram Heatmap".
        width: Plot width in pixels. Defaults to 800.
        height: Plot height in pixels. Defaults to 600.
        colorscale: Plotly colorscale name. Defaults to "Viridis".

    Returns:
        dict: Plotly figure dictionary with heatmap configuration.

    Examples:
        >>> # Basic heatmap of all samples
        >>> fig = create_heatmap(batch_result)
        >>>
        >>> # Heatmap with custom range and ordering
        >>> fig = create_heatmap(
        ...     batch_result,
        ...     temp_range=(50, 80),
        ...     sample_order=['A', 'B', 'C'],
        ...     colorscale='Plasma'
        ... )
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
