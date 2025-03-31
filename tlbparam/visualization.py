"""
Visualization module for thermogram data analysis.

This module provides visualization functions for thermogram data, including
raw data, baseline-subtracted data, detected peaks, and calculated metrics.
"""

from typing import Dict, List, Optional

import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from plotly.graph_objects import Bar, Heatmap


def plot_thermogram(
    data: pl.DataFrame,
    temp_col: str = "Temperature",
    value_col: str = "dCp",
    title: str = "Thermogram",
    show_grid: bool = True,
) -> go.Figure:
    """
    Create a basic plot of thermogram data.

    Args:
        data: DataFrame with thermogram data
        temp_col: Name of temperature column
        value_col: Name of value column
        title: Plot title
        show_grid: Whether to show grid lines

    Returns:
        Plotly Figure object
    """
    # Create figure
    fig = go.Figure()

    # Add thermogram trace
    fig.add_trace(
        go.Scatter(
            x=data.select(pl.col(temp_col)).to_numpy().flatten(),
            y=data.select(pl.col(value_col)).to_numpy().flatten(),
            mode="lines+markers",
            name="Thermogram Data",
            line=dict(color="blue", width=1.5),
            marker=dict(size=3),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Temperature (°C)",
        yaxis_title="Excess Heat Capacity (dCp)",
        template="plotly_white",
        showlegend=True,
    )

    # Add grid if requested
    if show_grid:
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        )

    return fig


def plot_with_baseline(
    original_data: pl.DataFrame,
    baseline_subtracted: pl.DataFrame,
    lower_endpoint: float,
    upper_endpoint: float,
    temp_col: str = "Temperature",
    value_col: str = "dCp",
    title: str = "Thermogram with Baseline Subtraction",
) -> go.Figure:
    """
    Plot original thermogram data with baseline-subtracted data.

    Args:
        original_data: DataFrame with original thermogram data
        baseline_subtracted: DataFrame with baseline-subtracted data
        lower_endpoint: Lower temperature endpoint for baseline
        upper_endpoint: Upper temperature endpoint for baseline
        temp_col: Name of temperature column
        value_col: Name of value column
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Create figure with subplots (original and baseline-subtracted)
    fig = sp.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Original Data", "Baseline Subtracted"),
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
    )

    # Add original data trace
    fig.add_trace(
        go.Scatter(
            x=original_data.select(pl.col(temp_col)).to_numpy().flatten(),
            y=original_data.select(pl.col(value_col)).to_numpy().flatten(),
            mode="lines+markers",
            name="Original Data",
            line=dict(color="blue", width=1.5),
            marker=dict(size=3),
        ),
        row=1,
        col=1,
    )

    # Add baseline-subtracted trace
    fig.add_trace(
        go.Scatter(
            x=baseline_subtracted.select(pl.col(temp_col)).to_numpy().flatten(),
            y=baseline_subtracted.select(pl.col(value_col)).to_numpy().flatten(),
            mode="lines+markers",
            name="Baseline Subtracted",
            line=dict(color="green", width=1.5),
            marker=dict(size=3),
        ),
        row=2,
        col=1,
    )

    # Add endpoint markers to the original data plot
    fig.add_vline(
        x=lower_endpoint,
        line=dict(color="red", width=1, dash="dash"),
        annotation_text="Lower Endpoint",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=upper_endpoint,
        line=dict(color="red", width=1, dash="dash"),
        annotation_text="Upper Endpoint",
        annotation_position="top left",
        row=1,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axis labels
    fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Excess Heat Capacity (dCp)", row=1, col=1)
    fig.update_yaxes(title_text="Baseline Subtracted dCp", row=2, col=1)

    return fig


def plot_with_peaks(
    data: pl.DataFrame,
    peaks: Dict[str, Dict[str, float]],
    temp_col: str = "Temperature",
    value_col: str = "dCp",
    title: str = "Thermogram with Detected Peaks",
) -> go.Figure:
    """
    Plot thermogram data with detected peaks.

    Args:
        data: DataFrame with thermogram data
        peaks: Dictionary with peak information from peak_detection module
        temp_col: Name of temperature column
        value_col: Name of value column
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Create figure
    fig = go.Figure()

    # Add thermogram trace
    fig.add_trace(
        go.Scatter(
            x=data.select(pl.col(temp_col)).to_numpy().flatten(),
            y=data.select(pl.col(value_col)).to_numpy().flatten(),
            mode="lines",
            name="Thermogram Data",
            line=dict(color="blue", width=1.5),
        )
    )

    # Add peak markers
    for peak_name, peak_info in peaks.items():
        if peak_name != "FWHM" and peak_info.get("peak_height", 0) > 0:
            fig.add_trace(
                go.Scatter(
                    x=[peak_info["peak_temp"]],
                    y=[peak_info["peak_height"]],
                    mode="markers",
                    name=f"{peak_name} ({peak_info['peak_temp']:.1f}°C)",
                    marker=dict(size=10, color="red", symbol="star"),
                )
            )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Temperature (°C)",
        yaxis_title="Excess Heat Capacity (dCp)",
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_metrics_comparison(
    metrics_df: pl.DataFrame,
    sample_id_col: str = "SampleID",
    metrics_to_plot: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create a comparison plot of thermogram metrics across samples.

    Args:
        metrics_df: DataFrame with calculated metrics
        sample_id_col: Name of sample ID column
        metrics_to_plot: List of metric names to plot (if None, selects common metrics)

    Returns:
        Plotly Figure object
    """
    # If no metrics specified, use a default set
    if metrics_to_plot is None:
        metrics_to_plot = [
            "Peak 1",
            "Peak 2",
            "Peak 3",
            "Peak 1 / Peak 2",
            "Width",
            "Area",
        ]

    # Filter to only include metrics that exist in the DataFrame
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]

    if not available_metrics:
        # Create an empty figure with a message if no metrics are available
        fig = go.Figure()
        fig.update_layout(
            title="No metrics available for plotting",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        fig.add_annotation(
            text="No matching metrics found in data",
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Create subplots - one for each metric
    fig = sp.make_subplots(
        rows=len(available_metrics),
        cols=1,
        subplot_titles=available_metrics,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    # Get unique sample IDs
    sample_ids = metrics_df.select(pl.col(sample_id_col)).unique().to_series().to_list()

    # Add bars for each metric
    for i, metric in enumerate(available_metrics):
        # Extract metric values for each sample
        values = [
            metrics_df.filter(pl.col(sample_id_col) == sample_id)
            .select(pl.col(metric))
            .item()
            for sample_id in sample_ids
        ]

        fig.add_trace(
            Bar(x=sample_ids, y=values, name=metric),
            row=i + 1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title="Metrics Comparison Across Samples",
        showlegend=False,
        height=300 * len(available_metrics),
        template="plotly_white",
    )

    # Update y-axis titles for each subplot
    for i, metric in enumerate(available_metrics):
        fig.update_yaxes(title_text=metric, row=i + 1, col=1)

    # Only show x-axis title on the bottom subplot
    fig.update_xaxes(title_text="Sample ID", row=len(available_metrics), col=1)

    return fig


def create_metrics_heatmap(
    metrics_df: pl.DataFrame,
    sample_id_col: str = "SampleID",
    metrics_to_include: Optional[List[str]] = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    """
    Create a heatmap of metrics for multiple samples.

    Args:
        metrics_df: DataFrame with calculated metrics
        sample_id_col: Name of sample ID column
        metrics_to_include: List of metrics to include in heatmap
        colorscale: Plotly colorscale to use

    Returns:
        Plotly Figure object
    """
    # If no metrics specified, use all numeric columns except sample ID
    if metrics_to_include is None:
        # Try to identify numeric columns
        numeric_cols = [
            col
            for col in metrics_df.columns
            if col != sample_id_col and metrics_df[col].dtype in (pl.Float64, pl.Int64)
        ]
        metrics_to_include = numeric_cols

    # Filter to only include metrics that exist in the DataFrame
    available_metrics = [m for m in metrics_to_include if m in metrics_df.columns]

    if not available_metrics:
        # Create an empty figure with a message if no metrics are available
        fig = go.Figure()
        fig.update_layout(
            title="No metrics available for heatmap",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        fig.add_annotation(
            text="No matching metrics found in data",
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Get unique sample IDs
    sample_ids = metrics_df.select(pl.col(sample_id_col)).unique().to_series().to_list()

    # Create empty matrix for heatmap
    heatmap_data = []
    for sample_id in sample_ids:
        # Get metrics for this sample
        sample_metrics = (
            metrics_df.filter(pl.col(sample_id_col) == sample_id)
            .select(available_metrics)
            .to_dicts()[0]
        )

        # Add to heatmap data
        row = [sample_metrics.get(metric, 0) for metric in available_metrics]
        heatmap_data.append(row)

    # Create heatmap
    fig = go.Figure(
        data=Heatmap(
            z=heatmap_data,
            x=available_metrics,
            y=sample_ids,
            colorscale=colorscale,
            hoverongaps=False,
            colorbar=dict(title="Value"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Metrics Heatmap",
        xaxis_title="Metric",
        yaxis_title="Sample ID",
        template="plotly_white",
    )

    return fig
