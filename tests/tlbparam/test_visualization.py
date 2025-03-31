"""
Tests for the visualization module.

This module contains tests for the thermogram visualization functionality,
verifying that it correctly generates Plotly figures for various scenarios.
"""

import numpy as np
import plotly.graph_objects as go
import polars as pl

from tlbparam.visualization import (
    create_metrics_heatmap,
    plot_metrics_comparison,
    plot_thermogram,
    plot_with_baseline,
    plot_with_peaks,
)


def create_test_data() -> pl.DataFrame:
    """Create test thermogram data for visualization testing."""
    # Create temperature range
    temps = np.linspace(45, 90, 100)

    # Create synthetic data with peaks
    peak1 = 0.3 * np.exp(-0.5 * ((temps - 63) / 2) ** 2)
    peak2 = 0.2 * np.exp(-0.5 * ((temps - 70) / 2) ** 2)
    peak3 = 0.15 * np.exp(-0.5 * ((temps - 77) / 2.5) ** 2)

    # Add baseline and noise
    baseline = 0.02 * (temps - 65)
    noise = 0.01 * np.random.randn(len(temps))

    # Combined signal
    dcp = peak1 + peak2 + peak3 + baseline + noise

    # Return as DataFrame
    return pl.DataFrame({"Temperature": temps, "dCp": dcp})


def create_test_metrics() -> pl.DataFrame:
    """Create test metrics data for visualization testing."""
    return pl.DataFrame(
        {
            "SampleID": ["Sample1", "Sample2", "Sample3"],
            "Peak 1": [0.3, 0.28, 0.32],
            "Peak 2": [0.2, 0.22, 0.18],
            "Peak 3": [0.15, 0.14, 0.16],
            "Peak 1 / Peak 2": [1.5, 1.27, 1.78],
            "Width": [10.5, 9.8, 11.2],
            "Area": [25.3, 23.7, 26.1],
        }
    )


def test_plot_thermogram():
    """Test basic thermogram plotting."""
    # Create test data
    data = create_test_data()

    # Create plot
    fig = plot_thermogram(data)

    # Check that the figure was created
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines+markers"
    assert len(fig.data[0].x) == len(data)
    assert len(fig.data[0].y) == len(data)


def test_plot_with_baseline():
    """Test plotting with baseline subtraction."""
    # Create test data
    original_data = create_test_data()

    # Create a simple baseline-subtracted version
    baseline = original_data.select("Temperature").with_columns(
        pl.lit(0.0).alias("dCp")
    )

    # Create plot
    fig = plot_with_baseline(original_data, baseline, 50.0, 85.0)

    # Check that the figure was created with correct elements
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two traces: original and baseline-subtracted
    assert fig["layout"]["shapes"][0]["x0"] == 50.0  # Lower endpoint
    assert fig["layout"]["shapes"][1]["x0"] == 85.0  # Upper endpoint


def test_plot_with_peaks():
    """Test plotting with detected peaks."""
    # Create test data
    data = create_test_data()

    # Create mock peak data
    peaks = {
        "Peak 1": {"peak_height": 0.3, "peak_temp": 63.0},
        "Peak 2": {"peak_height": 0.2, "peak_temp": 70.0},
        "Peak 3": {"peak_height": 0.15, "peak_temp": 77.0},
        "FWHM": {"fwhm": 10.5},
    }

    # Create plot
    fig = plot_with_peaks(data, peaks)

    # Check that the figure was created with correct elements
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # One for data, three for peaks (FWHM is not plotted)
    assert fig.data[0].mode == "lines"
    assert fig.data[1].mode == "markers"
    assert fig.data[1].marker.symbol == "star"


def test_plot_metrics_comparison():
    """Test metrics comparison plotting."""
    # Create test metrics data
    metrics_df = create_test_metrics()

    # Create plot with default metrics
    fig = plot_metrics_comparison(metrics_df)

    # Check that the figure was created with correct elements
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6  # One for each metric


def test_create_metrics_heatmap():
    """Test metrics heatmap creation."""
    # Create test metrics data
    metrics_df = create_test_metrics()

    # Create heatmap
    fig = create_metrics_heatmap(metrics_df)

    # Check that the figure was created with correct elements
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"
    assert len(fig.data[0].x) == 6  # Number of metrics
    assert len(fig.data[0].y) == 3  # Number of samples


def test_visualization_with_empty_data():
    """Test visualization functions with empty data."""
    # Create empty data
    empty_data = pl.DataFrame({"Temperature": [], "dCp": []})
    empty_metrics = pl.DataFrame({"SampleID": []})

    # Test each function
    fig1 = plot_thermogram(empty_data)
    assert isinstance(fig1, go.Figure)

    fig2 = plot_with_peaks(empty_data, {})
    assert isinstance(fig2, go.Figure)

    fig3 = plot_metrics_comparison(empty_metrics)
    assert isinstance(fig3, go.Figure)

    fig4 = create_metrics_heatmap(empty_metrics)
    assert isinstance(fig4, go.Figure)
