"""Tests for the visualization module."""

import numpy as np

from packages.thermogram_baseline.thermogram_baseline.batch import process_multiple
from packages.thermogram_baseline.thermogram_baseline.types import (
    BaselineResult,
    Endpoints,
    InterpolatedResult,
    ThermogramData,
)
from packages.thermogram_baseline.thermogram_baseline.visualization import (
    _generate_colors,
    create_heatmap,
    plot_baseline_result,
    plot_interpolated_result,
    plot_multiple_thermograms,
    plot_thermogram,
)
from tests.test_data_utils import generate_simple_thermogram, thermogram_to_dataframe


def test_plot_thermogram():
    """Test plotting a simple thermogram."""
    # Generate a test thermogram
    test_data = generate_simple_thermogram()

    # Create plot
    fig = plot_thermogram(test_data, title="Test Thermogram")

    # Check figure structure
    assert isinstance(fig, dict)
    assert "data" in fig
    assert "layout" in fig
    assert len(fig["data"]) == 1

    # Check data format
    data = fig["data"][0]
    assert "x" in data
    assert "y" in data
    assert len(data["x"]) == len(test_data.temperature)
    assert len(data["y"]) == len(test_data.dcp)

    # Check layout
    layout = fig["layout"]
    assert layout["title"]["text"] == "Test Thermogram"
    assert layout["xaxis"]["title"] == "Temperature (°C)"
    assert layout["yaxis"]["title"] == "Excess Heat Capacity (dCp)"


def test_plot_thermogram_with_dataframe():
    """Test plotting a thermogram from a DataFrame."""
    # Generate a test thermogram and convert to DataFrame
    test_data = generate_simple_thermogram()
    test_df = thermogram_to_dataframe(test_data)

    # Create plot
    fig = plot_thermogram(test_df)

    # Check figure structure
    assert isinstance(fig, dict)
    assert "data" in fig
    assert len(fig["data"]) == 1
    assert len(fig["data"][0]["x"]) == len(test_data.temperature)


def test_plot_baseline_result():
    """Test plotting baseline subtraction result."""
    # Create a mock baseline result
    original = generate_simple_thermogram()
    baseline = ThermogramData(temperature=original.temperature, dcp=original.dcp * 0.5)
    subtracted = ThermogramData(
        temperature=original.temperature, dcp=original.dcp * 0.5
    )
    endpoints = Endpoints(lower=50.0, upper=85.0)

    result = BaselineResult(
        original=original, baseline=baseline, subtracted=subtracted, endpoints=endpoints
    )

    # Create plot
    fig = plot_baseline_result(result)

    # Check figure structure
    assert isinstance(fig, dict)
    assert "data" in fig
    assert len(fig["data"]) == 4  # Original, baseline, subtracted, endpoints

    # Test with some options disabled
    fig2 = plot_baseline_result(result, show_endpoints=False, show_original=False)

    # Should have fewer data traces
    assert len(fig2["data"]) == 2  # Baseline and subtracted only


def test_plot_interpolated_result():
    """Test plotting interpolated result."""
    # Create a mock interpolated result
    original = generate_simple_thermogram()
    grid_temp = np.linspace(50, 85, 100)
    interpolated = ThermogramData(
        temperature=grid_temp, dcp=np.sin((grid_temp - 50) * np.pi / 35) + 0.5
    )

    result = InterpolatedResult(
        data=interpolated, grid_temp=grid_temp, original_data=original
    )

    # Create plot
    fig = plot_interpolated_result(result)

    # Check figure structure
    assert isinstance(fig, dict)
    assert "data" in fig
    assert len(fig["data"]) == 2  # Interpolated and original

    # Test without original
    fig2 = plot_interpolated_result(result, show_original=False)

    # Should have only interpolated data
    assert len(fig2["data"]) == 1


def test_plot_multiple_thermograms():
    """Test plotting multiple thermograms."""
    # Generate a few test thermograms
    thermograms = {
        "sample1": generate_simple_thermogram(peak_center=65.0),
        "sample2": generate_simple_thermogram(peak_center=70.0),
        "sample3": generate_simple_thermogram(peak_center=75.0),
    }

    # Create plot
    fig = plot_multiple_thermograms(thermograms)

    # Check figure structure
    assert isinstance(fig, dict)
    assert "data" in fig
    assert len(fig["data"]) == 3

    # Test with sample filtering
    fig2 = plot_multiple_thermograms(thermograms, sample_ids=["sample1", "sample3"])

    # Should have only two samples
    assert len(fig2["data"]) == 2

    # Check sample names
    names = [trace["name"] for trace in fig2["data"]]
    assert "sample1" in names
    assert "sample3" in names
    assert "sample2" not in names


def test_plot_multiple_thermograms_with_batch_result():
    """Test plotting multiple thermograms from a batch result."""
    # Generate test thermograms
    thermograms = {
        "sample1": generate_simple_thermogram(peak_center=65.0),
        "sample2": generate_simple_thermogram(peak_center=70.0),
    }

    # Process the thermograms
    batch_result = process_multiple(thermograms, verbose=False, max_workers=1)

    # Create plot
    fig = plot_multiple_thermograms(batch_result)

    # Check figure structure
    assert isinstance(fig, dict)
    assert "data" in fig
    assert len(fig["data"]) == 2

    # Check sample names
    names = [trace["name"] for trace in fig["data"]]
    assert "sample1" in names
    assert "sample2" in names


def test_create_heatmap():
    """Test creating a heatmap from batch results."""
    # Generate test thermograms
    thermograms = {
        "sample1": generate_simple_thermogram(peak_center=65.0),
        "sample2": generate_simple_thermogram(peak_center=70.0),
        "sample3": generate_simple_thermogram(peak_center=75.0),
    }

    # Process the thermograms
    batch_result = process_multiple(thermograms, verbose=False, max_workers=1)

    # Create heatmap
    fig = create_heatmap(batch_result)

    # Check figure structure
    assert isinstance(fig, dict)
    assert "data" in fig
    assert len(fig["data"]) == 1
    assert fig["data"][0]["type"] == "heatmap"

    # Check data dimensions
    assert len(fig["data"][0]["y"]) == 3  # 3 samples
    assert len(fig["data"][0]["x"]) == len(batch_result.grid_temp)  # All temperatures

    # Test with temperature range
    fig2 = create_heatmap(batch_result, temp_range=(60, 80))

    # Should have fewer temperature points
    assert len(fig2["data"][0]["x"]) < len(batch_result.grid_temp)

    # Test with sample order
    fig3 = create_heatmap(batch_result, sample_order=["sample3", "sample1", "sample2"])

    # Check sample order
    assert fig3["data"][0]["y"] == ["sample3", "sample1", "sample2"]


def test_generate_colors():
    """Test color generation for plots."""
    # Test default colors
    colors = _generate_colors(5, None)
    assert len(colors) == 5

    # Test rainbow colormap
    colors = _generate_colors(7, "rainbow")
    assert len(colors) == 7
    assert colors[0] == "red"
    assert colors[6] == "violet"

    # Test color repeating for large n
    colors = _generate_colors(15, "rainbow")
    assert len(colors) == 15
    assert colors[0] == colors[12]  # Should repeat after 12 colors
