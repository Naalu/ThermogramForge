"""Tests for the interpolation module."""

import numpy as np

from packages.thermogram_baseline.thermogram_baseline.interpolation import (
    interpolate_sample,
)
from packages.thermogram_baseline.thermogram_baseline.types import (
    BaselineResult,
    Endpoints,
    ThermogramData,
)
from tests.test_data_utils import generate_simple_thermogram, thermogram_to_dataframe


def test_interpolate_sample_with_thermogram_data():
    """Test interpolation with ThermogramData input."""
    # Generate a test thermogram with non-uniform temperatures
    temperatures = np.concatenate(
        [np.linspace(45, 60, 50), np.linspace(60.2, 70, 30), np.linspace(70.5, 90, 40)]
    )
    values = np.sin((temperatures - 45) * np.pi / 45) + 0.5
    test_data = ThermogramData(temperature=temperatures, dcp=values)

    # Define a uniform grid
    grid_temp = np.linspace(45, 90, 451)

    # Interpolate
    result = interpolate_sample(test_data, grid_temp=grid_temp)

    # Check result structure
    assert isinstance(result.data, ThermogramData)
    assert result.original_data is None
    assert result.baseline_result is None

    # Check grid properties
    assert len(result.data.temperature) == len(grid_temp)
    assert np.allclose(result.data.temperature, grid_temp)

    # Check that the interpolated data preserves the general shape
    # by comparing values at key points
    for temp in [50, 60, 70, 80]:
        orig_idx = np.abs(test_data.temperature - temp).argmin()
        interp_idx = np.abs(result.data.temperature - temp).argmin()

        assert abs(test_data.dcp[orig_idx] - result.data.dcp[interp_idx]) < 0.1


def test_interpolate_sample_with_baseline_result():
    """Test interpolation with BaselineResult input."""
    # Generate a test thermogram
    test_data = generate_simple_thermogram()

    # Create a mock baseline result
    baseline = ThermogramData(
        temperature=test_data.temperature, dcp=test_data.dcp * 0.5
    )

    subtracted = ThermogramData(
        temperature=test_data.temperature, dcp=test_data.dcp * 0.5
    )

    baseline_result = BaselineResult(
        original=test_data,
        baseline=baseline,
        subtracted=subtracted,
        endpoints=Endpoints(lower=50.0, upper=85.0),
    )

    # Interpolate
    result = interpolate_sample(baseline_result)

    # Check result structure
    assert isinstance(result.data, ThermogramData)
    assert result.original_data is test_data
    assert result.baseline_result is baseline_result

    # Check default grid properties
    assert len(result.data.temperature) == 451  # Default grid: 45 to 90 by 0.1
    assert result.data.temperature[0] == 45.0
    assert result.data.temperature[-1] == 90.0


def test_interpolate_sample_with_dataframe():
    """Test interpolation with DataFrame input."""
    # Generate a test thermogram and convert to DataFrame
    test_data = generate_simple_thermogram()
    test_df = thermogram_to_dataframe(test_data)

    # Interpolate
    result = interpolate_sample(test_df)

    # Check result structure
    assert isinstance(result.data, ThermogramData)
    assert result.original_data is None
    assert result.baseline_result is None

    # Check default grid properties
    assert len(result.data.temperature) == 451  # Default grid: 45 to 90 by 0.1


def test_interpolate_sample_with_custom_grid():
    """Test interpolation with a custom temperature grid."""
    # Generate a test thermogram
    test_data = generate_simple_thermogram()

    # Define a custom grid
    custom_grid = np.linspace(50, 85, 100)

    # Interpolate
    result = interpolate_sample(test_data, grid_temp=custom_grid)

    # Check grid properties
    assert len(result.data.temperature) == 100
    assert result.data.temperature[0] == 50.0
    assert result.data.temperature[-1] == 85.0
    assert np.allclose(result.data.temperature, custom_grid)
