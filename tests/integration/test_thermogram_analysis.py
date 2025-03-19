"""
Integration test for the full thermogram analysis workflow.

This test verifies that all components work together correctly,
from data loading to baseline subtraction, interpolation, and peak detection.
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from thermogram_baseline.baseline import subtract_baseline
from thermogram_baseline.endpoint_detection import detect_endpoints
from thermogram_baseline.interpolation import interpolate_thermogram
from tlbparam.peak_detection import PeakDetector


def create_complex_thermogram() -> pl.DataFrame:
    """
    Create a complex thermogram for testing with realistic features.

    Returns:
        A Polars DataFrame with synthetic thermogram data
    """
    # Create a realistic temperature range with some irregularity
    np.random.seed(42)  # For reproducibility
    n_points = 400
    temp_base = np.linspace(45, 90, n_points)
    # Add small random variations to temperature steps
    temp_noise = np.random.normal(0, 0.05, n_points - 1)
    temperature = np.zeros(n_points)
    temperature[0] = temp_base[0]
    for i in range(1, n_points):
        temperature[i] = (
            temperature[i - 1] + (temp_base[i] - temp_base[i - 1]) + temp_noise[i - 1]
        )

    # Create three main peaks for albumin, alpha-2, and gamma globulins
    peak1 = 0.3 * np.exp(-0.5 * ((temperature - 63) / 2) ** 2)  # Albumin ~ 63°C
    peak2 = 0.2 * np.exp(-0.5 * ((temperature - 70) / 2) ** 2)  # Alpha-2 ~ 70°C
    peak3 = 0.15 * np.exp(-0.5 * ((temperature - 77) / 2.5) ** 2)  # Gamma ~ 77°C

    # Add a small fibrinogen peak
    peak_f = 0.1 * np.exp(-0.5 * ((temperature - 52) / 1.5) ** 2)  # Fibrinogen ~ 52°C

    # Create a non-linear baseline
    baseline = 0.02 * (temperature - 65) + 0.001 * (temperature - 65) ** 2

    # Combine components with realistic noise
    dcp = peak1 + peak2 + peak3 + peak_f + baseline + 0.01 * np.random.randn(n_points)

    # Create DataFrame
    return pl.DataFrame({"Temperature": temperature, "dCp": dcp})


def test_full_workflow_integration():
    """Test the full thermogram analysis workflow with realistic data."""
    # Create realistic test data
    data = create_complex_thermogram()

    # Save to a temp file for testing file loading
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "test_thermogram.csv"
        data.write_csv(temp_path)

        # 1. Reload from file to test data loading
        loaded_data = pl.read_csv(temp_path)
        assert loaded_data.height == data.height
        assert all(col in loaded_data.columns for col in ["Temperature", "dCp"])

        # 2. Detect endpoints
        endpoints = detect_endpoints(loaded_data, w=45)

        # 3. Subtract baseline
        baseline_subtracted = subtract_baseline(
            loaded_data, endpoints.lower, endpoints.upper
        )

        # 4. Interpolate to fixed grid
        grid_temp = np.arange(45, 90.1, 0.1)
        interpolated = interpolate_thermogram(baseline_subtracted, grid_temp)

        # 5. Detect peaks
        detector = PeakDetector()
        peaks = detector.detect_peaks(interpolated)

        # Verify each step

        # Endpoints should be reasonable
        assert (
            45 <= endpoints.lower <= 60
        ), f"Lower endpoint {endpoints.lower} outside expected range"
        assert (
            80 <= endpoints.upper <= 90
        ), f"Upper endpoint {endpoints.upper} outside expected range"

        # Baseline subtraction should work
        assert baseline_subtracted.height == loaded_data.height

        # Values near endpoints should be close to zero after baseline subtraction
        lower_values = (
            baseline_subtracted.filter(
                pl.col("Temperature").is_between(
                    endpoints.lower - 1, endpoints.lower + 1
                )
            )
            .select("dCp")
            .to_numpy()
            .flatten()
        )
        upper_values = (
            baseline_subtracted.filter(
                pl.col("Temperature").is_between(
                    endpoints.upper - 1, endpoints.upper + 1
                )
            )
            .select("dCp")
            .to_numpy()
            .flatten()
        )

        assert (
            np.mean(np.abs(lower_values)) < 0.1
        ), "Baseline subtraction failed near lower endpoint"
        assert (
            np.mean(np.abs(upper_values)) < 0.1
        ), "Baseline subtraction failed near upper endpoint"

        # Interpolation should produce the correct grid
        assert interpolated.height == len(grid_temp)
        np.testing.assert_array_equal(
            interpolated.select("Temperature").to_numpy().flatten(), grid_temp
        )

        # Peak detection should find the expected peaks
        assert "Peak 1" in peaks
        assert "Peak 2" in peaks
        assert "Peak 3" in peaks
        assert "Peak F" in peaks
        assert "FWHM" in peaks

        # Peaks should be in the expected temperature ranges
        assert 60 <= peaks["Peak 1"]["peak_temp"] <= 66
        assert 67 <= peaks["Peak 2"]["peak_temp"] <= 73
        assert 73 <= peaks["Peak 3"]["peak_temp"] <= 81
        assert 50 <= peaks["Peak F"]["peak_temp"] <= 54

        # FWHM should be positive
        assert peaks["FWHM"]["fwhm"] > 0.0

        print("\nIntegration test results:")
        print(
            f"Detected endpoints: Lower={endpoints.lower:.2f}, "
            f"Upper={endpoints.upper:.2f}"
        )
        print(f"FWHM: {peaks['FWHM']['fwhm']:.2f}")
        for peak_name in ["Peak 1", "Peak 2", "Peak 3", "Peak F"]:
            print(
                f"{peak_name}: Height={peaks[peak_name]['peak_height']:.4f}, "
                f"Temperature={peaks[peak_name]['peak_temp']:.2f}°C"
            )


def test_r_integration_if_available():
    """Test integration with R if available."""
    # Skip if R is not available
    try:
        import rpy2  # type: ignore

        print(f"rpy2 version: {rpy2.__version__}")
        r_available = True
    except ImportError:
        r_available = False
        pytest.skip("rpy2 not available, skipping R integration test")

    if not r_available:
        return

    # Create test data
    data = create_complex_thermogram()

    # 1. Detect endpoints
    endpoints = detect_endpoints(data, w=45)

    # 2. Subtract baseline using R integration
    from thermogram_baseline.baseline import subtract_baseline

    # Try with R spline
    baseline_r = subtract_baseline(
        data, endpoints.lower, endpoints.upper, plot=True, use_r=True
    )

    # And with Python spline
    baseline_py = subtract_baseline(
        data, endpoints.lower, endpoints.upper, plot=True, use_r=False
    )

    # Results should be similar
    # Extract dCp values
    # Need to handle both plot = True and False cases
    if isinstance(baseline_r, pl.DataFrame):
        # plot = False
        r_values = baseline_r.select("dCp").to_numpy().flatten()
        py_values = baseline_py.select("dCp").to_numpy().flatten()
    else:
        # plot = True, we have a tuple with the DataFrame and the plot
        r_values = baseline_r[0].select("dCp").to_numpy().flatten()
        py_values = baseline_py[0].select("dCp").to_numpy().flatten()

    # Calculate differences
    diffs = np.abs(r_values - py_values)
    mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)

    print("\nR vs Python implementation differences:")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Maximum absolute difference: {max_diff:.6f}")

    # Results should be close but not identical
    assert (
        mean_diff < 0.05
    ), "Large average difference between R and Python implementations"
