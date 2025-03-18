"""
Tests for the metrics module.

This module contains tests for the thermogram metrics calculation functionality,
verifying that it correctly calculates various metrics from thermogram data.
"""

import numpy as np
import polars as pl
import pytest

from tlbparam.metrics import (
    ThermogramAnalyzer,
    calculate_global_metrics,
    calculate_peak_metrics,
    calculate_ratio_metrics,
    calculate_valley_metrics,
    generate_summary,
)


def create_synthetic_thermogram_with_peaks() -> pl.DataFrame:
    """
    Create a synthetic thermogram with well-defined peaks for testing.

    Returns:
        A Polars DataFrame with synthetic thermogram data
    """
    # Create temperature range
    temperatures = np.linspace(45, 90, 451)  # 0.1°C steps

    # Create distinct peaks at known locations
    peak1 = 0.5 * np.exp(-0.5 * ((temperatures - 63) / 1.5) ** 2)  # Peak at 63°C
    peak2 = 0.8 * np.exp(-0.5 * ((temperatures - 70) / 1.5) ** 2)  # Peak at 70°C
    peak3 = 0.6 * np.exp(-0.5 * ((temperatures - 77) / 1.5) ** 2)  # Peak at 77°C
    peak_f = 0.3 * np.exp(
        -0.5 * ((temperatures - 52) / 1.0) ** 2
    )  # Peak at 52°C (Fibrinogen)

    # Add valley between Peak 1 and Peak 2
    valley = 0.2 * np.ones_like(temperatures)
    mask = (temperatures >= 65) & (temperatures <= 68)
    valley[mask] = 0.1

    # Combine peaks with slight baseline
    baseline = 0.0 * (temperatures - 45)  # Flat baseline
    values = peak1 + peak2 + peak3 + peak_f + valley + baseline

    # Create a polars DataFrame
    return pl.DataFrame({"Temperature": temperatures, "dCp": values})


def test_calculate_peak_metrics():
    """Test calculation of peak metrics."""
    # Create test data with known peaks
    data = create_synthetic_thermogram_with_peaks()

    # Calculate peak metrics
    metrics = calculate_peak_metrics(data)

    # Check that all expected metrics are present
    expected_metrics = [
        "Peak 1",
        "TPeak 1",
        "Peak 2",
        "TPeak 2",
        "Peak 3",
        "TPeak 3",
        "Peak F",
        "TPeak F",
    ]
    for metric in expected_metrics:
        assert metric in metrics

    # Check peak temperatures are in expected ranges
    assert 62 <= metrics["TPeak 1"] <= 64
    assert 69 <= metrics["TPeak 2"] <= 71
    assert 76 <= metrics["TPeak 3"] <= 78
    assert 51 <= metrics["TPeak F"] <= 53

    # Check relative peak heights
    assert metrics["Peak 2"] > metrics["Peak 3"]
    assert metrics["Peak 3"] > metrics["Peak 1"]
    assert metrics["Peak 1"] > metrics["Peak F"]


def test_calculate_ratio_metrics():
    """Test calculation of ratio metrics."""
    # Create test peak metrics
    peak_metrics = {
        "Peak 1": 0.5,
        "Peak 2": 0.8,
        "Peak 3": 0.6,
        "TPeak 1": 63.0,
        "TPeak 2": 70.0,
        "TPeak 3": 77.0,
    }

    # Calculate ratio metrics
    metrics = calculate_ratio_metrics(peak_metrics)

    # Check that all expected metrics are present
    expected_metrics = ["Peak 1 / Peak 2", "Peak 1 / Peak 3", "Peak 2 / Peak 3"]
    for metric in expected_metrics:
        assert metric in metrics

    # Check values
    assert metrics["Peak 1 / Peak 2"] == pytest.approx(0.5 / 0.8)
    assert metrics["Peak 1 / Peak 3"] == pytest.approx(0.5 / 0.6)
    assert metrics["Peak 2 / Peak 3"] == pytest.approx(0.8 / 0.6)

    # Test with zero values
    peak_metrics_with_zeros = {
        "Peak 1": 0.5,
        "Peak 2": 0.0,
        "Peak 3": 0.0,
    }
    zero_metrics = calculate_ratio_metrics(peak_metrics_with_zeros)
    assert zero_metrics["Peak 1 / Peak 2"] == 0
    assert zero_metrics["Peak 1 / Peak 3"] == 0
    assert zero_metrics["Peak 2 / Peak 3"] == 0


def test_calculate_valley_metrics():
    """Test calculation of valley metrics."""
    # Create test data with known valley
    data = create_synthetic_thermogram_with_peaks()

    # Create test peak metrics
    peak_metrics = {
        "Peak 1": 0.5,
        "Peak 2": 0.8,
        "Peak 3": 0.6,
        "TPeak 1": 63.0,
        "TPeak 2": 70.0,
        "TPeak 3": 77.0,
    }

    # Calculate valley metrics
    metrics = calculate_valley_metrics(data, peak_metrics)

    # Check that all expected metrics are present
    expected_metrics = [
        "V1.2",
        "TV1.2",
        "V1.2 / Peak 1",
        "V1.2 / Peak 2",
        "V1.2 / Peak 3",
    ]
    for metric in expected_metrics:
        assert metric in metrics

    # Check valley temperature is between Peak 1 and Peak 2
    assert metrics["TV1.2"] > peak_metrics["TPeak 1"]
    assert metrics["TV1.2"] < peak_metrics["TPeak 2"]

    # Valley should be lower than peaks
    assert metrics["V1.2"] < peak_metrics["Peak 1"]
    assert metrics["V1.2"] < peak_metrics["Peak 2"]

    # Check ratio values
    assert metrics["V1.2 / Peak 1"] == pytest.approx(
        metrics["V1.2"] / peak_metrics["Peak 1"]
    )
    assert metrics["V1.2 / Peak 2"] == pytest.approx(
        metrics["V1.2"] / peak_metrics["Peak 2"]
    )
    assert metrics["V1.2 / Peak 3"] == pytest.approx(
        metrics["V1.2"] / peak_metrics["Peak 3"]
    )


def test_calculate_global_metrics():
    """Test calculation of global metrics."""
    # Create test data
    data = create_synthetic_thermogram_with_peaks()

    # Calculate global metrics
    metrics = calculate_global_metrics(data)

    # Check that all expected metrics are present
    expected_metrics = ["Max", "TMax", "Min", "TMin", "Median", "TFM", "Area", "Width"]
    for metric in expected_metrics:
        assert metric in metrics

    # Check values
    # Max should be at Peak 2 (around 70°C)
    assert 69 <= metrics["TMax"] <= 71

    # Area should be positive
    assert metrics["Area"] > 0

    # Width should be positive
    assert metrics["Width"] > 0

    # TFM should be in temperature range
    assert 45 <= metrics["TFM"] <= 90


def test_generate_summary():
    """Test generation of complete metrics summary."""
    # Create test data
    data = create_synthetic_thermogram_with_peaks()

    # Add sample ID
    data = data.with_columns(pl.lit("sample1").alias("SampleID"))

    # Generate summary
    metrics = generate_summary(data, sample_id_col="SampleID")

    # Check that sample ID is included
    assert metrics["SampleID"] == "sample1"

    # Check that metrics from all categories are present
    peak_metrics = ["Peak 1", "TPeak 1", "Peak 2", "TPeak 2"]
    ratio_metrics = ["Peak 1 / Peak 2", "Peak 1 / Peak 3"]
    valley_metrics = ["V1.2", "TV1.2", "V1.2 / Peak 1"]
    global_metrics = ["Max", "TMax", "Area", "Width"]

    for metric in peak_metrics + ratio_metrics + valley_metrics + global_metrics:
        assert metric in metrics


def test_thermogram_analyzer():
    """Test ThermogramAnalyzer class."""
    # Create test data
    data1 = create_synthetic_thermogram_with_peaks()
    data1 = data1.with_columns(pl.lit("sample1").alias("SampleID"))

    data2 = create_synthetic_thermogram_with_peaks()
    # Modify data2 to be slightly different
    data2 = data2.with_columns(pl.col("dCp") * 1.2)
    data2 = data2.with_columns(pl.lit("sample2").alias("SampleID"))

    # Create analyzer
    analyzer = ThermogramAnalyzer()

    # Calculate metrics for single thermogram
    metrics1 = analyzer.calculate_metrics(data1, sample_id_col="SampleID")
    assert metrics1["SampleID"] == "sample1"

    # Calculate metrics for batch
    results = analyzer.batch_calculate_metrics([data1, data2], sample_id_col="SampleID")

    # Check result shape
    assert results.height == 2

    # Check sample IDs
    sample_ids = results.select("SampleID").to_series().to_list()
    assert "sample1" in sample_ids
    assert "sample2" in sample_ids

    # Check metrics columns
    for col in ["Peak 1", "Peak 2", "Peak 3", "Width", "Area"]:
        assert col in results.columns

    # Check that metrics are different between samples
    # Find rows for each sample
    row1 = results.filter(pl.col("SampleID") == "sample1")
    row2 = results.filter(pl.col("SampleID") == "sample2")

    # Peak 2 should be higher in sample2
    assert row2.select("Peak 2")[0, 0] > row1.select("Peak 2")[0, 0]
