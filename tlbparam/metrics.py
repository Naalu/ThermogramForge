"""
Metrics module for thermogram data analysis.

This module implements the calculation of various metrics from thermogram data,
including peak metrics, ratio metrics, valley metrics, and global metrics.
"""

from typing import Dict, List, Mapping, Optional, Union, cast

import numpy as np
import polars as pl

from tlbparam.peak_detection import gen_fwhm, gen_peak

# Type aliases for clarity
MetricsDict = Dict[str, float]
ExtendedMetricsDict = Dict[str, Union[str, float]]


def calculate_peak_metrics(
    data: pl.DataFrame, temp_col: str = "Temperature", value_col: str = "dCp"
) -> MetricsDict:
    """
    Calculate peak-related metrics for thermogram data.

    Args:
        data: DataFrame with thermogram data
        temp_col: Name of temperature column
        value_col: Name of value column

    Returns:
        Dictionary with peak metrics
    """
    # Extract data
    temperatures = data.select(pl.col(temp_col)).to_numpy().flatten()
    values = data.select(pl.col(value_col)).to_numpy().flatten()

    # Define peak ranges based on standard temperature regions
    peak_ranges = {
        "Peak 1": (60.0, 66.0),  # Albumin
        "Peak 2": (67.0, 73.0),  # Alpha-2
        "Peak 3": (73.0, 81.0),  # Gamma
        "Peak F": (50.0, 54.0),  # Fibrinogen
    }

    # Calculate peak metrics
    metrics: MetricsDict = {}
    for peak_name, peak_range in peak_ranges.items():
        peak_info = gen_peak(values, temperatures, peak_range)
        metrics[peak_name] = peak_info["peak_height"]
        metrics[f"T{peak_name}"] = peak_info["peak_temp"]

    return metrics


def calculate_ratio_metrics(peak_metrics: MetricsDict) -> MetricsDict:
    """
    Calculate ratio metrics between peaks.

    Args:
        peak_metrics: Dictionary with peak metrics

    Returns:
        Dictionary with ratio metrics
    """
    metrics: MetricsDict = {}

    # Avoid division by zero
    if peak_metrics.get("Peak 2", 0) != 0:
        metrics["Peak 1 / Peak 2"] = peak_metrics.get("Peak 1", 0) / peak_metrics.get(
            "Peak 2", 1
        )
    else:
        metrics["Peak 1 / Peak 2"] = 0

    if peak_metrics.get("Peak 3", 0) != 0:
        metrics["Peak 1 / Peak 3"] = peak_metrics.get("Peak 1", 0) / peak_metrics.get(
            "Peak 3", 1
        )
        metrics["Peak 2 / Peak 3"] = peak_metrics.get("Peak 2", 0) / peak_metrics.get(
            "Peak 3", 1
        )
    else:
        metrics["Peak 1 / Peak 3"] = 0
        metrics["Peak 2 / Peak 3"] = 0

    return metrics


def calculate_valley_metrics(
    data: pl.DataFrame,
    peak_metrics: MetricsDict,
    temp_col: str = "Temperature",
    value_col: str = "dCp",
) -> MetricsDict:
    """
    Calculate valley-related metrics for thermogram data.

    Args:
        data: DataFrame with thermogram data
        peak_metrics: Dictionary with peak metrics
        temp_col: Name of temperature column
        value_col: Name of value column

    Returns:
        Dictionary with valley metrics
    """
    metrics: MetricsDict = {}

    # Extract data
    temperatures = data.select(pl.col(temp_col)).to_numpy().flatten()
    values = data.select(pl.col(value_col)).to_numpy().flatten()

    # Find valley between Peak 1 and Peak 2
    t_peak1 = peak_metrics.get("TPeak 1", 63)
    t_peak2 = peak_metrics.get("TPeak 2", 70)

    # Search for minimum between these temperatures
    in_range = (temperatures >= t_peak1) & (temperatures <= t_peak2)

    if np.any(in_range):
        temp_in_range = temperatures[in_range]
        values_in_range = values[in_range]

        min_idx = np.argmin(values_in_range)
        metrics["V1.2"] = float(values_in_range[min_idx])
        metrics["TV1.2"] = float(temp_in_range[min_idx])

        # Calculate valley-to-peak ratios
        if peak_metrics.get("Peak 1", 0) != 0:
            metrics["V1.2 / Peak 1"] = metrics["V1.2"] / peak_metrics["Peak 1"]
        else:
            metrics["V1.2 / Peak 1"] = 0

        if peak_metrics.get("Peak 2", 0) != 0:
            metrics["V1.2 / Peak 2"] = metrics["V1.2"] / peak_metrics["Peak 2"]
        else:
            metrics["V1.2 / Peak 2"] = 0

        if peak_metrics.get("Peak 3", 0) != 0:
            metrics["V1.2 / Peak 3"] = metrics["V1.2"] / peak_metrics["Peak 3"]
        else:
            metrics["V1.2 / Peak 3"] = 0
    else:
        # No valley found
        metrics["V1.2"] = 0.0
        metrics["TV1.2"] = 0.0
        metrics["V1.2 / Peak 1"] = 0.0
        metrics["V1.2 / Peak 2"] = 0.0
        metrics["V1.2 / Peak 3"] = 0.0

    return metrics


def calculate_global_metrics(
    data: pl.DataFrame, temp_col: str = "Temperature", value_col: str = "dCp"
) -> MetricsDict:
    """
    Calculate global metrics for thermogram data.

    Args:
        data: DataFrame with thermogram data
        temp_col: Name of temperature column
        value_col: Name of value column

    Returns:
        Dictionary with global metrics
    """
    metrics: MetricsDict = {}

    # Extract data
    temperatures = data.select(pl.col(temp_col)).to_numpy().flatten()
    values = data.select(pl.col(value_col)).to_numpy().flatten()

    if len(values) > 0:
        # Basic statistics
        metrics["Max"] = float(np.max(values))
        max_idx = np.argmax(values)
        metrics["TMax"] = float(temperatures[max_idx])

        metrics["Min"] = float(np.min(values))
        min_idx = np.argmin(values)
        metrics["TMin"] = float(temperatures[min_idx])

        metrics["Median"] = float(np.median(values))

        # First moment (weighted average of temperature)
        if np.sum(values) != 0:
            metrics["TFM"] = float(np.sum(temperatures * values) / np.sum(values))
        else:
            metrics["TFM"] = float(np.mean(temperatures))

        # Area under the curve (trapezoid rule)
        from scipy.integrate import trapezoid  # Use trapezoid instead of trapz

        metrics["Area"] = float(trapezoid(values, temperatures))

        # FWHM
        try:
            metrics["Width"] = gen_fwhm(values, temperatures)
        except ValueError:
            metrics["Width"] = 0.0
    else:
        # Empty data
        metrics["Max"] = 0.0
        metrics["TMax"] = 0.0
        metrics["Min"] = 0.0
        metrics["TMin"] = 0.0
        metrics["Median"] = 0.0
        metrics["TFM"] = 0.0
        metrics["Area"] = 0.0
        metrics["Width"] = 0.0

    return metrics


def generate_summary(
    data: pl.DataFrame,
    temp_col: str = "Temperature",
    value_col: str = "dCp",
    sample_id_col: Optional[str] = None,
) -> ExtendedMetricsDict:
    """
    Generate a summary of thermogram metrics.

    This function calculates all standard thermogram metrics, including
    peak metrics, ratio metrics, valley metrics, and global metrics.

    Args:
        data: DataFrame with thermogram data
        temp_col: Name of temperature column
        value_col: Name of value column
        sample_id_col: Name of sample ID column (if present)

    Returns:
        Dictionary with all metrics
    """
    # Initialize metrics dictionary
    metrics: ExtendedMetricsDict = {}

    # Add sample ID if present
    if sample_id_col and sample_id_col in data.columns:
        sample_ids = data.select(pl.col(sample_id_col)).unique().to_numpy().flatten()
        if len(sample_ids) == 1:
            metrics["SampleID"] = str(sample_ids[0])

    # Calculate peak metrics
    peak_metrics = calculate_peak_metrics(data, temp_col, value_col)
    metrics.update(cast(Mapping[str, Union[str, float]], peak_metrics))

    # Calculate ratio metrics
    ratio_metrics = calculate_ratio_metrics(peak_metrics)
    metrics.update(cast(Mapping[str, Union[str, float]], ratio_metrics))

    # Calculate valley metrics
    valley_metrics = calculate_valley_metrics(data, peak_metrics, temp_col, value_col)
    metrics.update(cast(Mapping[str, Union[str, float]], valley_metrics))

    # Calculate global metrics
    global_metrics = calculate_global_metrics(data, temp_col, value_col)
    metrics.update(cast(Mapping[str, Union[str, float]], global_metrics))

    return metrics


class ThermogramAnalyzer:
    """Class for analyzing thermogram data and calculating metrics."""

    def __init__(self) -> None:
        """Initialize ThermogramAnalyzer."""
        pass

    def calculate_metrics(
        self,
        data: pl.DataFrame,
        temp_col: str = "Temperature",
        value_col: str = "dCp",
        sample_id_col: Optional[str] = None,
    ) -> ExtendedMetricsDict:
        """
        Calculate all thermogram metrics.

        Args:
            data: DataFrame with thermogram data
            temp_col: Name of temperature column
            value_col: Name of value column
            sample_id_col: Name of sample ID column

        Returns:
            Dictionary with calculated metrics
        """
        return generate_summary(data, temp_col, value_col, sample_id_col)

    def batch_calculate_metrics(
        self,
        data_list: List[pl.DataFrame],
        temp_col: str = "Temperature",
        value_col: str = "dCp",
        sample_id_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Calculate metrics for multiple thermograms.

        Args:
            data_list: List of DataFrames with thermogram data
            temp_col: Name of temperature column
            value_col: Name of value column
            sample_id_col: Name of sample ID column

        Returns:
            DataFrame with metrics for all thermograms
        """
        results = []
        for data in data_list:
            metrics = self.calculate_metrics(data, temp_col, value_col, sample_id_col)
            results.append(metrics)

        # Convert list of dictionaries to DataFrame
        if results:
            return pl.DataFrame(results)
        else:
            # Return empty DataFrame with appropriate columns
            return pl.DataFrame()
