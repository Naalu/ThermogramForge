"""Utilities for generating test data for thermogram analysis tests."""

from typing import Tuple

import numpy as np
import polars as pl

from packages.thermogram_baseline.thermogram_baseline.types import ThermogramData


def generate_simple_thermogram(
    min_temp: float = 45.0,
    max_temp: float = 90.0,
    num_points: int = 451,
    noise_level: float = 0.02,
    baseline_slope: float = 0.01,
    baseline_intercept: float = 0.0,
    peak_center: float = 70.0,
    peak_height: float = 1.0,
    peak_width: float = 5.0,
    seed: int = 42,
) -> ThermogramData:
    """
    Generate a simple thermogram with a Gaussian peak and linear baseline.

    Args:
        min_temp: Minimum temperature
        max_temp: Maximum temperature
        num_points: Number of temperature points
        noise_level: Standard deviation of Gaussian noise
        baseline_slope: Slope of the linear baseline
        baseline_intercept: Intercept of the linear baseline
        peak_center: Center temperature of the Gaussian peak
        peak_height: Height of the Gaussian peak
        peak_width: Width (standard deviation) of the Gaussian peak
        seed: Random seed for reproducibility

    Returns:
        ThermogramData object containing the generated thermogram
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate temperature array
    temperatures = np.linspace(min_temp, max_temp, num_points)

    # Generate linear baseline
    baseline = baseline_slope * temperatures + baseline_intercept

    # Generate Gaussian peak
    peak = peak_height * np.exp(
        -((temperatures - peak_center) ** 2) / (2 * peak_width**2)
    )

    # Add noise
    noise = np.random.normal(0, noise_level, num_points)

    # Combine components
    values = baseline + peak + noise

    return ThermogramData(temperature=temperatures, dcp=values)


def generate_multi_peak_thermogram(
    min_temp: float = 45.0,
    max_temp: float = 90.0,
    num_points: int = 451,
    noise_level: float = 0.02,
    baseline_slope: float = 0.01,
    baseline_intercept: float = 0.0,
    peak_centers: Tuple[float, ...] = (60.0, 70.0, 80.0),
    peak_heights: Tuple[float, ...] = (0.8, 1.0, 0.6),
    peak_widths: Tuple[float, ...] = (3.0, 3.5, 4.0),
    seed: int = 42,
) -> ThermogramData:
    """
    Generate a thermogram with multiple Gaussian peaks and a linear baseline.

    Args:
        min_temp: Minimum temperature
        max_temp: Maximum temperature
        num_points: Number of temperature points
        noise_level: Standard deviation of Gaussian noise
        baseline_slope: Slope of the linear baseline
        baseline_intercept: Intercept of the linear baseline
        peak_centers: Center temperatures of the Gaussian peaks
        peak_heights: Heights of the Gaussian peaks
        peak_widths: Widths (standard deviations) of the Gaussian peaks
        seed: Random seed for reproducibility

    Returns:
        ThermogramData object containing the generated thermogram
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate temperature array
    temperatures = np.linspace(min_temp, max_temp, num_points)

    # Generate linear baseline
    baseline = baseline_slope * temperatures + baseline_intercept

    # Start with just the baseline
    values = baseline.copy()

    # Add each peak
    for center, height, width in zip(peak_centers, peak_heights, peak_widths):
        peak = height * np.exp(-((temperatures - center) ** 2) / (2 * width**2))
        values += peak

    # Add noise
    noise = np.random.normal(0, noise_level, num_points)
    values += noise

    return ThermogramData(temperature=temperatures, dcp=values)


def generate_real_like_thermogram(
    min_temp: float = 45.0,
    max_temp: float = 90.0,
    num_points: int = 451,
    noise_level: float = 0.01,
    seed: int = 42,
) -> ThermogramData:
    """
    Generate a thermogram that mimics real thermogram features.

    Args:
        min_temp: Minimum temperature
        max_temp: Maximum temperature
        num_points: Number of temperature points
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        ThermogramData object containing the generated thermogram
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate temperature array
    temperatures = np.linspace(min_temp, max_temp, num_points)

    # Generate baseline - slight curvature
    baseline = 0.01 * temperatures + 0.0001 * temperatures**2 - 0.5

    # Generate multiple peaks with different shapes to mimic protein components
    peak1 = 0.9 * np.exp(
        -((temperatures - 61) ** 2) / (2 * 2.5**2)
    )  # Sharp peak at 61°C
    peak2 = 1.2 * np.exp(
        -((temperatures - 68) ** 2) / (2 * 3.0**2)
    )  # Main peak at 68°C
    peak3 = 0.7 * np.exp(
        -((temperatures - 76) ** 2) / (2 * 4.0**2)
    )  # Broader peak at 76°C

    # Add asymmetry to peaks
    skew_factor = (temperatures - 68) / 20  # Skew more towards higher temperatures
    asymmetry = 0.2 * skew_factor * np.exp(-((temperatures - 72) ** 2) / (2 * 8.0**2))

    # Add noise
    noise = np.random.normal(0, noise_level, num_points)

    # Combine components
    values = baseline + peak1 + peak2 + peak3 + asymmetry + noise

    return ThermogramData(temperature=temperatures, dcp=values)


def thermogram_to_dataframe(data: ThermogramData) -> pl.DataFrame:
    """
    Convert ThermogramData to a Polars DataFrame.

    Args:
        data: ThermogramData object

    Returns:
        Polars DataFrame with Temperature and dCp columns
    """
    return pl.DataFrame({"Temperature": data.temperature, "dCp": data.dcp})
