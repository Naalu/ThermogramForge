#!/usr/bin/env python3
"""
Generate reference thermogram data for testing.

This script creates more sophisticated thermogram test data with realistic
features for comprehensive testing of the thermogram analysis functionality.
"""

from pathlib import Path

import numpy as np
import polars as pl


def generate_realistic_thermogram(
    temp_range=(45, 90),
    n_points=451,  # 0.1°C step size
    peak_centers=[55, 63, 70, 77],
    peak_heights=[0.1, 0.3, 0.2, 0.15],
    peak_widths=[1.5, 2.0, 2.0, 2.5],
    baseline_slope=0.02,
    baseline_intercept=0.0,
    noise_level=0.01,
    random_seed=None,
):
    """
    Generate a realistic thermogram with multiple peaks and baseline.

    Args:
        temp_range: Tuple of (min_temp, max_temp)
        n_points: Number of temperature points
        peak_centers: List of peak center temperatures
        peak_heights: List of peak heights
        peak_widths: List of peak widths (sigma)
        baseline_slope: Slope of the linear baseline
        baseline_intercept: Intercept of the linear baseline
        noise_level: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with Temperature and dCp columns
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create temperature range
    min_temp, max_temp = temp_range
    temps = np.linspace(min_temp, max_temp, n_points)

    # Initialize dCp with baseline
    dcp = baseline_slope * (temps - 65) + baseline_intercept

    # Add each peak
    for center, height, width in zip(peak_centers, peak_heights, peak_widths):
        peak = height * np.exp(-0.5 * ((temps - center) / width) ** 2)
        dcp += peak

    # Add noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(n_points)
        dcp += noise

    # Create DataFrame
    return pl.DataFrame({"Temperature": temps, "dCp": dcp})


def generate_reference_set():
    """Generate a complete set of reference thermograms."""
    output_dir = Path("tests/data/thermogram_reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standard reference thermogram (used for most tests)
    standard = generate_realistic_thermogram(random_seed=42)
    standard.write_csv(output_dir / "standard_thermogram.csv")
    print(f"Created standard_thermogram.csv with {standard.height} rows")

    # Variations on peak positions
    shifted = generate_realistic_thermogram(
        peak_centers=[53, 61, 72, 79],
        random_seed=42,  # Shifted from standard
    )
    shifted.write_csv(output_dir / "shifted_peaks_thermogram.csv")
    print(f"Created shifted_peaks_thermogram.csv with {shifted.height} rows")

    # Variations on peak heights
    varied_heights = generate_realistic_thermogram(
        peak_heights=[0.05, 0.4, 0.15, 0.25],
        random_seed=42,  # Different heights
    )
    varied_heights.write_csv(output_dir / "varied_heights_thermogram.csv")
    print(f"Created varied_heights_thermogram.csv with {varied_heights.height} rows")

    # Thermogram with steep baseline
    steep_baseline = generate_realistic_thermogram(
        baseline_slope=0.05,
        random_seed=42,  # Steeper baseline
    )
    steep_baseline.write_csv(output_dir / "steep_baseline_thermogram.csv")
    print(f"Created steep_baseline_thermogram.csv with {steep_baseline.height} rows")

    # Thermogram with nonlinear baseline
    # Create temp array first for nonlinear baseline
    temps = np.linspace(45, 90, 451)
    nonlinear_baseline = generate_realistic_thermogram(
        baseline_slope=0.0,
        random_seed=42,  # No linear component
    )
    # Add nonlinear baseline directly to dCp
    nonlinear_dcp = nonlinear_baseline.select("dCp").to_numpy().flatten()
    nonlinear_dcp += 0.0003 * (temps - 65) ** 2  # Quadratic term

    nonlinear_df = pl.DataFrame({"Temperature": temps, "dCp": nonlinear_dcp})
    nonlinear_df.write_csv(output_dir / "nonlinear_baseline_thermogram.csv")
    print(f"Created nonlinear_baseline_thermogram.csv with {nonlinear_df.height} rows")

    # Thermogram with varying noise levels
    for noise in [0.005, 0.02, 0.05]:
        noisy = generate_realistic_thermogram(noise_level=noise, random_seed=42)
        noisy.write_csv(output_dir / f"noise_{int(noise * 1000)}_thermogram.csv")
        print(
            f"Created noise_{int(noise * 1000)}_thermogram.csv with {noisy.height} rows"
        )

    print("Reference thermogram generation complete")


if __name__ == "__main__":
    generate_reference_set()
