#!/usr/bin/env python3
"""
Script to generate sample thermogram data for testing.

This script creates synthetic thermogram data that mimics real thermograms,
with peaks at typical temperatures and various levels of noise.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.subplots as sp  # type: ignore
import polars as pl


def generate_synthetic_thermogram(
    temp_range: Optional[np.ndarray] = None,
    peak1_params: Tuple[float, float, float] = (63, 2.0, 0.3),
    peak2_params: Tuple[float, float, float] = (70, 2.0, 0.2),
    peak3_params: Tuple[float, float, float] = (77, 2.5, 0.15),
    baseline_slope: float = 0.02,
    baseline_intercept: float = 0.0,
    noise_level: float = 0.002,
    random_seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Generate a synthetic thermogram with multiple peaks.

    Args:
        temp_range: Array of temperatures, default is 45-90°C at 0.1°C increments
        peak1_params: Parameters for first peak (center, width, height)
        peak2_params: Parameters for second peak
        peak3_params: Parameters for third peak
        baseline_slope: Slope of linear baseline
        baseline_intercept: Intercept of linear baseline
        noise_level: Standard deviation of Gaussian noise
        random_seed: Seed for random number generator

    Returns:
        DataFrame with Temperature and dCp columns
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create temperature range if not provided
    if temp_range is None:
        temp_range = np.linspace(45, 90, 450)

    # Create peaks (Gaussian curves)
    peak1_center, peak1_width, peak1_height = peak1_params
    peak2_center, peak2_width, peak2_height = peak2_params
    peak3_center, peak3_width, peak3_height = peak3_params

    peak1 = peak1_height * np.exp(
        -0.5 * ((temp_range - peak1_center) / peak1_width) ** 2
    )
    peak2 = peak2_height * np.exp(
        -0.5 * ((temp_range - peak2_center) / peak2_width) ** 2
    )
    peak3 = peak3_height * np.exp(
        -0.5 * ((temp_range - peak3_center) / peak3_width) ** 2
    )

    # Create linear baseline
    baseline = baseline_slope * (temp_range - 65) + baseline_intercept

    # Add noise
    noise = noise_level * np.random.randn(len(temp_range))

    # Combine components
    dcp = peak1 + peak2 + peak3 + baseline + noise

    # Create DataFrame
    return pl.DataFrame({"Temperature": temp_range, "dCp": dcp})


def main() -> None:
    """Generate and save sample thermogram data."""
    # Create output directory
    data_dir = Path("data/generated")
    data_dir.mkdir(exist_ok=True, parents=True)

    # Generate a standard thermogram
    print("Generating standard thermogram...")
    standard = generate_synthetic_thermogram(random_seed=42)
    standard.write_csv(data_dir / "standard_thermogram.csv")

    # Generate a thermogram with more noise
    print("Generating noisy thermogram...")
    noisy = generate_synthetic_thermogram(noise_level=0.05, random_seed=43)
    noisy.write_csv(data_dir / "noisy_thermogram.csv")

    # Generate a thermogram with modified peak heights
    print("Generating thermogram with modified peaks...")
    modified = generate_synthetic_thermogram(
        peak1_params=(63, 2.0, 0.4),  # Higher peak 1
        peak2_params=(70, 2.0, 0.15),  # Lower peak 2
        peak3_params=(77, 2.5, 0.25),  # Higher peak 3
        random_seed=44,
    )
    modified.write_csv(data_dir / "modified_thermogram.csv")

    # Generate thermogram with shifted peaks
    print("Generating thermogram with shifted peaks...")
    shifted = generate_synthetic_thermogram(
        peak1_params=(61, 2.0, 0.3),  # Shifted left
        peak2_params=(72, 2.0, 0.2),  # Shifted right
        peak3_params=(79, 2.5, 0.15),  # Shifted right
        random_seed=45,
    )
    shifted.write_csv(data_dir / "shifted_thermogram.csv")

    # Create a 2x2 subplot with Plotly
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Standard Thermogram",
            "Noisy Thermogram",
            "Modified Peaks",
            "Shifted Peaks",
        ),
    )

    # Add data to each subplot
    fig.add_trace(
        go.Scatter(
            x=standard["Temperature"],
            y=standard["dCp"],
            mode="markers",
            marker=dict(size=3, color="blue"),
            name="Standard",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=noisy["Temperature"],
            y=noisy["dCp"],
            mode="markers",
            marker=dict(size=3, color="red"),
            name="Noisy",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=modified["Temperature"],
            y=modified["dCp"],
            mode="markers",
            marker=dict(size=3, color="green"),
            name="Modified",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=shifted["Temperature"],
            y=shifted["dCp"],
            mode="markers",
            marker=dict(size=3, color="purple"),
            name="Shifted",
        ),
        row=2,
        col=2,
    )

    # Update layout with common axis labels
    fig.update_layout(
        title="Synthetic Thermogram Samples",
        height=800,
        width=1000,
        showlegend=False,
    )

    # Update all x and y axis labels
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="dCp (kJ/mol·K)")

    # Save as HTML for interactive viewing
    fig.write_html(data_dir / "thermogram_samples.html")

    # Save as PNG for static viewing
    fig.write_image(data_dir / "thermogram_samples.png")

    print(f"Saved interactive plot to {data_dir}/thermogram_samples.html")
    print(f"Saved static plot to {data_dir}/thermogram_samples.png")
    print(f"Generated 4 sample thermograms in {data_dir}")


if __name__ == "__main__":
    main()
