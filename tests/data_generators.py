"""
Test data generation utilities for thermogram analysis.
"""

import numpy as np
import polars as pl


def create_basic_thermogram(n_points=100, noise_level=0.01, random_seed=None):
    """
    Create basic thermogram data with standard peaks.

    Args:
        n_points: Number of data points
        noise_level: Level of random noise to add
        random_seed: Random seed for reproducibility

    Returns:
        polars.DataFrame with Temperature and dCp columns
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create temperature range
    temps = np.linspace(45, 90, n_points)

    # Create peaks
    peak1 = 0.3 * np.exp(-0.5 * ((temps - 63) / 2) ** 2)  # Peak at ~63°C
    peak2 = 0.2 * np.exp(-0.5 * ((temps - 70) / 2) ** 2)  # Peak at ~70°C
    peak3 = 0.15 * np.exp(-0.5 * ((temps - 77) / 2.5) ** 2)  # Peak at ~77°C

    # Add baseline
    baseline = 0.02 * (temps - 65)

    # Add noise
    noise = noise_level * np.random.randn(n_points)

    # Combine components
    dcp = peak1 + peak2 + peak3 + baseline + noise

    # Create DataFrame
    return pl.DataFrame({"Temperature": temps, "dCp": dcp})


def create_edge_case_thermogram(case_type, n_points=100, random_seed=None):
    """
    Create edge case thermogram data for testing.

    Args:
        case_type: Type of edge case ('sparse', 'noisy', 'flat', 'single_peak')
        n_points: Number of data points
        random_seed: Random seed for reproducibility

    Returns:
        polars.DataFrame with Temperature and dCp columns
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create temperature range
    temps = np.linspace(45, 90, n_points)

    if case_type == "sparse":
        # Create sparse data by randomly selecting 20% of points
        indices = np.sort(np.random.choice(n_points, n_points // 5, replace=False))
        temps = temps[indices]
        # Create basic signal
        peak = 0.3 * np.exp(-0.5 * ((temps - 70) / 3) ** 2)
        baseline = 0.02 * (temps - 65)
        noise = 0.01 * np.random.randn(len(temps))
        dcp = peak + baseline + noise

    elif case_type == "noisy":
        # Create very noisy data
        peak1 = 0.3 * np.exp(-0.5 * ((temps - 63) / 2) ** 2)
        peak2 = 0.2 * np.exp(-0.5 * ((temps - 70) / 2) ** 2)
        baseline = 0.02 * (temps - 65)
        # Add high noise level
        noise = 0.2 * np.random.randn(n_points)
        dcp = peak1 + peak2 + baseline + noise

    elif case_type == "flat":
        # Create nearly flat data
        baseline = 0.02 * (temps - 65)
        noise = 0.01 * np.random.randn(n_points)
        dcp = baseline + noise

    elif case_type == "single_peak":
        # Create data with a single sharp peak
        peak = 0.5 * np.exp(-0.5 * ((temps - 65) / 1) ** 2)  # Narrow peak
        baseline = 0.02 * (temps - 65)
        noise = 0.01 * np.random.randn(n_points)
        dcp = peak + baseline + noise

    else:
        raise ValueError(f"Unknown case type: {case_type}")

    # Create DataFrame
    return pl.DataFrame({"Temperature": temps, "dCp": dcp})


def save_test_data(data, filename, directory="tests/data"):
    """
    Save test data to a CSV file.

    Args:
        data: polars.DataFrame to save
        filename: Name of the file
        directory: Directory to save in
    """
    import os

    os.makedirs(directory, exist_ok=True)

    # Save to CSV
    data.write_csv(f"{directory}/{filename}")
    print(f"Saved test data to {directory}/{filename}")
