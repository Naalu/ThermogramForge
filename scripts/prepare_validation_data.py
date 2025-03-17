#!/usr/bin/env python3
"""
Script to prepare validation data for R comparison.

This script:
1. Creates data directories for R reference data
2. Generates synthetic test datasets to be processed by R
3. Provides instructions for running the R validation script
"""

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, Tuple, TypeAlias, Union

import numpy as np
import polars as pl

# Update DataGenerator type to support keyword arguments
DataGenerator: TypeAlias = Callable[..., Tuple[np.ndarray, np.ndarray]]


def create_sine_wave_data(
    n: int = 100, noise_level: float = 0.1, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sine wave test data.

    Args:
        n: Number of data points
        noise_level: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of x and y arrays
    """
    np.random.seed(random_seed)
    x = np.linspace(0, 10, n)
    y = np.sin(x) + noise_level * np.random.randn(n)
    return x, y


def create_exponential_data(
    n: int = 100, noise_level: float = 0.5, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create exponential test data.

    Args:
        n: Number of data points
        noise_level: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of x and y arrays
    """
    np.random.seed(random_seed)
    x = np.linspace(0, 5, n)
    y = np.exp(x / 2) + noise_level * np.random.randn(n)
    return x, y


def create_peaks_data(
    n: int = 100, noise_level: float = 0.02, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create thermogram-like data with multiple peaks.

    Args:
        n: Number of data points
        noise_level: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of x and y arrays
    """
    np.random.seed(random_seed)
    x = np.linspace(45, 90, n)
    y = (
        0.3 * np.exp(-0.5 * ((x - 55) / 2) ** 2)
        + 0.2 * np.exp(-0.5 * ((x - 70) / 3) ** 2)
        + 0.15 * np.exp(-0.5 * ((x - 82) / 2) ** 2)
        + noise_level * np.random.randn(n)
    )
    return x, y


def create_flat_data(
    n: int = 100, noise_level: float = 0.1, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create nearly constant data with small variations.

    Args:
        n: Number of data points
        noise_level: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of x and y arrays
    """
    np.random.seed(random_seed)
    x = np.linspace(0, 10, n)
    y = np.ones_like(x) * 5 + noise_level * np.random.randn(n)
    return x, y


def create_noisy_data(
    n: int = 100, noise_level: float = 0.5, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create very noisy data.

    Args:
        n: Number of data points
        noise_level: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of x and y arrays
    """
    np.random.seed(random_seed)
    x = np.linspace(0, 10, n)
    y = np.sin(x) + noise_level * np.random.randn(n)
    return x, y


def create_thermogram_data(
    n: int = 200, noise_level: float = 0.01, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create realistic thermogram test data.

    Args:
        n: Number of data points (default: 200)
        noise_level: Standard deviation of Gaussian noise (default: 0.01)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of x and y arrays (temperature and heat capacity)
    """
    np.random.seed(random_seed)
    x = np.linspace(45, 90, n)
    y = (
        0.3 * np.exp(-0.5 * ((x - 55) / 2.5) ** 2)
        + 0.2 * np.exp(-0.5 * ((x - 67) / 1.8) ** 2)
        + 0.15 * np.exp(-0.5 * ((x - 78) / 2.2) ** 2)
        + noise_level * np.random.randn(len(x))
    )
    return x, y


def main(output_dir: Union[str, Path], skip_confirmation: bool = False) -> None:
    """
    Generate validation data for R comparison.

    Args:
        output_dir: Directory to save generated data files
        skip_confirmation: Whether to skip confirmation prompts
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)  # Add this line to create directory

    # Standard test datasets
    datasets: Dict[str, DataGenerator] = {
        "sine": create_sine_wave_data,
        "exponential": create_exponential_data,
        "peaks": create_peaks_data,
        "flat": create_flat_data,
        "noisy": create_noisy_data,
    }

    # Generate and save standard datasets
    print(f"Generating test datasets in {output_dir}...")
    for name, generator in datasets.items():
        # Standard size (n=100)
        x, y = generator()  # Defaults to n=100
        # Create a polars DataFrame instead of pandas
        df = pl.DataFrame({"x": x, "y": y})
        # Use polars write_csv method
        df.write_csv(output_dir / f"standard_{name}_input.csv")
        print(f"Created standard_{name}_input.csv")

        # Small size (n=20) for a few patterns
        if name in ["sine", "peaks"]:
            x_small, y_small = generator(n=20)
            # Create a polars DataFrame for small datasets
            df_small = pl.DataFrame({"x": x_small, "y": y_small})
            df_small.write_csv(output_dir / f"small_{name}_input.csv")
            print(f"Created small_{name}_input.csv")

    # Generate thermogram data
    x_thermo, y_thermo = create_thermogram_data()
    # Create polars DataFrame for thermogram data
    df_thermo = pl.DataFrame({"x": x_thermo, "y": y_thermo})
    df_thermo.write_csv(output_dir / "thermogram_raw.csv")
    print("Created thermogram_raw.csv")

    # Print instructions for R processing
    print("\nTest datasets have been created. To complete validation setup:")
    print("1. Install R if not already installed")
    print("2. Run the R script to process these datasets:")
    print("   Rscript scripts/generate_r_reference_data.R")
    print(
        "3. This will create reference data for validation in tests/data/r_reference/"
    )
    print("4. Run the validation tests with:")
    print("   python -m pytest tests/thermogram_baseline/test_r_validation.py -v")

    # Ask user if they want to proceed with R script if R is available
    if not skip_confirmation:
        r_available = os.system("Rscript --version > /dev/null 2>&1") == 0
        if r_available:
            proceed = input("\nR is available. Run the R script now? (y/n): ")
            if proceed.lower() == "y":
                print("\nRunning R script...")
                os.system("Rscript scripts/generate_r_reference_data.R")
                print("\nR reference data generation complete.")
                print("You can now run the validation tests.")
        else:
            print("\nR does not appear to be installed or available in your PATH.")
            print("Please install R and run the script manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate validation data for R comparison."
    )
    parser.add_argument(
        "--output-dir",
        default="tests/data/r_reference",
        help="Directory to save test data (default: tests/data/r_reference)",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts",
    )
    args = parser.parse_args()

    main(args.output_dir, args.no_confirm)
