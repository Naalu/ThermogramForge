#!/usr/bin/env python3
"""
Script to create mock thermogram reference data.

This script creates simulated thermogram data and R-like output for testing.
"""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import savgol_filter  # type: ignore

# Define output directory
OUTPUT_DIR = Path("tests/data/r_reference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create thermogram-like data
x = np.linspace(45, 90, 200)
# Create a curve with multiple peaks
peak1 = 0.3 * np.exp(-0.5 * ((x - 63) / 2) ** 2)  # Peak ~ 63°C
peak2 = 0.2 * np.exp(-0.5 * ((x - 70) / 2) ** 2)  # Peak ~ 70°C
peak3 = 0.15 * np.exp(-0.5 * ((x - 77) / 2.5) ** 2)  # Peak ~ 77°C
baseline = 0.02 * (x - 65)  # Slight linear baseline

# Set random seed for reproducibility
np.random.seed(42)

# Combine components with noise
y = peak1 + peak2 + peak3 + baseline + 0.02 * np.random.randn(len(x))

# Create simulated fitted values (slightly smoother version of y)
fitted = savgol_filter(y, window_length=21, polyorder=3)

# Create thermogram data DataFrame
thermogram_df = pl.DataFrame({"x": x, "y": y, "fitted": fitted})

# Save thermogram fits
thermogram_df.write_csv(OUTPUT_DIR / "thermogram_fits.csv")
print(f"Created {OUTPUT_DIR / 'thermogram_fits.csv'}")

# Create params DataFrame with spar value
params_df = pl.DataFrame(
    {"cv": ["TRUE"], "spar": [0.5], "df": [30.0], "lambda": [0.0001]}
)

# Save params
params_df.write_csv(OUTPUT_DIR / "thermogram_params.csv")
print(f"Created {OUTPUT_DIR / 'thermogram_params.csv'}")

print("\nMock thermogram reference data created successfully.")
