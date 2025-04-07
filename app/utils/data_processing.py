"""
Utilities for data processing.
"""

import pandas as pd


def preprocess_thermogram_data(df):
    """
    Preprocess thermogram data.

    Args:
        df: Raw DataFrame

    Returns:
        Processed DataFrame
    """
    # Check for required columns
    if "Temperature" not in df.columns or "dCp" not in df.columns:
        # Try to guess columns - first column might be temp, second might be dCp
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "Temperature", df.columns[1]: "dCp"})

    # Make sure columns are the right type
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["dCp"] = pd.to_numeric(df["dCp"], errors="coerce")

    # Drop NaN values
    df = df.dropna(subset=["Temperature", "dCp"])

    # Sort by temperature
    df = df.sort_values("Temperature")

    # Reset index
    df = df.reset_index(drop=True)

    return df


def simple_baseline_subtraction(data, lower_temp, upper_temp):
    """
    Perform a simple linear baseline subtraction.

    This is a simplified version that just connects the endpoints with a straight line.

    Args:
        data: DataFrame with Temperature and dCp columns
        lower_temp: Lower temperature endpoint
        upper_temp: Upper temperature endpoint

    Returns:
        DataFrame with baseline-subtracted data
    """
    # Make a copy to avoid modifying the original
    result = data.copy()

    # Get values at endpoints
    lower_idx = (data["Temperature"] - lower_temp).abs().idxmin()
    upper_idx = (data["Temperature"] - upper_temp).abs().idxmin()

    lower_temp_actual = data.loc[lower_idx, "Temperature"]
    upper_temp_actual = data.loc[upper_idx, "Temperature"]

    lower_dcp = data.loc[lower_idx, "dCp"]
    upper_dcp = data.loc[upper_idx, "dCp"]

    # Calculate slope and intercept for linear baseline
    slope = (upper_dcp - lower_dcp) / (upper_temp_actual - lower_temp_actual)
    intercept = lower_dcp - slope * lower_temp_actual

    # Calculate baseline for each point
    baseline = slope * data["Temperature"] + intercept

    # Subtract baseline
    result["dCp_baseline"] = baseline
    result["dCp_subtracted"] = result["dCp"] - baseline

    return result
