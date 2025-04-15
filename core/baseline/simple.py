# core/baseline/simple.py
"""Provides a simple linear baseline subtraction method."""

import pandas as pd
from pandas.api.types import is_numeric_dtype


def simple_baseline_subtraction(
    data: pd.DataFrame, lower_temp: float, upper_temp: float
) -> pd.DataFrame:
    """Performs simple linear baseline subtraction on thermogram data.

    Calculates a linear baseline by connecting the dCp values at the specified
    lower and upper temperature endpoints and subtracts this line from the
    original dCp data.

    Args:
        data: DataFrame containing at least 'Temperature' and 'dCp' columns.
              'Temperature' and 'dCp' must be numeric.
        lower_temp: The lower temperature boundary for baseline calculation.
        upper_temp: The upper temperature boundary for baseline calculation.

    Returns:
        A new DataFrame with two additional columns:
        'dCp_baseline': The calculated linear baseline value for each point.
        'dCp_subtracted': The original 'dCp' value minus the 'dCp_baseline'.

    Raises:
        ValueError: If 'Temperature' or 'dCp' columns are missing, not numeric,
                    or if the lower/upper temperatures are invalid or result
                    in division by zero.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    if "Temperature" not in data.columns or "dCp" not in data.columns:
        raise ValueError("DataFrame must contain 'Temperature' and 'dCp' columns.")
    if not is_numeric_dtype(data["Temperature"]) or not is_numeric_dtype(data["dCp"]):
        raise ValueError("'Temperature' and 'dCp' columns must be numeric.")
    if data.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    if lower_temp >= upper_temp:
        raise ValueError("Lower temperature must be less than upper temperature.")

    # Make a copy to avoid modifying the original DataFrame
    result = data.copy()

    # Find the indices closest to the specified lower and upper temperatures
    try:
        lower_idx: int = (result["Temperature"] - lower_temp).abs().idxmin()
        upper_idx: int = (result["Temperature"] - upper_temp).abs().idxmin()
    except Exception as e:
        raise ValueError(f"Error finding temperature indices: {e}")

    lower_temp_actual: float = result.loc[lower_idx, "Temperature"]
    upper_temp_actual: float = result.loc[upper_idx, "Temperature"]
    lower_dcp: float = result.loc[lower_idx, "dCp"]
    upper_dcp: float = result.loc[upper_idx, "dCp"]

    # Prevent division by zero if the closest points have the same temperature
    if pd.isclose(lower_temp_actual, upper_temp_actual):
        raise ValueError(
            "Identified lower and upper temperature points are identical or too close, "
            "cannot calculate a linear baseline."
        )

    # Calculate slope (m) and intercept (b) for the linear baseline (y = mx + b)
    slope: float = (upper_dcp - lower_dcp) / (upper_temp_actual - lower_temp_actual)
    intercept: float = lower_dcp - slope * lower_temp_actual

    # Calculate the baseline value for each temperature point
    baseline: pd.Series = slope * result["Temperature"] + intercept

    # Subtract the baseline from the original dCp values
    result["dCp_baseline"] = baseline
    result["dCp_subtracted"] = result["dCp"] - baseline

    return result
