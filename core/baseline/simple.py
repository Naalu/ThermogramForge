# core/baseline/simple.py
import pandas as pd


def simple_baseline_subtraction(
    data: pd.DataFrame, lower_temp: float, upper_temp: float
) -> pd.DataFrame:
    """
    Perform a simple linear baseline subtraction.

    This is a simplified version that just connects the endpoints with a straight line.
    We'll implement more sophisticated methods incrementally.

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
