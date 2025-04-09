"""Utilities for baseline calculations on thermogram data."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def find_value_at_temp(
    df: pd.DataFrame, temp: float, temp_col: str = "Temperature", val_col: str = "dCp"
) -> Optional[float]:
    """Finds the value in val_col closest to the given temperature.

    Args:
        df: DataFrame containing temperature and value data.
        temp: Target temperature.
        temp_col: Name of the temperature column.
        val_col: Name of the value column.

    Returns:
        The value from val_col corresponding to the temperature closest
        to the target temp, or None if not found or columns are missing.
    """
    if temp_col not in df.columns or val_col not in df.columns:
        logger.warning(
            f"Missing required columns '{temp_col}' or '{val_col}' in DataFrame."
        )
        return None
    try:
        closest_index = (df[temp_col] - temp).abs().idxmin()
        closest_value = df.loc[closest_index, val_col]
        return float(closest_value)
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Error finding value at temp {temp:.2f}: {e}")
        return None


def calculate_linear_baseline(
    lower_temp: float, lower_val: float, upper_temp: float, upper_val: float
) -> Tuple[float, float]:
    """Calculates the slope and intercept of a linear baseline.

    Args:
        lower_temp: Temperature of the lower baseline point.
        lower_val: Value (e.g., dCp) of the lower baseline point.
        upper_temp: Temperature of the upper baseline point.
        upper_val: Value (e.g., dCp) of the upper baseline point.

    Returns:
        A tuple containing the slope (m) and intercept (b) of the line.
        Returns (0.0, lower_val) if temperatures are identical to avoid division by zero.
    """
    if upper_temp == lower_temp:
        logger.warning(
            "Upper and lower baseline temperatures are identical. Using horizontal baseline."
        )
        return 0.0, lower_val  # Slope is 0, intercept is the value at that temp

    try:
        slope = (upper_val - lower_val) / (upper_temp - lower_temp)
        intercept = lower_val - slope * lower_temp
        return slope, intercept
    except Exception as e:  # Catch any potential calculation errors
        logger.error(f"Error calculating linear baseline parameters: {e}")
        # Fallback to horizontal baseline at lower point
        return 0.0, lower_val


def simple_baseline_subtraction(
    df_raw: pd.DataFrame, lower_endpoint_temp: float, upper_endpoint_temp: float
) -> Optional[pd.DataFrame]:
    """Performs simple linear baseline subtraction.

    Finds the dCp values at the temperatures closest to the provided lower and
    upper endpoints, calculates a linear baseline between these points, and
    subtracts this baseline from the original dCp values.

    Args:
        df_raw: DataFrame with raw thermogram data (must include 'Temperature' and 'dCp').
        lower_endpoint_temp: Temperature for the lower baseline endpoint.
        upper_endpoint_temp: Temperature for the upper baseline endpoint.

    Returns:
        A new DataFrame with added 'dCp_baseline' and 'dCp_subtracted' columns,
        or None if subtraction could not be performed (e.g., missing data, endpoints invalid).
    """
    if (
        df_raw.empty
        or "Temperature" not in df_raw.columns
        or "dCp" not in df_raw.columns
    ):
        logger.error(
            "Input DataFrame for baseline subtraction is empty or missing required columns."
        )
        return None

    # Ensure endpoints are ordered correctly
    if lower_endpoint_temp > upper_endpoint_temp:
        logger.warning(
            "Lower endpoint temp > upper endpoint temp. Swapping them for baseline calculation."
        )
        lower_endpoint_temp, upper_endpoint_temp = (
            upper_endpoint_temp,
            lower_endpoint_temp,
        )

    # Find dCp values at the endpoint temperatures
    lower_dcp_at_endpoint = find_value_at_temp(df_raw, lower_endpoint_temp)
    upper_dcp_at_endpoint = find_value_at_temp(df_raw, upper_endpoint_temp)

    if lower_dcp_at_endpoint is None or upper_dcp_at_endpoint is None:
        logger.error(
            "Could not find dCp values at one or both baseline endpoint temperatures."
        )
        return None

    # Create a copy to avoid modifying the original DataFrame
    result_df = df_raw.copy()

    try:
        # Calculate linear baseline parameters (slope m, intercept b)
        slope, intercept = calculate_linear_baseline(
            lower_endpoint_temp,
            lower_dcp_at_endpoint,
            upper_endpoint_temp,
            upper_dcp_at_endpoint,
        )

        # Calculate baseline values for all temperatures
        result_df["dCp_baseline"] = slope * result_df["Temperature"] + intercept

        # Perform subtraction
        result_df["dCp_subtracted"] = result_df["dCp"] - result_df["dCp_baseline"]
        logger.debug("Baseline subtraction performed successfully.")

    except Exception as e:
        logger.error(
            f"Error during baseline subtraction calculation: {e}", exc_info=True
        )
        # Assign NaN or handle error appropriately if calculation fails
        result_df["dCp_baseline"] = np.nan
        result_df["dCp_subtracted"] = np.nan
        return None  # Indicate failure

    return result_df
