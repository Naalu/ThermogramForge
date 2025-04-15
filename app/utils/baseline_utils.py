"""Utility functions supporting baseline calculations and data lookup.

These helpers are primarily used by baseline-related callbacks.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)


def find_value_at_temp(
    df: pd.DataFrame, temp: float, temp_col: str = "Temperature", val_col: str = "dCp"
) -> Optional[float]:
    """Finds the value in `val_col` closest to the given temperature in `temp_col`.

    Uses the absolute difference to find the row with the temperature closest
    to the target `temp` and returns the corresponding value from `val_col`.

    Args:
        df: DataFrame containing temperature and value data.
        temp: Target temperature.
        temp_col: Name of the temperature column.
        val_col: Name of the value column.

    Returns:
        The value from `val_col` corresponding to the temperature closest
        to the target `temp` (as float), or None if columns are missing, not numeric,
        or an error occurs.
    """
    # --- Input Validation --- Start
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.debug("find_value_at_temp: Input DataFrame is empty or invalid.")
        return None
    if temp_col not in df.columns or val_col not in df.columns:
        logger.warning(
            f"find_value_at_temp: Missing required columns '{temp_col}' or '{val_col}'."
        )
        return None
    if not is_numeric_dtype(df[temp_col]) or not is_numeric_dtype(df[val_col]):
        logger.warning(
            f"find_value_at_temp: Column '{temp_col}' or '{val_col}' is not numeric."
        )
        # Attempt conversion or return None? Return None for safety.
        return None
    # --- Input Validation --- End

    try:
        # Calculate absolute difference in temperature and find the index of the minimum difference
        closest_index = (df[temp_col] - temp).abs().idxmin()

        # Retrieve the value from the specified value column at that index
        closest_value = df.loc[closest_index, val_col]

        # Ensure the value is a float, handle potential NaNs returned by .loc
        if pd.isna(closest_value):
            logger.debug(f"Value at closest temp to {temp:.2f} is NaN.")
            return None

        return float(closest_value)
    except (KeyError, TypeError, ValueError) as e:
        # Catch potential errors during index finding or value retrieval
        logger.error(f"Error finding value at temp {temp:.2f}: {e}", exc_info=True)
        return None


def calculate_linear_baseline(
    lower_temp: float, lower_val: float, upper_temp: float, upper_val: float
) -> Tuple[float, float]:
    """Calculates the slope and intercept of a linear baseline between two points.

    Args:
        lower_temp: Temperature (x-coordinate) of the lower baseline point.
        lower_val: Value (y-coordinate, e.g., dCp) of the lower baseline point.
        upper_temp: Temperature (x-coordinate) of the upper baseline point.
        upper_val: Value (y-coordinate, e.g., dCp) of the upper baseline point.

    Returns:
        A tuple containing the slope (m) and intercept (b) of the line (y = mx + b).
        Returns (0.0, lower_val) if temperatures are identical or too close to
        avoid division by zero, effectively creating a horizontal baseline segment.
    """
    # Check if temperatures are identical or very close to prevent division by zero
    if np.isclose(upper_temp, lower_temp):
        logger.warning(
            "Upper and lower baseline temperatures are identical or too close. "
            "Returning horizontal baseline parameters (slope=0)."
        )
        # Slope is 0, intercept is the value at that temperature
        return 0.0, lower_val

    try:
        # Calculate slope: m = (y2 - y1) / (x2 - x1)
        slope = (upper_val - lower_val) / (upper_temp - lower_temp)
        # Calculate intercept: b = y1 - m * x1
        intercept = lower_val - slope * lower_temp
        return slope, intercept
    except Exception as e:  # Catch any unexpected calculation errors
        logger.error(
            f"Error calculating linear baseline parameters: {e}", exc_info=True
        )
        # Fallback to horizontal baseline at the lower point as a safety measure
        return 0.0, lower_val


# Removed redundant simple_baseline_subtraction function.
# The canonical version now resides in core/baseline/simple.py
