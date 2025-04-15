"""Provides functions for detecting peaks in baseline-subtracted thermogram data."""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)

# Default temperature regions (min_temp, max_temp) for identifying specific peaks.
# Used as the default value for the `regions` argument in `detect_thermogram_peaks`.
PEAK_REGIONS: Dict[str, Tuple[float, float]] = {
    "Peak_F": (50, 54),
    "Peak_1": (60, 66),
    "Peak_2": (67, 73),
    "Peak_3": (73, 81),
}


def _find_max_in_region(
    df: pd.DataFrame, temp_min: float, temp_max: float
) -> Optional[Tuple[float, float]]:
    """Finds the temperature and height of the maximum dCp within a range.

    Filters the DataFrame to the specified temperature range [temp_min, temp_max]
    and returns the temperature and dCp value corresponding to the maximum
    dCp value found within that range.

    Args:
        df: DataFrame containing numeric 'Temperature' and 'dCp_subtracted' columns.
        temp_min: The minimum temperature (inclusive) of the search region.
        temp_max: The maximum temperature (inclusive) of the search region.

    Returns:
        A tuple (peak_temperature, peak_height) if a valid maximum is found,
        otherwise None.
    """
    # --- Input Validation --- Start
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.debug("_find_max_in_region: Input DataFrame is empty or invalid.")
        return None
    if "Temperature" not in df.columns or "dCp_subtracted" not in df.columns:
        logger.error("_find_max_in_region: DataFrame missing required columns.")
        # Or raise ValueError?
        return None
    if not is_numeric_dtype(df["Temperature"]) or not is_numeric_dtype(
        df["dCp_subtracted"]
    ):
        logger.error(
            "_find_max_in_region: Temperature or dCp_subtracted column is not numeric."
        )
        # Or raise TypeError?
        return None
    if temp_min > temp_max:
        logger.error(
            f"_find_max_in_region: temp_min ({temp_min}) > temp_max ({temp_max})."
        )
        return None
    # --- Input Validation --- End

    # Filter the DataFrame for the specified temperature range
    # Using .between() is generally robust.
    region_df = df[df["Temperature"].between(temp_min, temp_max, inclusive="both")]

    if region_df.empty:
        logger.debug(f"No data found in region {temp_min}-{temp_max}°C.")
        return None

    try:
        # Find index of max dCp_subtracted within the filtered region
        # Ensure we handle cases where the max value itself might be NaN
        dcp_in_region = region_df["dCp_subtracted"]
        if dcp_in_region.isnull().all():
            logger.debug(f"All dCp values are NaN in region {temp_min}-{temp_max}°C.")
            return None

        max_idx = dcp_in_region.idxmax()

        # idxmax can return NaN if all values are NaN (handled above), but check just in case
        if pd.isna(max_idx):
            logger.debug(
                f"Could not determine max index in region {temp_min}-{temp_max}°C."
            )
            return None

        peak_temp: float = region_df.loc[max_idx, "Temperature"]
        peak_height: float = region_df.loc[max_idx, "dCp_subtracted"]

        # Check if the identified peak values themselves are NaN
        if pd.isna(peak_temp) or pd.isna(peak_height):
            logger.debug(
                f"NaN value found for peak temp or height in region {temp_min}-{temp_max}°C."
            )
            return None

        # Optional: Check if peak height must be positive
        # if peak_height <= 0:
        #     logger.debug(
        #         f"Max value in region {temp_min}-{temp_max}°C is not positive ({peak_height:.4f}). Treating as no peak."
        #     )
        #     return None

        logger.debug(
            f"Found max in region {temp_min}-{temp_max}°C: T={peak_temp:.2f}, H={peak_height:.4f}"
        )
        return peak_temp, peak_height

    except Exception as e:
        logger.error(
            f"Error finding max in region {temp_min}-{temp_max}°C: {e}", exc_info=True
        )
        return None


def detect_thermogram_peaks(
    df_processed: pd.DataFrame, regions: Dict[str, Tuple[float, float]] = PEAK_REGIONS
) -> Dict[str, Dict[str, Optional[float]]]:
    """Detects thermogram peaks by finding the maximum within defined regions.

    Iterates through predefined or user-specified temperature regions and finds the
    point with the maximum baseline-subtracted dCp value within each region.

    Args:
        df_processed: DataFrame containing baseline-subtracted data with numeric
                      'Temperature' and 'dCp_subtracted' columns.
        regions: A dictionary defining the peak search regions. Keys are peak names
                 (str), and values are tuples of (min_temp, max_temp).
                 Defaults to `PEAK_REGIONS` constant defined in this module.

    Returns:
        A dictionary where keys are the region names (from the `regions` input)
        and values are dictionaries containing the detected peak's temperature
        ('peak_temp') and height ('peak_height'). Values within the inner
        dictionary will be None if no valid peak was found in that region.
        Example: {"Peak_1": {"peak_temp": 65.1, "peak_height": 1.23}, ...}
    """
    # --- Input Validation --- Start
    if not isinstance(df_processed, pd.DataFrame) or df_processed.empty:
        logger.warning("detect_thermogram_peaks: Input DataFrame is invalid or empty.")
        # Return dict with None for all expected regions based on input `regions`
        return {name: {"peak_temp": None, "peak_height": None} for name in regions}

    required_cols = ["Temperature", "dCp_subtracted"]
    if not all(col in df_processed.columns for col in required_cols):
        logger.error(
            f"detect_thermogram_peaks: DataFrame missing required columns: {required_cols}."
        )
        return {name: {"peak_temp": None, "peak_height": None} for name in regions}

    if not is_numeric_dtype(df_processed["Temperature"]) or not is_numeric_dtype(
        df_processed["dCp_subtracted"]
    ):
        logger.error(
            "detect_thermogram_peaks: Temperature or dCp_subtracted column is not numeric."
        )
        return {name: {"peak_temp": None, "peak_height": None} for name in regions}

    if not isinstance(regions, dict) or not regions:
        logger.error(
            "detect_thermogram_peaks: `regions` argument must be a non-empty dictionary."
        )
        # Cannot determine output structure, return empty dict
        return {}
    # Further validation could check tuple structure and numeric types within regions
    # --- Input Validation --- End

    # Ensure data is sorted by temperature - important for consistency and potential future algorithms
    if not df_processed["Temperature"].is_monotonic_increasing:
        logger.debug("Input DataFrame is not sorted by Temperature. Sorting...")
        # Use .copy() to avoid SettingWithCopyWarning if df_processed is a slice
        df_processed = (
            df_processed.sort_values("Temperature").reset_index(drop=True).copy()
        )

    peaks_data: Dict[str, Dict[str, Optional[float]]] = {}
    logger.info(f"Detecting peaks in regions: {list(regions.keys())}")

    # Iterate through each defined region and find the maximum peak
    for region_name, (temp_min, temp_max) in regions.items():
        # Use the helper function to find the max point in the current region
        result = _find_max_in_region(df_processed, temp_min, temp_max)

        # Store the result (temperature and height) or None if no peak found
        if result:
            peak_temp, peak_height = result
            peaks_data[region_name] = {
                "peak_temp": peak_temp,
                "peak_height": peak_height,
            }
        else:
            peaks_data[region_name] = {"peak_temp": None, "peak_height": None}
            logger.debug(f"No valid peak found for region: {region_name}")

    logger.info("Peak detection finished.")
    return peaks_data
