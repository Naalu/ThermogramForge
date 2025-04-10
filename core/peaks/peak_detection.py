"""Functions for detecting peaks in thermogram data."""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Default temperature regions for peak detection
PEAK_REGIONS = {
    "Peak_F": (50, 54),
    "Peak_1": (60, 66),
    "Peak_2": (67, 73),
    "Peak_3": (73, 81),
}


def _find_max_in_region(
    df: pd.DataFrame, temp_min: float, temp_max: float
) -> Optional[Tuple[float, float]]:
    """Finds the temperature and height of the maximum dCp within a specified range."""
    if df.empty:
        return None

    # REMOVED redundant type casting - Assume input df has correct types
    # df[\"Temperature\"] = pd.to_numeric(df[\"Temperature\"])
    # df[\"dCp_subtracted\"] = pd.to_numeric(df[\"dCp_subtracted\"])

    # Add logging to see the dtypes *as received*
    logger.debug(f"_find_max_in_region received dtypes:\n{df.dtypes}")
    # logger.info(f\"_find_max_in_region: Input dtypes:\\n{df.dtypes}\") # Original log line, can be removed or kept
    logger.info(f"_find_max_in_region: Checking region {temp_min}-{temp_max}")

    # Filter the DataFrame for the specified temperature range using Series.between for robustness
    region_df = df[df["Temperature"].between(temp_min, temp_max, inclusive="both")]

    logger.info(f"_find_max_in_region: region_df shape: {region_df.shape}")

    if region_df.empty:
        logger.info(f"No data found in region {temp_min}-{temp_max}°C.")
        return None

    # Find index of max dCp_subtracted
    max_idx = region_df["dCp_subtracted"].idxmax()

    # Check if max_idx is NaN (can happen if all dCp are NaN in region)
    if pd.isna(max_idx):
        logger.debug(f"All dCp values are NaN in region {temp_min}-{temp_max}°C.")
        return None

    peak_temp = region_df.loc[max_idx, "Temperature"]
    peak_height = region_df.loc[max_idx, "dCp_subtracted"]

    # Check if height is NaN (can happen if the max value itself is NaN)
    if pd.isna(peak_height):
        logger.debug(
            f"Maximum dCp value is NaN in region {temp_min}-{temp_max}°C at T={peak_temp:.2f}. Treating as no peak."
        )
        return None

    # ADDED CHECK: Ensure peak height is positive (Optional, but good practice)
    # If peak detection requires positive values, uncomment below
    # if peak_height <= 0:
    #     logger.debug(
    #         f"Max value in region {temp_min}-{temp_max}°C is not positive ({peak_height:.4f}). Treating as no peak."
    #     )
    #     return None

    logger.debug(
        f"Found max in region {temp_min}-{temp_max}°C: T={peak_temp:.2f}, H={peak_height:.4f}"
    )
    return peak_temp, peak_height


def detect_thermogram_peaks(
    df_processed: pd.DataFrame, regions: Dict[str, Tuple[float, float]] = PEAK_REGIONS
) -> Dict[str, Dict[str, Optional[float]]]:
    """Detects thermogram peaks based on maximum height within defined regions.

    Args:
        df_processed: DataFrame with 'Temperature' and 'dCp_subtracted' columns,
                      sorted by Temperature.
        regions: Dictionary defining peak regions, e.g.,
                 {"PeakName": (min_temp, max_temp)}. Defaults to PEAK_REGIONS.

    Returns:
        Dictionary containing peak information for each region:
        {"RegionName": {"peak_temp": T, "peak_height": H}, ...}
        Values are None if no peak is found in a region.
    """
    if not isinstance(df_processed, pd.DataFrame) or df_processed.empty:
        logger.warning("detect_thermogram_peaks: Input DataFrame is invalid or empty.")
        return {name: {"peak_temp": None, "peak_height": None} for name in regions}
    if (
        "Temperature" not in df_processed.columns
        or "dCp_subtracted" not in df_processed.columns
    ):
        logger.error(
            "detect_thermogram_peaks: DataFrame missing required columns ('Temperature', 'dCp_subtracted')."
        )
        return {name: {"peak_temp": None, "peak_height": None} for name in regions}

    # Ensure data is sorted by temperature (important for some potential future algorithms)
    if not df_processed["Temperature"].is_monotonic_increasing:
        logger.warning("Input DataFrame is not sorted by Temperature. Sorting...")
        df_processed = df_processed.sort_values("Temperature").reset_index(drop=True)

    peaks_data: Dict[str, Dict[str, Optional[float]]] = {}
    logger.info(f"Detecting peaks in regions: {list(regions.keys())}")

    for region_name, (temp_min, temp_max) in regions.items():
        result = _find_max_in_region(df_processed, temp_min, temp_max)
        if result:
            peaks_data[region_name] = {"peak_temp": result[0], "peak_height": result[1]}
        else:
            peaks_data[region_name] = {"peak_temp": None, "peak_height": None}
            logger.info(f"No valid peak found for region: {region_name}")

    logger.info("Peak detection finished.")
    return peaks_data
