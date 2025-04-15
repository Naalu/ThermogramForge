"""Functions for calculating standard metrics from thermogram data."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.integrate import trapezoid  # Use scipy.integrate.trapezoid
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# --- Helper Functions for Specific Metrics --- Start


def _calculate_area(df: pd.DataFrame) -> Optional[float]:
    """Calculates the area under the baseline-subtracted curve.

    Uses the trapezoidal rule for numerical integration.

    Args:
        df: DataFrame containing numeric 'Temperature' and 'dCp_subtracted' columns.

    Returns:
        The calculated area (float), or None if calculation is not possible
        (e.g., insufficient valid data points).
    """
    if (
        not isinstance(df, pd.DataFrame)
        or "Temperature" not in df.columns
        or "dCp_subtracted" not in df.columns
    ):
        logger.warning("_calculate_area: Invalid DataFrame input.")
        return None

    try:
        temps = pd.to_numeric(df["Temperature"], errors="coerce")
        dcp = pd.to_numeric(df["dCp_subtracted"], errors="coerce")

        # Remove rows with NaN in either temperature or dCp after coercion
        valid_mask = ~pd.isna(temps) & ~pd.isna(dcp)
        temps_valid = temps[valid_mask].to_numpy()
        dcp_valid = dcp[valid_mask].to_numpy()

        if len(temps_valid) < 2:
            logger.debug(
                "_calculate_area: Need at least two valid points for integration."
            )
            return None

        # Ensure temperatures are sorted for trapz
        sort_idx = np.argsort(temps_valid)
        temps_sorted = temps_valid[sort_idx]
        dcp_sorted = dcp_valid[sort_idx]

        # Calculate area using the trapezoidal rule
        area = trapezoid(dcp_sorted, temps_sorted)
        return float(area)  # Ensure return type is float

    except Exception as e:
        logger.error(f"Error calculating area: {e}", exc_info=True)
        return None


def _calculate_tfm(df: pd.DataFrame) -> Optional[float]:
    """Calculates the First Moment Temperature (TFM), a weighted average temperature.

    TFM = sum(T_i * dCp_i) / sum(dCp_i)

    Args:
        df: DataFrame containing numeric 'Temperature' and 'dCp_subtracted' columns.

    Returns:
        The calculated TFM (float), or None if calculation is not possible
        (e.g., sum of dCp is zero or invalid data).
    """
    if (
        not isinstance(df, pd.DataFrame)
        or "Temperature" not in df.columns
        or "dCp_subtracted" not in df.columns
    ):
        logger.warning("_calculate_tfm: Invalid DataFrame input.")
        return None

    try:
        temps = pd.to_numeric(df["Temperature"], errors="coerce")
        dcp = pd.to_numeric(df["dCp_subtracted"], errors="coerce")

        # Remove rows with NaN in either temperature or dCp
        valid_mask = ~pd.isna(temps) & ~pd.isna(dcp)
        temps_valid = temps[valid_mask].to_numpy()
        dcp_valid = dcp[valid_mask].to_numpy()

        if len(temps_valid) == 0:
            logger.debug("_calculate_tfm: No valid data points found.")
            return None

        # Calculate numerator and denominator for TFM formula
        numerator = np.sum(temps_valid * dcp_valid)
        denominator = np.sum(dcp_valid)

        # Check for zero or NaN denominator to avoid division errors
        if denominator == 0 or pd.isna(denominator):
            logger.debug(
                f"_calculate_tfm: Denominator (sum of dCp) is {denominator}, cannot calculate TFM."
            )
            return None

        tfm = numerator / denominator
        return float(tfm)

    except Exception as e:
        logger.error(f"Error calculating TFM: {e}", exc_info=True)
        return None


def _calculate_fwhm(df: pd.DataFrame) -> Optional[float]:
    """Calculates the Full Width at Half Maximum (FWHM) of the main peak.

    Finds the maximum peak height, calculates the half-maximum height, and uses
    linear interpolation to find the temperatures at which the curve crosses
    this half-height.

    Args:
        df: DataFrame containing numeric 'Temperature' and 'dCp_subtracted' columns.

    Returns:
        The calculated FWHM (float), or None if calculation is not possible (e.g.,
        no clear peak, insufficient points for interpolation, or cannot find crossings).
    """
    if (
        not isinstance(df, pd.DataFrame)
        or "Temperature" not in df.columns
        or "dCp_subtracted" not in df.columns
    ):
        logger.warning("_calculate_fwhm: Invalid DataFrame input.")
        return None

    try:
        temps = pd.to_numeric(df["Temperature"], errors="coerce")
        dcp = pd.to_numeric(df["dCp_subtracted"], errors="coerce")

        # Remove rows with NaN in either temperature or dCp
        valid_mask = ~pd.isna(temps) & ~pd.isna(dcp)
        temps_valid = temps[valid_mask]
        dcp_valid = dcp[valid_mask]

        if (
            len(temps_valid) < 3
        ):  # Need at least 3 points for reasonable peak finding/interp
            logger.debug("_calculate_fwhm: Need at least 3 valid points.")
            return None

        # Find the minimum and maximum dCp values and the temperature of the max peak
        min_dcp = dcp_valid.min()
        max_idx = dcp_valid.idxmax()  # Use idxmax on Series to get index
        max_dcp = dcp_valid.loc[max_idx]
        # max_temp = temps_valid.loc[max_idx] # Temperature at max peak

        # Check if there is a peak (max > min)
        if max_dcp <= min_dcp:
            logger.debug("_calculate_fwhm: No peak found (max_dcp <= min_dcp).")
            return 0.0  # FWHM is arguably 0 if there's no peak

        # Calculate the half-maximum height
        half_height = min_dcp + (max_dcp - min_dcp) / 2.0

        # --- Interpolation Setup ---
        # Ensure temperatures are sorted and unique for interpolation
        sorted_indices = temps_valid.argsort()
        temps_sorted = temps_valid.iloc[sorted_indices].to_numpy()
        dcp_sorted = dcp_valid.iloc[sorted_indices].to_numpy()

        unique_temps, unique_indices = np.unique(temps_sorted, return_index=True)
        if len(unique_temps) < 2:
            logger.debug(
                "_calculate_fwhm: Need at least 2 unique temperature points for interpolation."
            )
            return None
        unique_dcp = dcp_sorted[unique_indices]

        # Create the interpolation function
        interp_func = interp1d(
            unique_temps,
            unique_dcp,
            kind="linear",  # Use linear interpolation
            bounds_error=False,  # Do not raise error for out-of-bounds
            fill_value=min_dcp,  # Use min_dcp for out-of-bounds points
        )

        # --- Find Crossings ---
        # Find indices where the dCp curve crosses the half_height value
        # Check where the sign of (dCp - half_height) changes
        sign_diff = np.sign(dcp_sorted - half_height)
        crossings_idx = np.where(np.diff(sign_diff))[0]

        if len(crossings_idx) < 2:
            logger.warning(
                f"_calculate_fwhm: Could not find two crossings at half height ({half_height:.4g}). "
                f"Found {len(crossings_idx)} crossings."
            )
            # Attempt to interpolate even if only one crossing? No, FWHM needs two.
            return None

        # Interpolate to find the precise temperatures at the first and last crossings
        t_crossings = []
        for idx in [
            crossings_idx[0],
            crossings_idx[-1],
        ]:  # First and last crossing index
            # Temperature and dCp values around the crossing
            t1, t2 = temps_sorted[idx], temps_sorted[idx + 1]
            d1, d2 = dcp_sorted[idx], dcp_sorted[idx + 1]

            # Linear interpolation formula: t = t1 + (t2 - t1) * (half_h - d1) / (d2 - d1)
            if d2 == d1:  # Avoid division by zero if dCp is flat at crossing
                t_cross = (t1 + t2) / 2.0
            else:
                t_cross = t1 + (t2 - t1) * (half_height - d1) / (d2 - d1)
            t_crossings.append(t_cross)

        if len(t_crossings) < 2:
            logger.warning(
                "_calculate_fwhm: Failed to interpolate crossing temperatures."
            )
            return None

        fwhm = abs(t_crossings[1] - t_crossings[0])
        return float(fwhm)

    except Exception as e:
        logger.error(f"Error calculating FWHM: {e}", exc_info=True)
        return None


def _calculate_valley_v12(
    df: pd.DataFrame, t_peak1: Optional[float], t_peak2: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Finds the minimum dCp (V1.2) and its temperature (TV1.2) between two peaks.

    Searches the DataFrame for the minimum `dCp_subtracted` value strictly between
    the temperatures of Peak 1 and Peak 2.

    Args:
        df: DataFrame containing numeric 'Temperature' and 'dCp_subtracted' columns.
        t_peak1: Temperature of the first peak.
        t_peak2: Temperature of the second peak.

    Returns:
        A tuple containing: (valley_height, valley_temperature). Both values are
        float if the valley is found, otherwise None.
    """
    if t_peak1 is None or t_peak2 is None or pd.isna(t_peak1) or pd.isna(t_peak2):
        logger.debug("_calculate_valley_v12: One or both peak temperatures are None.")
        return None, None
    if not isinstance(df, pd.DataFrame):
        logger.warning("_calculate_valley_v12: Invalid DataFrame input.")
        return None, None
    if "Temperature" not in df.columns or "dCp_subtracted" not in df.columns:
        logger.error("_calculate_valley_v12: Missing required columns.")
        return None, None  # Or raise?

    # Ensure correct temperature range for searching
    t_start = min(t_peak1, t_peak2)
    t_end = max(t_peak1, t_peak2)

    try:
        # Filter DataFrame to the region strictly between the peaks
        valley_df = df[
            (df["Temperature"] > t_start) & (df["Temperature"] < t_end)
        ].copy()

        # Coerce dCp to numeric and find minimum
        valley_df["dCp_numeric"] = pd.to_numeric(
            valley_df["dCp_subtracted"], errors="coerce"
        )

        if valley_df.empty or valley_df["dCp_numeric"].isnull().all():
            logger.debug("_calculate_valley_v12: No valid data points between peaks.")
            return None, None

        # Find the index of the minimum dCp value in the filtered region
        min_idx = valley_df["dCp_numeric"].idxmin()

        # Check if idxmin returned NaN (can happen if all are NaN)
        if pd.isna(min_idx):
            logger.debug(
                "_calculate_valley_v12: Could not find minimum index (all NaN?)."
            )
            return None, None

        # Retrieve the valley temperature and height using the original index
        valley_temp = df.loc[min_idx, "Temperature"]
        valley_height = df.loc[
            min_idx, "dCp_subtracted"
        ]  # Use original for consistency

        # Final check if retrieved values are valid
        if pd.isna(valley_temp) or pd.isna(valley_height):
            logger.debug(
                "_calculate_valley_v12: Retrieved valley temp or height is NaN."
            )
            return None, None

        return float(valley_height), float(valley_temp)

    except Exception as e:
        logger.error(f"Error calculating valley V1.2: {e}", exc_info=True)
        return None, None


def _safe_ratio(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    """Safely calculates the ratio of two numbers, handling None and zero denominator."""
    if (
        numerator is None
        or denominator is None
        or pd.isna(numerator)
        or pd.isna(denominator)
    ):
        return None
    if denominator == 0:
        return None  # Or np.inf? Returning None is safer for general metrics.
    try:
        return float(numerator / denominator)
    except TypeError:
        return None  # Should not happen if inputs are validated, but as safety


# --- Helper Functions for Specific Metrics --- End


def calculate_thermogram_metrics(
    df_processed: pd.DataFrame, peaks_data: Dict[str, Dict[str, Optional[float]]]
) -> Dict[str, Any]:
    """Calculates a standard set of metrics from processed thermogram data.

    Utilizes detected peak information to calculate peak heights, temperatures,
    ratios, valley metrics, and global curve properties like Area, TFM, and FWHM.

    Args:
        df_processed: DataFrame containing baseline-subtracted data with at least
                      numeric 'Temperature' and 'dCp_subtracted' columns.
        peaks_data: A dictionary where keys are peak region names (e.g., 'Peak_1')
                    and values are dictionaries containing 'peak_temp' and
                    'peak_height' for that peak.
                    Example: {'Peak_1': {'peak_temp': 55.2, 'peak_height': 0.8}}.

    Returns:
        A dictionary where keys are metric names (str) and values are the
        calculated metric values (float or None if calculation failed).
    """
    metrics: Dict[str, Any] = {}
    logger.info("Calculating thermogram metrics.")

    # --- Input Validation --- Start
    if not isinstance(df_processed, pd.DataFrame) or df_processed.empty:
        logger.warning(
            "calculate_thermogram_metrics: Input DataFrame is invalid or empty."
        )
        return metrics  # Return empty dict

    required_cols = ["Temperature", "dCp_subtracted"]
    if not all(col in df_processed.columns for col in required_cols):
        logger.error(
            f"calculate_thermogram_metrics: DataFrame missing required columns: {required_cols}."
        )
        return metrics
    if not is_numeric_dtype(df_processed["Temperature"]) or not is_numeric_dtype(
        df_processed["dCp_subtracted"]
    ):
        logger.error(
            "calculate_thermogram_metrics: 'Temperature' or 'dCp_subtracted' not numeric."
        )
        return metrics

    if not isinstance(peaks_data, dict):
        logger.warning(
            "calculate_thermogram_metrics: peaks_data is not a dict. Proceeding without peak data."
        )
        peaks_data = {}  # Treat as empty if invalid format
    # --- Input Validation --- End

    # --- Peak-Based Metrics --- Start
    peak_regions = ["Peak_F", "Peak_1", "Peak_2", "Peak_3"]
    peak_heights: Dict[str, Optional[float]] = {}
    peak_temps: Dict[str, Optional[float]] = {}

    for region in peak_regions:
        # Get peak info safely, defaulting to None if region or keys are missing
        peak_info = peaks_data.get(region, {})  # Default to empty dict
        height = peak_info.get("peak_height")
        temp = peak_info.get("peak_temp")

        # Store raw height and temp for ratio calculations
        peak_heights[region] = height
        peak_temps[f"T{region}"] = temp

        # Add direct metrics to the output dictionary
        metrics[region] = height  # Metric name is Peak_X
        metrics[f"T{region}"] = temp  # Metric name is TPeak_X

    # Peak Ratios (using the safe helper function)
    metrics["Peak1_Peak2_Ratio"] = _safe_ratio(
        peak_heights.get("Peak_1"), peak_heights.get("Peak_2")
    )
    metrics["Peak1_Peak3_Ratio"] = _safe_ratio(
        peak_heights.get("Peak_1"), peak_heights.get("Peak_3")
    )
    metrics["Peak2_Peak3_Ratio"] = _safe_ratio(
        peak_heights.get("Peak_2"), peak_heights.get("Peak_3")
    )
    # --- Peak-Based Metrics --- End

    # --- Valley Metrics --- Start
    v12_height, tv12_temp = _calculate_valley_v12(
        df_processed, metrics.get("TPeak_1"), metrics.get("TPeak_2")
    )
    metrics["V1.2"] = v12_height  # Height of the valley between Peak 1 and Peak 2
    metrics["TV1.2"] = tv12_temp  # Temperature of the valley between Peak 1 and Peak 2

    # Valley Ratios
    metrics["V1.2_Peak1_Ratio"] = _safe_ratio(v12_height, peak_heights.get("Peak_1"))
    metrics["V1.2_Peak2_Ratio"] = _safe_ratio(v12_height, peak_heights.get("Peak_2"))
    metrics["V1.2_Peak3_Ratio"] = _safe_ratio(
        v12_height, peak_heights.get("Peak_3")
    )  # Added for completeness
    # --- Valley Metrics --- End

    # --- Global Curve Metrics --- Start
    metrics["TotalArea"] = _calculate_area(df_processed)
    metrics["TFM"] = _calculate_tfm(df_processed)
    metrics["FWHM"] = _calculate_fwhm(df_processed)
    # --- Global Curve Metrics --- End

    # Final check: Replace any potential np.nan floats with None for consistency
    for key, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            metrics[key] = None

    logger.info(
        f"Calculated metrics: { {k: v for k, v in metrics.items() if v is not None} }"
    )
    return metrics
