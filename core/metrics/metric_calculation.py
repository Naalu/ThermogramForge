"""Functions for calculating metrics from thermogram data."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d  # For FWHM

logger = logging.getLogger(__name__)

# --- Helper Functions for Specific Metrics --- Start


def _calculate_area(df: pd.DataFrame) -> Optional[float]:
    """Calculates the total area under the dCp_subtracted curve using trapezoidal rule."""
    try:
        if df.empty or df["dCp_subtracted"].isnull().all():
            return None
        # Ensure numeric types and handle potential NaNs before integration
        temps = pd.to_numeric(df["Temperature"], errors="coerce")
        dcp = pd.to_numeric(df["dCp_subtracted"], errors="coerce")
        valid_mask = ~np.isnan(temps) & ~np.isnan(dcp)
        if valid_mask.sum() < 2:  # Need at least two points for trapz
            return None
        return np.trapz(dcp[valid_mask], temps[valid_mask])
    except KeyError as e:
        logger.error(f"Area calc: Missing column: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calculating area: {e}", exc_info=True)
        return None


def _calculate_tfm(df: pd.DataFrame) -> Optional[float]:
    """Calculates the First Moment Temperature (TFM)."""
    try:
        if (
            df.empty
            or df["dCp_subtracted"].isnull().all()
            or (df["dCp_subtracted"] <= 0).all()
        ):
            # Avoid division by zero or TFM for negative/zero curves
            return None
        temps = pd.to_numeric(df["Temperature"], errors="coerce")
        dcp = pd.to_numeric(df["dCp_subtracted"], errors="coerce")

        # Consider only positive dCp contributions for TFM?
        # Based on formula, seems all points are used, but denominator must be non-zero.
        # Filter NaNs first
        valid_mask = ~np.isnan(temps) & ~np.isnan(dcp)
        if not valid_mask.any():
            return None
        temps_valid = temps[valid_mask]
        dcp_valid = dcp[valid_mask]

        numerator = np.sum(temps_valid * dcp_valid)
        denominator = np.sum(dcp_valid)

        if denominator == 0 or pd.isna(denominator):
            return None

        return numerator / denominator
    except KeyError as e:
        logger.error(f"TFM calc: Missing column: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calculating TFM: {e}", exc_info=True)
        return None


def _calculate_fwhm(df: pd.DataFrame) -> Optional[float]:
    """Calculates the Full Width at Half Maximum (FWHM)."""
    try:
        if df.empty or df["dCp_subtracted"].isnull().all():
            return None

        temps = pd.to_numeric(df["Temperature"], errors="coerce")
        dcp = pd.to_numeric(df["dCp_subtracted"], errors="coerce")
        valid_mask = ~np.isnan(temps) & ~np.isnan(dcp)

        if valid_mask.sum() < 2:
            return None
        temps_valid = temps[valid_mask]
        dcp_valid = dcp[valid_mask]

        min_height = np.min(dcp_valid)
        max_height_idx = np.argmax(dcp_valid)
        max_height = dcp_valid.iloc[max_height_idx]
        # max_temp = temps_valid.iloc[max_height_idx]

        if max_height <= min_height:  # No peak or flat line
            return 0.0  # Or None?

        half_height = min_height + (max_height - min_height) / 2.0

        # Create interpolation function (ensure temps are strictly increasing)
        temps_unique, unique_idx = np.unique(temps_valid, return_index=True)
        if len(temps_unique) < 2:
            return None  # Need at least 2 unique points to interpolate
        interp_func = interp1d(
            temps_unique,
            dcp_valid.iloc[unique_idx],
            kind="linear",
            bounds_error=False,
            fill_value=min_height,
        )

        # Find temperatures crossing half_height
        # Approach 1: Evaluate interp_func over a fine grid and find crossings
        fine_temps = np.linspace(
            temps_valid.min(), temps_valid.max(), num=len(temps_valid) * 10
        )
        interp_dcp = interp_func(fine_temps)
        above_half = interp_dcp > half_height
        crossings_idx = np.where(np.diff(above_half))[0]

        if len(crossings_idx) < 2:
            logger.warning("FWHM: Could not find two crossings at half height.")
            return None  # Or 0.0?

        # Get temperatures at the crossings (indices are for the fine grid)
        # Average the temp just before and just after the crossing for better estimate
        t1 = (fine_temps[crossings_idx[0]] + fine_temps[crossings_idx[0] + 1]) / 2
        t2 = (fine_temps[crossings_idx[-1]] + fine_temps[crossings_idx[-1] + 1]) / 2

        return abs(t2 - t1)

    except KeyError as e:
        logger.error(f"FWHM calc: Missing column: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calculating FWHM: {e}", exc_info=True)
        return None


def _calculate_valley_v12(
    df: pd.DataFrame, t_peak1: Optional[float], t_peak2: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Calculates the minimum dCp (V1.2) and its temperature (TV1.2) between Peak 1 and Peak 2."""
    if t_peak1 is None or t_peak2 is None or pd.isna(t_peak1) or pd.isna(t_peak2):
        return None, None

    t_start = min(t_peak1, t_peak2)
    t_end = max(t_peak1, t_peak2)

    try:
        valley_df = df[(df["Temperature"] > t_start) & (df["Temperature"] < t_end)]
        if valley_df.empty or valley_df["dCp_subtracted"].isnull().all():
            return None, None

        min_idx = valley_df["dCp_subtracted"].idxmin()
        if pd.isna(min_idx):
            return None, None

        valley_temp = valley_df.loc[min_idx, "Temperature"]
        valley_height = valley_df.loc[min_idx, "dCp_subtracted"]

        if pd.isna(valley_height):
            return None, None

        return valley_height, valley_temp
    except KeyError as e:
        logger.error(f"Valley calc: Missing column: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error calculating valley V1.2: {e}", exc_info=True)
        return None, None


# --- Helper Functions for Specific Metrics --- End


def calculate_thermogram_metrics(
    df_processed: pd.DataFrame, peaks_data: Dict[str, Dict[str, Optional[float]]]
) -> Dict[str, Any]:
    """Calculates a comprehensive set of metrics from processed thermogram data and detected peaks.

    Args:
        df_processed: DataFrame with 'Temperature' and 'dCp_subtracted'.
        peaks_data: Dictionary from detect_thermogram_peaks containing peak info.

    Returns:
        Dictionary containing all calculated metrics.
    """
    metrics: Dict[str, Any] = {}
    logger.info("Calculating thermogram metrics.")

    if not isinstance(df_processed, pd.DataFrame) or df_processed.empty:
        logger.warning(
            "calculate_thermogram_metrics: Input DataFrame is invalid or empty."
        )
        return metrics  # Return empty dict
    if not isinstance(peaks_data, dict):
        logger.warning("calculate_thermogram_metrics: peaks_data is invalid.")
        # Proceed without peak-dependent metrics or return empty?
        # Let's proceed but log heavily
        peaks_data = {}  # Use empty dict to avoid errors

    required_cols = ["Temperature", "dCp_subtracted"]
    if not all(col in df_processed.columns for col in required_cols):
        logger.error(
            f"calculate_thermogram_metrics: DataFrame missing required columns: {required_cols}."
        )
        return metrics

    # --- Peak-Based Metrics --- Start
    peak_regions = ["Peak_F", "Peak_1", "Peak_2", "Peak_3"]
    peak_heights: Dict[str, Optional[float]] = {}
    peak_temps: Dict[str, Optional[float]] = {}
    for region in peak_regions:
        peak_info = peaks_data.get(region, {"peak_temp": None, "peak_height": None})
        peak_heights[region] = peak_info.get("peak_height")
        peak_temps[f"T{region}"] = peak_info.get("peak_temp")  # Store temps as TPeak_X
        metrics[region] = peak_heights[region]  # Add direct height metric Peak_X
        metrics[f"T{region}"] = peak_temps[
            f"T{region}"
        ]  # Add direct temp metric TPeak_X

    # Peak Ratios (handle None values)
    def safe_ratio(num, den):
        if num is not None and den is not None and den != 0:
            return num / den
        return None

    metrics["Peak1_Peak2_Ratio"] = safe_ratio(
        peak_heights.get("Peak_1"), peak_heights.get("Peak_2")
    )
    metrics["Peak1_Peak3_Ratio"] = safe_ratio(
        peak_heights.get("Peak_1"), peak_heights.get("Peak_3")
    )
    metrics["Peak2_Peak3_Ratio"] = safe_ratio(
        peak_heights.get("Peak_2"), peak_heights.get("Peak_3")
    )
    # --- Peak-Based Metrics --- End

    # --- Valley Metrics --- Start
    v12_height, tv12_temp = _calculate_valley_v12(
        df_processed, metrics.get("TPeak_1"), metrics.get("TPeak_2")
    )
    metrics["V1.2"] = v12_height
    metrics["TV1.2"] = tv12_temp

    # Valley Ratios
    metrics["V1.2_Peak1_Ratio"] = safe_ratio(v12_height, peak_heights.get("Peak_1"))
    metrics["V1.2_Peak2_Ratio"] = safe_ratio(v12_height, peak_heights.get("Peak_2"))
    metrics["V1.2_Peak3_Ratio"] = safe_ratio(v12_height, peak_heights.get("Peak_3"))
    # --- Valley Metrics --- End

    # --- Global Metrics --- Start
    try:
        dcp_numeric = pd.to_numeric(
            df_processed["dCp_subtracted"], errors="coerce"
        ).dropna()
        temp_numeric = pd.to_numeric(
            df_processed["Temperature"], errors="coerce"
        ).dropna()
        df_numeric = df_processed.dropna(subset=["Temperature", "dCp_subtracted"])

        if not dcp_numeric.empty:
            max_idx = dcp_numeric.idxmax()
            min_idx = dcp_numeric.idxmin()
            metrics["Max"] = dcp_numeric.loc[max_idx]
            metrics["TMax"] = df_processed.loc[max_idx, "Temperature"]
            metrics["Min"] = dcp_numeric.loc[min_idx]
            metrics["TMin"] = df_processed.loc[min_idx, "Temperature"]
            metrics["Median"] = dcp_numeric.median()
        else:
            metrics["Max"] = metrics["TMax"] = metrics["Min"] = metrics[
                "TMin"
            ] = metrics["Median"] = None

        # Use helper functions for complex global metrics
        metrics["Area"] = _calculate_area(df_numeric)
        metrics["TFM"] = _calculate_tfm(df_numeric)
        metrics["FWHM"] = _calculate_fwhm(df_numeric)

    except KeyError as e:
        logger.error(f"Global metric calc: Missing column: {e}")
    except Exception as e:
        logger.error(f"Error calculating global metrics: {e}", exc_info=True)
    # --- Global Metrics --- End

    # Filter out None values before returning? Optional.
    # final_metrics = {k: v for k, v in metrics.items() if v is not None}

    logger.info(f"Metrics calculation finished. Found {len(metrics)} metrics.")
    # logger.debug(f"Calculated metrics: {metrics}")
    return metrics
