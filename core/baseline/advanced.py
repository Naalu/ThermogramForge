"""
Advanced baseline processing using spline fitting.
"""

import logging
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)


class EndpointSelectionMethod(Enum):
    """Methods for selecting endpoints from multiple candidates."""

    INNERMOST = "Innermost"
    OUTERMOST = "Outermost"
    MIDDLE = "Middle"


def _fit_smoothing_spline(
    x: np.ndarray,
    y: np.ndarray,
    s: Optional[float] = None,  # Smoothing factor (higher = smoother)
    k: int = 3,  # Spline degree (cubic default)
    w: Optional[np.ndarray] = None,  # Weights (optional)
) -> Optional[UnivariateSpline]:
    """Fits a smoothing spline using scipy.

    Args:
        x: 1D array of temperature values.
        y: 1D array of dCp values.
        s: Smoothing factor for UnivariateSpline. If None, chooses automatically.
        k: Degree of the spline.
        w: Weights for data points.

    Returns:
        A fitted UnivariateSpline object, or None if fitting fails.
    """
    try:
        # Ensure inputs are sorted by x for spline fitting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        if w is not None:
            w_sorted = w[sort_idx]
        else:
            w_sorted = None

        # Ensure no duplicate x values (required by UnivariateSpline)
        unique_x, unique_idx = np.unique(x_sorted, return_index=True)
        if len(unique_x) < len(x_sorted):
            logger.debug(
                "Duplicate x values found, averaging y values for spline fitting."
            )
            unique_y = np.array([np.mean(y_sorted[x_sorted == ux]) for ux in unique_x])
            if w_sorted is not None:
                unique_w = np.array(
                    [np.mean(w_sorted[x_sorted == ux]) for ux in unique_x]
                )
            else:
                unique_w = None
            x_in = unique_x
            y_in = unique_y
            w_in = unique_w
        else:
            x_in = x_sorted
            y_in = y_sorted
            w_in = w_sorted

        if len(x_in) <= k:
            logger.warning(
                f"Not enough unique points ({len(x_in)}) to fit spline of degree {k}. Need > {k}."
            )
            return None

        spline = UnivariateSpline(x_in, y_in, k=k, s=s, w=w_in)
        logger.debug(f"Successfully fitted smoothing spline (k={k}, s={s or 'auto'}).")
        return spline
    except Exception as e:
        logger.error(f"Error fitting smoothing spline: {e}", exc_info=True)
        return None


def find_spline_endpoints(
    df: pd.DataFrame,
    lower_exclusion_temp: float,
    upper_exclusion_temp: float,
    window_size: int = 10,
    spline_smooth_factor: Optional[float] = None,
    point_selection_method: EndpointSelectionMethod = EndpointSelectionMethod.INNERMOST,
) -> Dict[str, Optional[float]]:
    """Detects baseline endpoints using rolling variance on a fitted spline.

    Args:
        df: DataFrame with 'Temperature' and 'dCp' columns.
        lower_exclusion_temp: Lower bound of the transition region to exclude.
        upper_exclusion_temp: Upper bound of the transition region to exclude.
        window_size: Number of points for the rolling variance window.
        spline_smooth_factor: Smoothing factor `s` for the initial spline fit.
                              If None, `UnivariateSpline` chooses automatically.
        point_selection_method: Method ('Innermost', 'Outermost', 'Middle')
                                to select endpoints if multiple candidates exist.

    Returns:
        A dictionary with 'lower' and 'upper' endpoint temperatures, or None
        if detection fails.
    """
    logger.info(
        f"Finding spline endpoints. Exclusion: {lower_exclusion_temp}-{upper_exclusion_temp}, "
        f"Window: {window_size}, Method: {point_selection_method.value}"
    )
    endpoints = {"lower": None, "upper": None}

    if df.empty or "Temperature" not in df or "dCp" not in df:
        logger.warning("Input DataFrame invalid or empty.")
        return endpoints

    temp = df["Temperature"].values
    dcp = df["dCp"].values

    # 1. Fit initial smoothing spline to the whole dataset
    initial_spline = _fit_smoothing_spline(temp, dcp, s=spline_smooth_factor)
    if initial_spline is None:
        logger.warning("Failed to fit initial spline for endpoint detection.")
        return endpoints

    # Predict dCp values using the spline across the original temperature range
    dcp_spline_pred = initial_spline(temp)

    # 2. Calculate rolling variance on the *spline values*
    try:
        rolling_variance = (
            pd.Series(dcp_spline_pred).rolling(window=window_size, center=True).var()
        )
        # Need to handle NaNs at edges introduced by rolling window
        rolling_variance = rolling_variance.fillna(method="bfill").fillna(
            method="ffill"
        )
    except Exception as e:
        logger.error(f"Error calculating rolling variance: {e}", exc_info=True)
        return endpoints

    # 3. Find minimum variance points outside exclusion zone
    df_analysis = pd.DataFrame(
        {"Temperature": temp, "RollingVariance": rolling_variance}
    )

    # Lower baseline region
    lower_region = df_analysis[df_analysis["Temperature"] < lower_exclusion_temp]
    # Upper baseline region
    upper_region = df_analysis[df_analysis["Temperature"] > upper_exclusion_temp]

    min_var_candidates_lower = []
    min_var_candidates_upper = []

    if not lower_region.empty:
        min_var_lower_idx = lower_region["RollingVariance"].idxmin()
        min_var_candidates_lower = lower_region[
            lower_region["RollingVariance"] == lower_region["RollingVariance"].min()
        ]["Temperature"].tolist()
        # endpoints["lower"] = lower_region.loc[min_var_lower_idx, 'Temperature']
        logger.debug(
            f"Min variance in lower region ({lower_region['RollingVariance'].min():.4g}) found at temps: {min_var_candidates_lower}"
        )
    else:
        logger.warning("No data points found in the lower baseline region.")

    if not upper_region.empty:
        min_var_upper_idx = upper_region["RollingVariance"].idxmin()
        min_var_candidates_upper = upper_region[
            upper_region["RollingVariance"] == upper_region["RollingVariance"].min()
        ]["Temperature"].tolist()
        # endpoints["upper"] = upper_region.loc[min_var_upper_idx, 'Temperature']
        logger.debug(
            f"Min variance in upper region ({upper_region['RollingVariance'].min():.4g}) found at temps: {min_var_candidates_upper}"
        )
    else:
        logger.warning("No data points found in the upper baseline region.")

    # 4. Select endpoint based on method
    if min_var_candidates_lower:
        if point_selection_method == EndpointSelectionMethod.INNERMOST:
            endpoints["lower"] = max(min_var_candidates_lower)
        elif point_selection_method == EndpointSelectionMethod.OUTERMOST:
            endpoints["lower"] = min(min_var_candidates_lower)
        elif point_selection_method == EndpointSelectionMethod.MIDDLE:
            endpoints["lower"] = sorted(min_var_candidates_lower)[
                len(min_var_candidates_lower) // 2
            ]
        else:
            endpoints["lower"] = max(min_var_candidates_lower)  # Default to innermost

    if min_var_candidates_upper:
        if point_selection_method == EndpointSelectionMethod.INNERMOST:
            endpoints["upper"] = min(min_var_candidates_upper)
        elif point_selection_method == EndpointSelectionMethod.OUTERMOST:
            endpoints["upper"] = max(min_var_candidates_upper)
        elif point_selection_method == EndpointSelectionMethod.MIDDLE:
            endpoints["upper"] = sorted(min_var_candidates_upper)[
                len(min_var_candidates_upper) // 2
            ]
        else:
            endpoints["upper"] = min(min_var_candidates_upper)  # Default to innermost

    logger.info(
        f"Selected endpoints: Lower={endpoints['lower']}, Upper={endpoints['upper']}"
    )
    return endpoints


def subtract_spline_baseline(
    df: pd.DataFrame,
    lower_endpoint: float,
    upper_endpoint: float,
    spline_smooth_factor: Optional[
        float
    ] = None,  # Smoothing factor for baseline splines
    k: int = 3,  # Degree for baseline splines
) -> Optional[pd.DataFrame]:
    """Fits splines to baseline regions and subtracts the constructed baseline.

    Args:
        df: DataFrame with 'Temperature' and 'dCp' columns.
        lower_endpoint: Temperature of the lower baseline endpoint.
        upper_endpoint: Temperature of the upper baseline endpoint.
        spline_smooth_factor: Smoothing factor `s` for fitting baseline splines.
        k: Degree for baseline splines.

    Returns:
        DataFrame with 'Temperature' and 'dCp_subtracted' columns, or None if failed.
    """
    logger.info(
        f"Subtracting spline baseline. Endpoints: {lower_endpoint:.2f}-{upper_endpoint:.2f}"
    )
    if df.empty or "Temperature" not in df or "dCp" not in df:
        logger.warning("Input DataFrame invalid or empty for baseline subtraction.")
        return None
    if (
        lower_endpoint is None
        or upper_endpoint is None
        or lower_endpoint >= upper_endpoint
    ):
        logger.warning(
            "Invalid or unordered endpoints provided for baseline subtraction."
        )
        return None

    temp = df["Temperature"].values
    dcp = df["dCp"].values

    # 1. Define baseline regions
    lower_mask = temp < lower_endpoint
    upper_mask = temp > upper_endpoint
    middle_mask = (temp >= lower_endpoint) & (temp <= upper_endpoint)

    # Check if baseline regions have enough points
    if np.sum(lower_mask) <= k:
        logger.warning(
            f"Not enough points ({np.sum(lower_mask)}) in lower baseline region to fit spline."
        )
        return None
    if np.sum(upper_mask) <= k:
        logger.warning(
            f"Not enough points ({np.sum(upper_mask)}) in upper baseline region to fit spline."
        )
        return None

    # 2. Fit splines to baseline regions
    spline_lower = _fit_smoothing_spline(
        temp[lower_mask], dcp[lower_mask], s=spline_smooth_factor, k=k
    )
    spline_upper = _fit_smoothing_spline(
        temp[upper_mask], dcp[upper_mask], s=spline_smooth_factor, k=k
    )

    if spline_lower is None or spline_upper is None:
        logger.error("Failed to fit one or both baseline splines.")
        return None

    # 3. Construct the full baseline
    baseline_dcp = np.zeros_like(temp)

    # Evaluate splines in their respective regions
    baseline_dcp[lower_mask] = spline_lower(temp[lower_mask])
    baseline_dcp[upper_mask] = spline_upper(temp[upper_mask])

    # Linear interpolation for the middle region (transition)
    # Get the spline values AT the endpoints
    try:
        dcp_at_lower_endpoint = spline_lower(lower_endpoint)
        dcp_at_upper_endpoint = spline_upper(upper_endpoint)
    except ValueError as ve:
        logger.error(
            f"Could not evaluate baseline spline at endpoint: {ve}. Check endpoint values vs data range."
        )
        # Attempt recovery using nearest point?
        # For now, fail.
        return None

    # Apply linear interpolation only where middle_mask is True
    if np.any(middle_mask):
        temp_middle = temp[middle_mask]
        baseline_dcp[middle_mask] = np.interp(
            temp_middle,
            [lower_endpoint, upper_endpoint],
            [dcp_at_lower_endpoint, dcp_at_upper_endpoint],
        )

    # 4. Subtract baseline
    dcp_subtracted = dcp - baseline_dcp

    # 5. Return result DataFrame
    result_df = pd.DataFrame(
        {
            "Temperature": temp,
            "dCp_subtracted": dcp_subtracted,
            "Baseline": baseline_dcp,  # Optionally include the calculated baseline
        }
    )
    logger.info("Successfully subtracted spline baseline.")
    return result_df
