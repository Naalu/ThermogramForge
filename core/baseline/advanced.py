"""Provides advanced baseline subtraction using spline fitting techniques.

This module contains functions to automatically detect baseline endpoints
using rolling variance on spline-smoothed data and to subtract a baseline
constructed from splines fitted to the identified baseline regions.
"""

import logging
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)


class EndpointSelectionMethod(Enum):
    """Enumeration for methods to select endpoints when multiple candidates exist."""

    INNERMOST = "Innermost"
    OUTERMOST = "Outermost"
    MIDDLE = "Middle"


def _fit_smoothing_spline(
    x: np.ndarray,
    y: np.ndarray,
    s: Optional[float] = None,
    k: int = 3,
    w: Optional[np.ndarray] = None,
) -> Optional[UnivariateSpline]:
    """Fits a univariate smoothing spline to the provided data.

    Handles sorting, duplicate x-value averaging, and checks for sufficient points.

    Args:
        x: 1D array of independent variable values (e.g., Temperature).
        y: 1D array of dependent variable values (e.g., dCp).
        s: Smoothing factor for UnivariateSpline. Controls the tradeoff between
           smoothness and closeness to the data. If None, SciPy attempts to
           determine a value automatically.
        k: Degree of the smoothing spline. Must be 1 <= k <= 5. Default is 3 (cubic).
        w: Optional 1D array of weights for data points.

    Returns:
        A fitted scipy.interpolate.UnivariateSpline object, or None if fitting fails
        (e.g., due to insufficient unique points).

    Raises:
        ValueError: If input arrays x and y have different lengths.
    """
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length.")
    if w is not None and len(x) != len(w):
        raise ValueError("Weight array w must have the same length as x and y.")

    try:
        # Ensure inputs are sorted by x, as required by UnivariateSpline
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        w_sorted = w[sort_idx] if w is not None else None

        # UnivariateSpline requires strictly increasing x values.
        # Average y values (and weights) for duplicate x.
        unique_x, unique_idx, inverse_idx, counts = np.unique(
            x_sorted, return_index=True, return_inverse=True, return_counts=True
        )
        if len(unique_x) < len(x_sorted):
            logger.debug(
                f"Duplicate x values found ({len(x_sorted) - len(unique_x)} instances). "
                f"Averaging y values for spline fitting."
            )
            unique_y = np.zeros_like(unique_x, dtype=float)
            unique_w = (
                np.zeros_like(unique_x, dtype=float) if w_sorted is not None else None
            )
            np.add.at(unique_y, inverse_idx, y_sorted)
            unique_y /= counts
            if unique_w is not None:
                np.add.at(unique_w, inverse_idx, w_sorted)
                unique_w /= counts

            x_in, y_in, w_in = unique_x, unique_y, unique_w
        else:
            x_in, y_in, w_in = x_sorted, y_sorted, w_sorted

        # Check if enough unique points exist for the chosen spline degree
        if len(x_in) <= k:
            logger.warning(
                f"Not enough unique points ({len(x_in)}) to fit spline of degree {k}. "
                f"Need more than {k} unique points."
            )
            return None

        # Fit the spline
        spline = UnivariateSpline(x_in, y_in, k=k, s=s, w=w_in)
        logger.debug(
            f"Successfully fitted smoothing spline (k={k}, s={s if s is not None else 'auto'})."
        )
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
    """Detects baseline endpoints by analyzing rolling variance on a fitted spline.

    This method identifies regions of low variance in the smoothed data (using a
    spline) outside a specified exclusion zone, assuming these represent the
    stable baseline regions.

    Args:
        df: DataFrame containing 'Temperature' and 'dCp' columns.
        lower_exclusion_temp: Lower temperature bound of the transition region to exclude
                                from baseline endpoint consideration.
        upper_exclusion_temp: Upper temperature bound of the transition region to exclude.
        window_size: Number of data points to use in the rolling variance window.
                     Must be a positive integer.
        spline_smooth_factor: Smoothing factor `s` passed to `_fit_smoothing_spline`
                              for the initial data smoothing. If None, smoothing is
                              determined automatically by the spline fitter.
        point_selection_method: Method from `EndpointSelectionMethod` enum used to
                                select a single endpoint if multiple points share the
                                minimum variance (`INNERMOST`, `OUTERMOST`, `MIDDLE`).

    Returns:
        A dictionary containing the identified endpoint temperatures:
        {'lower': float | None, 'upper': float | None}.
        Returns None for an endpoint if it cannot be reliably determined.

    Raises:
        ValueError: If input DataFrame is invalid, exclusion temperatures are illogical,
                    or window_size is non-positive.
    """
    logger.info(
        f"Finding spline endpoints. Exclusion: {lower_exclusion_temp:.2f}-"
        f"{upper_exclusion_temp:.2f}, Window: {window_size}, "
        f"Method: {point_selection_method.value}"
    )
    endpoints: Dict[str, Optional[float]] = {"lower": None, "upper": None}

    # --- Input Validation --- START
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Input must be a non-empty pandas DataFrame.")
        return endpoints
    if "Temperature" not in df.columns or "dCp" not in df.columns:
        raise ValueError("DataFrame must contain 'Temperature' and 'dCp' columns.")
    if not is_numeric_dtype(df["Temperature"]) or not is_numeric_dtype(df["dCp"]):
        raise ValueError("'Temperature' and 'dCp' columns must be numeric.")
    if lower_exclusion_temp >= upper_exclusion_temp:
        raise ValueError(
            "Lower exclusion temperature must be less than upper exclusion temperature."
        )
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    # --- Input Validation --- END

    temp: np.ndarray = df["Temperature"].values
    dcp: np.ndarray = df["dCp"].values

    # 1. Fit an initial smoothing spline to the entire dataset
    # This helps to reduce noise before calculating variance.
    initial_spline = _fit_smoothing_spline(temp, dcp, s=spline_smooth_factor)
    if initial_spline is None:
        logger.warning("Failed to fit initial spline for endpoint detection.")
        return endpoints

    # 2. Calculate rolling variance on the *predicted spline values*
    # Low variance in the smoothed curve suggests a stable baseline region.
    try:
        dcp_spline_pred: np.ndarray = initial_spline(temp)
        # Use pandas for convenient rolling calculation
        rolling_variance_series = (
            pd.Series(dcp_spline_pred).rolling(window=window_size, center=True).var()
        )
        # Rolling window introduces NaNs at the edges; fill them by propagating
        # the nearest valid variance value.
        rolling_variance_series = rolling_variance_series.fillna(method="bfill").fillna(
            method="ffill"
        )
        # Handle cases where all values might still be NaN (e.g., very small df)
        if rolling_variance_series.isnull().all():
            logger.warning("Rolling variance calculation resulted in all NaNs.")
            return endpoints
        rolling_variance: np.ndarray = rolling_variance_series.values

    except Exception as e:
        logger.error(f"Error calculating rolling variance: {e}", exc_info=True)
        return endpoints

    # 3. Identify minimum variance points outside the exclusion zone
    df_analysis = pd.DataFrame(
        {"Temperature": temp, "RollingVariance": rolling_variance}
    )

    # Define boolean masks for the regions
    lower_mask = df_analysis["Temperature"] < lower_exclusion_temp
    upper_mask = df_analysis["Temperature"] > upper_exclusion_temp

    min_var_candidates_lower: list[float] = []
    min_var_candidates_upper: list[float] = []

    # Analyze lower baseline region
    if lower_mask.any():
        lower_region = df_analysis[lower_mask]
        min_var_lower = lower_region["RollingVariance"].min()
        # Find all temperatures matching the minimum variance
        min_var_candidates_lower = lower_region.loc[
            np.isclose(lower_region["RollingVariance"], min_var_lower),
            "Temperature",
        ].tolist()
        logger.debug(
            f"Min variance in lower region ({min_var_lower:.4g}) found at temps: "
            f"{min_var_candidates_lower}"
        )
    else:
        logger.warning("No data points found in the lower baseline region.")

    # Analyze upper baseline region
    if upper_mask.any():
        upper_region = df_analysis[upper_mask]
        min_var_upper = upper_region["RollingVariance"].min()
        # Find all temperatures matching the minimum variance
        min_var_candidates_upper = upper_region.loc[
            np.isclose(upper_region["RollingVariance"], min_var_upper),
            "Temperature",
        ].tolist()
        logger.debug(
            f"Min variance in upper region ({min_var_upper:.4g}) found at temps: "
            f"{min_var_candidates_upper}"
        )
    else:
        logger.warning("No data points found in the upper baseline region.")

    # 4. Select a single endpoint from candidates based on the chosen method
    if min_var_candidates_lower:
        candidates = sorted(min_var_candidates_lower)
        n_candidates = len(candidates)
        if point_selection_method == EndpointSelectionMethod.INNERMOST:
            endpoints["lower"] = candidates[-1]  # Highest temperature
        elif point_selection_method == EndpointSelectionMethod.OUTERMOST:
            endpoints["lower"] = candidates[0]  # Lowest temperature
        elif point_selection_method == EndpointSelectionMethod.MIDDLE:
            endpoints["lower"] = candidates[n_candidates // 2]  # Middle element
        else:  # Default fallback to INNERMOST
            endpoints["lower"] = candidates[-1]

    if min_var_candidates_upper:
        candidates = sorted(min_var_candidates_upper)
        n_candidates = len(candidates)
        if point_selection_method == EndpointSelectionMethod.INNERMOST:
            endpoints["upper"] = candidates[0]  # Lowest temperature
        elif point_selection_method == EndpointSelectionMethod.OUTERMOST:
            endpoints["upper"] = candidates[-1]  # Highest temperature
        elif point_selection_method == EndpointSelectionMethod.MIDDLE:
            endpoints["upper"] = candidates[n_candidates // 2]  # Middle element
        else:  # Default fallback to INNERMOST
            endpoints["upper"] = candidates[0]

    if endpoints["lower"] is None or endpoints["upper"] is None:
        logger.warning(
            f"Could not determine both endpoints. Lower: {endpoints['lower']}, Upper: {endpoints['upper']}"
        )
    else:
        logger.info(
            f"Selected endpoints: Lower={endpoints['lower']:.2f}, Upper={endpoints['upper']:.2f}"
        )
    return endpoints


def subtract_spline_baseline(
    df: pd.DataFrame,
    lower_endpoint: float,
    upper_endpoint: float,
    spline_smooth_factor: Optional[float] = None,
    k: int = 3,
) -> Optional[pd.DataFrame]:
    """Fits splines to baseline regions and subtracts the combined baseline.

    This function defines two baseline regions based on the provided endpoints.
    It fits separate smoothing splines to the data points within each region.
    A combined baseline is constructed using these splines and linear interpolation
    between them across the transition region. This combined baseline is then
    subtracted from the original dCp data.

    Args:
        df: DataFrame containing at least 'Temperature' and 'dCp' columns.
        lower_endpoint: The temperature defining the upper limit of the lower
                        baseline region and the start of the transition.
        upper_endpoint: The temperature defining the lower limit of the upper
                        baseline region and the end of the transition.
        spline_smooth_factor: Smoothing factor `s` passed to `_fit_smoothing_spline`
                              for fitting the baseline splines. If None, smoothing
                              is determined automatically by the spline fitter.
        k: Degree of the smoothing splines for the baseline regions (default 3).

    Returns:
        A new DataFrame containing 'Temperature' and 'dCp_subtracted' columns,
        representing the baseline-subtracted data. Returns None if baseline
        subtraction fails (e.g., invalid inputs, spline fitting issues).

    Raises:
        ValueError: If input DataFrame is invalid or endpoints are illogical.
    """
    logger.info(
        f"Subtracting spline baseline. Endpoints: {lower_endpoint:.2f} - "
        f"{upper_endpoint:.2f}, Spline k={k}, s={spline_smooth_factor or 'auto'}"
    )

    # --- Input Validation --- START
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Input must be a non-empty pandas DataFrame.")
        return None
    if "Temperature" not in df.columns or "dCp" not in df.columns:
        raise ValueError("DataFrame must contain 'Temperature' and 'dCp' columns.")
    if not is_numeric_dtype(df["Temperature"]) or not is_numeric_dtype(df["dCp"]):
        raise ValueError("'Temperature' and 'dCp' columns must be numeric.")
    if lower_endpoint >= upper_endpoint:
        raise ValueError("Lower endpoint temperature must be less than upper endpoint.")
    # --- Input Validation --- END

    temp_all: np.ndarray = df["Temperature"].values
    dcp_all: np.ndarray = df["dCp"].values

    # Define masks for the baseline regions
    lower_baseline_mask = temp_all <= lower_endpoint
    upper_baseline_mask = temp_all >= upper_endpoint

    # Extract data for baseline fitting
    temp_lower = temp_all[lower_baseline_mask]
    dcp_lower = dcp_all[lower_baseline_mask]
    temp_upper = temp_all[upper_baseline_mask]
    dcp_upper = dcp_all[upper_baseline_mask]

    # Check if baseline regions have enough data points for spline fitting
    min_points_required = k + 1
    if len(temp_lower) < min_points_required:
        logger.warning(
            f"Lower baseline region has only {len(temp_lower)} points. "
            f"Need at least {min_points_required} for spline degree {k}. Cannot fit lower spline."
        )
        return None
    if len(temp_upper) < min_points_required:
        logger.warning(
            f"Upper baseline region has only {len(temp_upper)} points. "
            f"Need at least {min_points_required} for spline degree {k}. Cannot fit upper spline."
        )
        return None

    # Fit splines to the baseline regions
    spline_lower = _fit_smoothing_spline(
        temp_lower, dcp_lower, s=spline_smooth_factor, k=k
    )
    spline_upper = _fit_smoothing_spline(
        temp_upper, dcp_upper, s=spline_smooth_factor, k=k
    )

    if spline_lower is None or spline_upper is None:
        logger.error("Failed to fit splines to one or both baseline regions.")
        return None

    # Generate the baseline across the entire temperature range
    baseline = np.zeros_like(temp_all)

    # Apply lower spline to its region
    baseline[lower_baseline_mask] = spline_lower(temp_lower)

    # Apply upper spline to its region
    baseline[upper_baseline_mask] = spline_upper(temp_upper)

    # Linearly interpolate between the endpoints in the transition region
    transition_mask = (temp_all > lower_endpoint) & (temp_all < upper_endpoint)
    if np.any(transition_mask):
        temp_transition = temp_all[transition_mask]

        # Get the baseline values predicted by the splines AT the endpoints
        lower_baseline_val_at_endpoint = spline_lower(lower_endpoint)
        upper_baseline_val_at_endpoint = spline_upper(upper_endpoint)

        # Perform linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        slope = (upper_baseline_val_at_endpoint - lower_baseline_val_at_endpoint) / (
            upper_endpoint - lower_endpoint
        )
        interpolated_baseline = (
            lower_baseline_val_at_endpoint + (temp_transition - lower_endpoint) * slope
        )

        baseline[transition_mask] = interpolated_baseline

    # Subtract the calculated baseline
    dcp_subtracted = dcp_all - baseline

    # Create the result DataFrame
    result_df = pd.DataFrame(
        {
            "Temperature": temp_all,
            "dCp_subtracted": dcp_subtracted,
            "dCp_baseline": baseline,
        }
    )

    logger.info("Successfully subtracted spline baseline.")
    return result_df
