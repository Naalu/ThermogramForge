"""Signal detection algorithms for thermogram data."""

from typing import Union

import numpy as np
import polars as pl
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from .types import SignalDetectionResult, ThermogramData


def detect_signal(
    data: Union[ThermogramData, pl.DataFrame],
    threshold: float = 0.05,
    method: str = "arima",
    verbose: bool = False,
) -> SignalDetectionResult:
    """
    Determine if a thermogram contains meaningful signal or just noise.

    This function analyzes thermogram data to distinguish between:
    - Meaningful signals representing actual thermal transitions
    - Random noise or baseline fluctuations

    Args:
        data: Thermogram data to analyze
        threshold: Detection threshold (significance level)
        method: Method for signal detection ('arima' or 'adf')
        verbose: Whether to print progress information

    Returns:
        SignalDetectionResult indicating whether the thermogram contains signal
    """
    # Convert input to ThermogramData if it's a DataFrame
    if isinstance(data, pl.DataFrame):
        therm_data = ThermogramData.from_dataframe(data)
    else:
        therm_data = data

    # Extract the dCp values for analysis
    values = therm_data.dcp

    # Calculate first difference to remove trends
    diff_values = _calculate_first_difference(values)

    if method == "arima":
        # Use ARIMA model approach (similar to R's auto.arima)
        is_signal, confidence, details = _detect_signal_arima(
            diff_values, threshold, verbose
        )
    elif method == "adf":
        # Use Augmented Dickey-Fuller test
        is_signal, confidence, details = _detect_signal_adf(
            diff_values, threshold, verbose
        )
    else:
        raise ValueError(f"Unknown detection method: {method}")

    if verbose:
        print(f"Signal detection result: {'SIGNAL' if is_signal else 'NO SIGNAL'}")
        print(f"Confidence: {confidence:.4f}")

    return SignalDetectionResult(
        is_signal=is_signal, confidence=confidence, details=details
    )


def _calculate_first_difference(values: np.ndarray) -> np.ndarray:
    """
    Calculate the first difference of a series.

    Args:
        values: Values to differentiate

    Returns:
        First difference of the values
    """
    return np.diff(values)


def _detect_signal_arima(
    diff_values: np.ndarray, threshold: float, verbose: bool
) -> tuple[bool, float, dict]:
    """
    Detect signal using ARIMA model.

    This method fits an ARIMA model to the first difference of the data
    and checks if differencing is required. If no additional differencing
    is needed, the data is considered to have a signal.

    Args:
        diff_values: First difference of thermogram values
        threshold: Detection threshold
        verbose: Whether to print details

    Returns:
        Tuple of (is_signal, confidence, details)
    """
    try:
        # Fit ARIMA model with no additional differencing
        model = ARIMA(diff_values, order=(2, 0, 2))
        model_fit = model.fit()

        # Try a model with differencing
        model_diff = ARIMA(diff_values, order=(2, 1, 2))
        model_diff_fit = model_diff.fit()

        # Compare models using AIC
        aic_no_diff = model_fit.aic
        aic_diff = model_diff_fit.aic

        # If the model without differencing is better, we likely have a signal
        is_signal = aic_no_diff < aic_diff

        # Calculate confidence based on AIC difference
        aic_diff_value = abs(aic_no_diff - aic_diff)
        confidence = 1 - np.exp(-aic_diff_value / 2)

        details = {
            "aic_no_diff": aic_no_diff,
            "aic_diff": aic_diff,
            "aic_difference": aic_diff_value,
            "method": "arima",
        }

        if verbose:
            print(
                f"ARIMA model comparison - No diff AIC: {aic_no_diff:.2f}, Diff AIC: {aic_diff:.2f}"
            )
            print(
                f"AIC difference: {aic_diff_value:.2f}, suggests {'signal' if is_signal else 'no signal'}"
            )

        return is_signal, confidence, details

    except Exception as e:
        if verbose:
            print(f"Error in ARIMA modeling: {e}")
            print("Falling back to variance analysis")

        # Fallback to simple variance analysis
        return _detect_signal_fallback(diff_values, threshold, verbose)


def _detect_signal_adf(
    diff_values: np.ndarray, threshold: float, verbose: bool
) -> tuple[bool, float, dict]:
    """
    Detect signal using Augmented Dickey-Fuller test.

    This method tests for stationarity in the time series.
    If the series is stationary after first differencing,
    it likely contains a meaningful signal.

    Args:
        diff_values: First difference of thermogram values
        threshold: Significance level threshold
        verbose: Whether to print details

    Returns:
        Tuple of (is_signal, confidence, details)
    """
    try:
        # Perform ADF test
        result = adfuller(diff_values)

        # Extract statistics
        adf_stat = result[0]
        p_value = result[1]

        # If p-value is below threshold, series is stationary
        is_signal = p_value < threshold

        # Confidence is inverse of p-value
        confidence = 1 - p_value

        details = {
            "adf_statistic": adf_stat,
            "p_value": p_value,
            "critical_values": result[4],
            "method": "adf",
        }

        if verbose:
            print(f"ADF test - Statistic: {adf_stat:.4f}, p-value: {p_value:.4f}")
            print(f"Result suggests {'signal' if is_signal else 'no signal'}")

        return is_signal, confidence, details

    except Exception as e:
        if verbose:
            print(f"Error in ADF test: {e}")
            print("Falling back to variance analysis")

        # Fallback to simple variance analysis
        return _detect_signal_fallback(diff_values, threshold, verbose)


def _detect_signal_fallback(
    diff_values: np.ndarray, threshold: float, verbose: bool
) -> tuple[bool, float, dict]:
    """
    Fallback signal detection using variance analysis.

    This method is used when more sophisticated methods fail.
    It compares the variance of the first difference to a baseline.

    Args:
        diff_values: First difference of thermogram values
        threshold: Detection threshold
        verbose: Whether to print details

    Returns:
        Tuple of (is_signal, confidence, details)
    """
    # Calculate variance of differenced values
    variance = np.var(diff_values)

    # Calculate expected variance for pure noise
    # This is a heuristic based on typical noise levels
    noise_threshold = 0.0025  # Baseline variance threshold

    # If variance is significantly larger than expected for noise
    is_signal = variance > noise_threshold

    # Calculate confidence based on variance ratio
    variance_ratio = variance / noise_threshold
    confidence = 1 - (1 / (1 + variance_ratio))

    details = {
        "variance": variance,
        "noise_threshold": noise_threshold,
        "variance_ratio": variance_ratio,
        "method": "variance_fallback",
    }

    if verbose:
        print(
            f"Variance analysis - Value: {variance:.6f}, Threshold: {noise_threshold:.6f}"
        )
        print(
            f"Variance ratio: {variance_ratio:.2f}, suggests {'signal' if is_signal else 'no signal'}"
        )

    return is_signal, confidence, details
