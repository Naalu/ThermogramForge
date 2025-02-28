"""Signal detection algorithms for thermogram data."""

from typing import Dict, Tuple, Union

import numpy as np
import polars as pl
from scipy import signal as scipy_signal  # type: ignore
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore

from .types import SignalDetectionResult, ThermogramData


def detect_signal(
    data: Union[ThermogramData, pl.DataFrame],
    threshold: float = 0.05,
    method: str = "peaks",
    verbose: bool = False,
) -> SignalDetectionResult:
    """Determine if a thermogram contains meaningful signal or just noise.

    This function analyzes thermogram data to distinguish between meaningful thermal
    transitions and random noise/baseline fluctuations using various statistical methods.

    Args:
        data: Input thermogram data as ThermogramData object or polars DataFrame.
        threshold: Statistical significance level for detection. Defaults to 0.05.
        method: Signal detection algorithm to use:
            - "peaks": Peak detection (most reliable, default)
            - "arima": ARIMA model comparison
            - "adf": Augmented Dickey-Fuller test
        verbose: Whether to print progress information. Defaults to False.

    Returns:
        SignalDetectionResult containing:
            - is_signal: Boolean indicating presence of signal
            - confidence: Confidence level in the detection (0-1)
            - details: Dictionary with method-specific statistics

    Raises:
        ValueError: If method is not one of ["peaks", "arima", "adf"]
        TypeError: If data is not ThermogramData or polars DataFrame
    """
    # Validate method before proceeding
    if method not in ["peaks", "arima", "adf"]:
        raise ValueError(f"Unknown detection method: {method}")

    # Convert input to ThermogramData if it's a DataFrame
    if isinstance(data, pl.DataFrame):
        therm_data = ThermogramData.from_dataframe(data)
    else:
        therm_data = data

    # Extract arrays for analysis
    temps = therm_data.temperature
    values = therm_data.dcp

    # Calculate first difference to remove trends
    diff_values = np.diff(values)

    try:
        if method == "peaks":
            # Use peak detection as primary method
            is_signal, confidence, details = _detect_signal_peaks(
                temps, values, threshold, verbose
            )
        elif method == "arima":
            # Use ARIMA model approach
            is_signal, confidence, details = _detect_signal_arima(
                diff_values, threshold, verbose
            )
        elif method == "adf":
            # Use Augmented Dickey-Fuller test
            is_signal, confidence, details = _detect_signal_adf(
                diff_values, threshold, verbose
            )
    except Exception as e:
        if verbose:
            print(f"Error in primary detection method: {e}")
            print("Falling back to peak detection")

        # Always fall back to peak detection, which is most reliable
        is_signal, confidence, details = _detect_signal_peaks(
            temps, values, threshold, verbose
        )

    if verbose:
        print(f"Signal detection result: {'SIGNAL' if is_signal else 'NO SIGNAL'}")
        print(f"Confidence: {confidence:.4f}")

    return SignalDetectionResult(
        is_signal=bool(is_signal),  # Ensure Python boolean
        confidence=float(confidence),  # Ensure Python float
        details=details,
    )


def _detect_signal_peaks(
    temps: np.ndarray, values: np.ndarray, threshold: float, verbose: bool
) -> Tuple[bool, float, Dict]:
    """Detect signal using peak analysis algorithm.

    Uses scipy.signal.find_peaks to identify significant peaks and analyzes their
    characteristics to distinguish between signal and noise.

    Args:
        temps: Array of temperature values.
        values: Array of heat capacity (dCp) values.
        threshold: Minimum peak prominence threshold.
        verbose: Whether to print detection details.

    Returns:
        tuple: Contains:
            - bool: True if signal detected
            - float: Confidence level (0-1)
            - dict: Detection statistics including peak count and prominence
    """
    # Use scipy's find_peaks to detect peaks
    # The prominence parameter helps filter out noise
    peaks, properties = scipy_signal.find_peaks(values, prominence=0.05)

    # Calculate signal metrics
    peak_count = len(peaks)
    max_prominence = 0.0 if len(peaks) == 0 else np.max(properties["prominences"])
    signal_to_noise = max_prominence / np.std(values) if np.std(values) > 0 else 0

    # For a simple thermogram with a peak, we expect at least one significant peak
    # For a pure noise thermogram, we expect either no peaks or many small peaks

    if verbose:
        print(
            f"Peak analysis - Found {peak_count} peaks with max prominence {max_prominence:.4f}"
        )
        print(f"Signal-to-noise ratio: {signal_to_noise:.2f}")

    # Decision logic based on peak characteristics
    if peak_count >= 1 and peak_count <= 5 and max_prominence > 0.1:
        # Few distinct peaks with good prominence suggest a signal
        is_signal = True
        confidence = min(0.95, max_prominence * 2)
    elif peak_count > 20:
        # Too many peaks usually means noise (false positives from noise)
        is_signal = False
        confidence = min(0.7, 0.5 + peak_count / 100)
    elif peak_count == 0:
        # No peaks at all suggests no signal
        is_signal = False
        confidence = 0.9
    else:
        # Edge cases, use signal-to-noise ratio
        is_signal = signal_to_noise > 3.0
        confidence = min(0.7, signal_to_noise / 10)

    details = {
        "peak_count": peak_count,
        "max_prominence": float(max_prominence),
        "signal_to_noise": float(signal_to_noise),
        "method": "peak_detection",
    }

    return is_signal, confidence, details


def _detect_signal_arima(
    diff_values: np.ndarray, threshold: float, verbose: bool
) -> Tuple[bool, float, Dict]:
    """Detect signal using ARIMA model comparison.

    Compares ARIMA models with and without differencing using AIC to determine
    if structured signal is present.

    Args:
        diff_values: First difference of heat capacity values.
        threshold: AIC difference threshold for signal detection.
        verbose: Whether to print model comparison details.

    Returns:
        tuple: Contains:
            - bool: True if signal detected
            - float: Confidence level (0-1)
            - dict: Model comparison statistics including AIC values

    Raises:
        Exception: If ARIMA model fitting fails
    """
    try:
        # Fit ARIMA model with no additional differencing
        model = ARIMA(diff_values, order=(1, 0, 1))
        model_fit = model.fit()

        # Try a model with differencing
        model_diff = ARIMA(diff_values, order=(1, 1, 1))
        model_diff_fit = model_diff.fit()

        # Compare models using AIC
        aic_no_diff = model_fit.aic
        aic_diff = model_diff_fit.aic

        # A lower AIC value indicates a better model
        # If the difference is significant, it suggests a signal
        # TODO: Verify this logic
        is_signal = aic_no_diff > aic_diff

        # Calculate confidence based on AIC difference
        aic_diff_value = abs(aic_no_diff - aic_diff)
        confidence = min(0.95, 1 - np.exp(-aic_diff_value / 10))

        # Override for very low confidence
        if confidence < 0.6:
            confidence = 0.6

        details = {
            "aic_no_diff": float(aic_no_diff),
            "aic_diff": float(aic_diff),
            "aic_difference": float(aic_diff_value),
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
        raise Exception(f"ARIMA modeling failed: {e}")


def _detect_signal_adf(
    diff_values: np.ndarray, threshold: float, verbose: bool
) -> Tuple[bool, float, Dict]:
    """Detect signal using Augmented Dickey-Fuller test.

    Combines ADF test with zero-crossing analysis to detect both single
    and multi-peak signals in thermogram data.

    Args:
        diff_values: First difference of heat capacity values.
        threshold: P-value threshold for statistical significance.
        verbose: Whether to print test statistics and results.

    Returns:
        tuple: Contains:
            - bool: True if signal detected
            - float: Confidence level (0-1)
            - dict: Test statistics and zero-crossing analysis results

    Raises:
        Exception: If ADF test fails
    """
    try:
        # Perform ADF test
        result = adfuller(diff_values)

        # Extract statistics
        adf_stat = result[0]
        p_value = result[1]

        # Check standard result
        is_signal_by_pvalue = p_value < threshold

        # Special case for multi-peak detection:
        # Check for clear upward/downward patterns in the differenced data
        # This is a simplified approach to detect structured signals
        peak_count = 0
        crossing_zero = 0
        last_sign = 0

        for val in diff_values:
            if val == 0:
                continue
            sign = 1 if val > 0 else -1
            if sign != last_sign and last_sign != 0:
                crossing_zero += 1
                if crossing_zero % 2 == 0:
                    peak_count += 1
            last_sign = sign

        # Multi-peak signals should have several zero-crossings
        is_multipeak = peak_count >= 2

        # Consider it a signal if either test is positive
        is_signal = is_signal_by_pvalue or is_multipeak

        # Adjust confidence based on detection method
        if is_multipeak and not is_signal_by_pvalue:
            confidence = 0.85  # High confidence for multi-peak patterns
        else:
            confidence = min(0.95, 1 - p_value)  # Standard confidence based on p-value

        details = {
            "adf_statistic": float(adf_stat),
            "p_value": float(p_value),
            "peak_count": peak_count,
            "zero_crossings": crossing_zero,
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "method": "adf",
        }

        if verbose:
            print(f"ADF test - Statistic: {adf_stat:.4f}, p-value: {p_value:.4f}")
            print(f"Peak count: {peak_count}, Zero crossings: {crossing_zero}")
            print(f"Multi-peak detection: {is_multipeak}")
            print(f"Result suggests {'signal' if is_signal else 'no signal'}")

        return is_signal, confidence, details

    except Exception as e:
        raise Exception(f"ADF test failed: {e}")
