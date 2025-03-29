"""
Test utility functions.
"""

import numpy as np


def compare_results(result, reference, rtol=1e-3, atol=1e-6):
    """
    Compare numerical results with tolerance.

    Args:
        result: Result to test
        reference: Reference result to compare against
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dict with comparison metrics
    """
    # Convert to numpy arrays if needed
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    if not isinstance(reference, np.ndarray):
        reference = np.array(reference)

    # Calculate differences
    abs_diff = np.abs(result - reference)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Calculate relative differences (avoid division by zero)
    mask = np.abs(reference) > atol
    rel_diff = np.zeros_like(abs_diff)
    if np.any(mask):
        rel_diff[mask] = abs_diff[mask] / np.abs(reference[mask])
    max_rel_diff = np.max(rel_diff) * 100  # as percentage
    mean_rel_diff = np.mean(rel_diff) * 100  # as percentage

    # Check if within tolerance
    is_close = np.allclose(result, reference, rtol=rtol, atol=atol)

    return {
        "is_close": is_close,
        "max_absolute_difference": float(max_abs_diff),
        "mean_absolute_difference": float(mean_abs_diff),
        "max_relative_difference_percent": float(max_rel_diff),
        "mean_relative_difference_percent": float(mean_rel_diff),
    }
