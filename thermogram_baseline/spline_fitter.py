"""
SplineFitter module for fitting splines to thermogram data.

This module implements custom spline fitting to match R's smooth.spline functionality,
with a focus on generalized cross-validation (GCV) for automatic smoothing
parameter selection.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np  # type: ignore
from scipy import interpolate  # type: ignore
from scipy.optimize import minimize  # type: ignore


class SplineFitter:
    """
    Custom spline fitting class to replicate R's smooth.spline with cv=TRUE.

    This class implements cross-validation approaches to closely match
    R's generalized cross-validation for smoothing parameter selection.
    """

    def __init__(self) -> None:
        """Initialize SplineFitter."""
        pass

    def fit_with_gcv(
        self, x: np.ndarray, y: np.ndarray, spar: Optional[float] = None
    ) -> interpolate.UnivariateSpline:
        """
        Fit a spline with generalized cross-validation to mimic
        R's smooth.spline(cv=TRUE).

        Args:
            x: Array of x coordinates.
            y: Array of y coordinates.
            spar: Optional smoothing parameter (0-1) similar to R's spar.
                If None, determined automatically via GCV.

        Returns:
            A UnivariateSpline object with optimal smoothing parameter.
        """
        # Validate inputs
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        if len(x) < 4:
            # For very small datasets, use a simple spline with minimal smoothing
            warnings.warn("Very few data points provided. Using simple interpolation.")
            sorted_idx = np.argsort(x)
            x_sorted = x[sorted_idx]
            y_sorted = y[sorted_idx]

            # Handle repeated x values by averaging y values
            x_unique, indices = np.unique(x_sorted, return_inverse=True)
            if len(x_unique) < len(x_sorted):
                # If there are duplicates, average y values for same x
                y_averaged = np.zeros_like(x_unique, dtype=float)
                for i in range(len(x_unique)):
                    mask = indices == i
                    y_averaged[i] = np.mean(y_sorted[mask])

                # Use the averaged data
                x_sorted = x_unique
                y_sorted = y_averaged

            # For small datasets, use a small smoothing parameter
            s = max(0.001, len(x_sorted) * 0.05)
            spline = interpolate.UnivariateSpline(x_sorted, y_sorted, s=s)
            spline.s_opt = s  # Store the optimal smoothing parameter
            spline.cv_score = None  # No CV score available
            return spline

        # Sort data (required for spline fitting)
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        # Handle repeated x values by averaging y values
        x_unique, indices = np.unique(x_sorted, return_inverse=True)
        if len(x_unique) < len(x_sorted):
            # If there are duplicates, average y values for same x
            y_averaged = np.zeros_like(x_unique, dtype=float)
            for i in range(len(x_unique)):
                mask = indices == i
                y_averaged[i] = np.mean(y_sorted[mask])

            # Use the averaged data
            x_sorted = x_unique
            y_sorted = y_averaged

        # Check for constant y values (edge case)
        if np.all(np.abs(y_sorted - y_sorted[0]) < 1e-10):
            # For constant data, use a spline with high smoothing
            s = len(x_sorted) * 10
            spline = interpolate.UnivariateSpline(x_sorted, y_sorted, s=s)
            spline.s_opt = s  # Store the optimal smoothing parameter
            spline.cv_score = None  # No CV score available
            return spline

        # If spar is provided directly, convert to scipy's s parameter
        if spar is not None:
            # Convert spar (0-1) to scipy's s parameter
            # R's spar=0 corresponds to interpolation (s ~ 0)
            # R's spar=1 corresponds to high smoothing (s ~ n)
            n = len(x_sorted)
            if spar <= 0:
                s = 0.0001  # Almost interpolation
            elif spar >= 1:
                s = n * 10  # Very smooth
            else:
                # Exponential mapping seems to work better than linear
                # s = n * (1 - spar)  # Linear mapping
                s = 0.0001 * np.exp((1 - spar) * np.log(n * 100 / 0.0001))

            # Fit with the derived smoothing parameter
            spline = interpolate.UnivariateSpline(x_sorted, y_sorted, s=s)
            spline.s_opt = s
            spline.cv_score = None  # No CV score since we didn't use GCV
            return spline

        # Use GCV to find optimal smoothing parameter
        best_s, best_score = self._find_optimal_smoothing(x_sorted, y_sorted)

        # Fit final spline with optimal smoothing parameter
        spline = interpolate.UnivariateSpline(x_sorted, y_sorted, s=best_s)

        # Store the optimal smoothing parameter for reference
        spline.s_opt = best_s
        spline.cv_score = best_score

        # Try to convert scipy s to R spar (very approximate)
        n = len(x_sorted)
        if best_s <= 0.0001:
            spar_approx = 0
        elif best_s >= n * 10:
            spar_approx = 1
        else:
            # Log-based inverse mapping
            spar_approx = 1 - (np.log(best_s / 0.0001) / np.log(n * 100 / 0.0001))
            spar_approx = max(0, min(1, spar_approx))  # Clamp to [0, 1]

        spline.spar_approx = spar_approx

        return spline

    def _find_optimal_smoothing(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find optimal smoothing parameter using GCV.

        Args:
            x: x coordinates (sorted)
            y: y coordinates

        Returns:
            Tuple of (optimal_s, best_gcv_score)
        """

        # Define function to calculate GCV score for a given smoothing parameter
        def gcv_score(s_param: np.ndarray) -> float:
            # Use the first element of s_param (minimize expects array-like input)
            s = s_param[0]

            try:
                # Prevent too small smoothing values that cause numerical issues
                if s < 0.001:
                    return np.inf

                # Fit spline with current smoothing parameter
                spline = interpolate.UnivariateSpline(x, y, s=s)

                # Calculate fitted values
                y_fitted = spline(x)

                # Calculate residuals
                residuals = y - y_fitted

                # Calculate RSS (residual sum of squares)
                rss = np.sum(residuals**2)

                # Calculate effective degrees of freedom
                # In UnivariateSpline, degrees of freedom is related
                # to number of coefficients
                n = len(x)
                df = n - spline.get_coeffs().size

                # Calculate GCV score: n*RSS/(n-df)^2
                if n <= df:
                    return np.inf  # Avoid division by zero or negative values

                gcv: float = (n * rss) / ((n - df) ** 2)

                return gcv
            except Exception:
                # If fitting fails, return a high score
                return np.inf

        # Find optimal smoothing parameter using optimization
        # Use a range of s values scaled to dataset size
        n = len(x)

        # Try a range of initial values based on dataset size
        # These values are chosen to cover a wide range of smoothing levels
        initial_s_values = [
            0.001,  # Almost interpolation
            0.01 * n,  # Low smoothing
            0.1 * n,  # Medium smoothing
            1.0 * n,  # High smoothing
            10.0 * n,  # Very high smoothing
        ]

        best_s = None
        best_score = np.inf

        for s_init in initial_s_values:
            try:
                # Use Nelder-Mead optimization (robust for this problem)
                res = minimize(
                    gcv_score,
                    [float(s_init)],  # Convert to float and wrap in array
                    method="Nelder-Mead",
                    bounds=[(0.001, None)],  # Lower bound to prevent numerical issues
                    options={"xatol": 1e-3, "maxiter": 100},
                )

                if res.success and res.fun < best_score:
                    best_s = res.x[0]
                    best_score = res.fun
            except Exception:
                continue

        # If optimization failed, use a default value
        if best_s is None or not np.isfinite(best_s):
            # Use a default smoothing parameter based on data size
            # The formula is a common heuristic: roughly n/10
            best_s = max(0.1, len(x) * 0.1)
            best_score = np.inf

        return best_s, best_score

    def compare_with_r_output(
        self,
        x: np.ndarray,
        y: np.ndarray,
        r_fitted: np.ndarray,
        r_spar: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compare Python spline fit with R output.

        This function can be used during testing to validate that
        the Python implementation closely matches R's behavior.

        Args:
            x: x coordinates
            y: y coordinates
            r_fitted: fitted values from R's smooth.spline
            r_spar: Optional R smoothing parameter (spar) if available

        Returns:
            Dictionary with comparison metrics
        """
        # Fit using our implementation
        if r_spar is not None:
            # If we know R's spar, use that directly
            spline = self.fit_with_gcv(x, y, spar=r_spar)
        else:
            # Otherwise, use GCV
            spline = self.fit_with_gcv(x, y)

        py_fitted = spline(x)

        # Calculate various comparison metrics
        # Mean Squared Error
        mse = np.mean((r_fitted - py_fitted) ** 2)

        # Mean Absolute Error
        mae = np.mean(np.abs(r_fitted - py_fitted))

        # Maximum Absolute Difference
        max_diff = np.max(np.abs(r_fitted - py_fitted))

        # Calculate relative differences where r_fitted is not near zero
        mask = np.abs(r_fitted) > 1e-6
        rel_diff = np.abs(py_fitted[mask] - r_fitted[mask]) / np.abs(r_fitted[mask])
        mean_rel_diff = np.mean(rel_diff) * 100 if len(rel_diff) > 0 else 0
        max_rel_diff = np.max(rel_diff) * 100 if len(rel_diff) > 0 else 0

        return {
            "mean_squared_error": mse,
            "mean_absolute_error": mae,
            "max_absolute_difference": max_diff,
            "mean_relative_difference_percent": mean_rel_diff,
            "max_relative_difference_percent": max_rel_diff,
            "optimal_smoothing_parameter": float(getattr(spline, "s_opt", 0.0)),
            "r_spar": float(r_spar if r_spar is not None else 0.0),
            "python_spar_approx": float(getattr(spline, "spar_approx", 0.0)),
        }

    def _calculate_gcv_score(self, x: np.ndarray, y: np.ndarray, s: float) -> float:
        """
        Calculate the GCV score for a given smoothing parameter.

        The GCV criterion is:
            GCV(λ) = RSS(λ) / [n * (1 - df(λ)/n)²]

        Where:
            RSS(λ) = sum of squared residuals
            n = number of data points
            df(λ) = effective degrees of freedom
            λ = smoothing parameter

        Args:
            x: x coordinates (sorted)
            y: y coordinates
            s: smoothing parameter

        Returns:
            GCV score (lower is better)
        """
        try:
            # Prevent too small smoothing values that cause numerical issues
            if s < 0.001:
                return np.inf

            # Fit spline with current smoothing parameter
            spline = interpolate.UnivariateSpline(x, y, s=s)

            # Calculate fitted values
            y_fitted = spline(x)

            # Calculate residuals
            residuals = y - y_fitted

            # Calculate RSS (residual sum of squares)
            rss = np.sum(residuals**2)

            # Calculate effective degrees of freedom
            n = len(x)
            df = n - spline.get_coeffs().size

            # Calculate GCV score: n*RSS/(n-df)^2
            if n <= df:
                return np.inf  # Avoid division by zero or negative

            gcv: float = (n * rss) / ((n - df) ** 2)

            return gcv
        except Exception:
            # If fitting fails, return infinity
            return np.inf
