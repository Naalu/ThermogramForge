"""
SplineFitter module for fitting splines to thermogram data.

This module implements custom spline fitting to match R's smooth.spline functionality,
with a focus on generalized cross-validation (GCV) for automatic smoothing parameter
selection. It includes an optional rpy2 implementation for direct R integration.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import interpolate  # type: ignore
from scipy.optimize import minimize  # type: ignore

# Check environment variables for configuration
ENV_USE_R = os.environ.get("THERMOGRAM_FORGE_USE_R", "1").lower() in (
    "1",
    "true",
    "yes",
)
ENV_VERBOSE = os.environ.get("THERMOGRAM_FORGE_VERBOSE", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Try to import rpy2 for direct R integration
try:
    import rpy2.robjects as robjects  # type: ignore
    import rpy2.robjects.packages as rpackages  # type: ignore
    from rpy2.robjects import numpy2ri  # type: ignore

    # Activate numpy to R conversion
    numpy2ri.activate()

    # Import R's stats package which contains smooth.spline
    stats = rpackages.importr("stats")

    rpy2_available = True
except ImportError:
    rpy2_available = False


class RSpline:
    """
    Wrapper class for an R smooth.spline object.

    This class provides a Python-friendly interface to an R smooth.spline
    object, making it compatible with the UnivariateSpline interface.
    """

    def __init__(
        self,
        x: np.ndarray,
        fitted: np.ndarray,
        r_spline: Any,
        spar: float,
        df: float,
        lambda_val: Optional[float] = None,
    ) -> None:
        """
        Initialize RSpline.

        Args:
            x: x coordinates
            fitted: fitted y values
            r_spline: R smooth.spline object
            spar: smoothing parameter (0-1)
            df: effective degrees of freedom
            lambda_val: lambda parameter (optional)
        """
        self.x: np.ndarray = x
        self.fitted: np.ndarray = fitted
        self.r_spline: Any = r_spline
        self.spar: float = spar
        self.df: float = df
        self.lambda_val: Optional[float] = lambda_val

        # Store for compatibility with UnivariateSpline
        self.spar_approx: Optional[float] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Predict values at new x points."""
        return self.predict(x)

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """
        Predict values at new x points using the R spline.

        Args:
            x_new: New x coordinates for prediction

        Returns:
            Predicted y values
        """
        # Convert to R vector
        r_x_new = robjects.FloatVector(x_new)

        # Call predict on the R spline
        r_pred = stats.predict(self.r_spline, x=r_x_new)

        # Extract the y values from the prediction
        y_pred = np.array(r_pred.rx2("y"))

        return y_pred

    @property
    def s_opt(self) -> float:
        """Return optimization parameter for compatibility."""
        return getattr(self, "spar", 0.0)

    def get_knots(self) -> np.ndarray:
        """Return knots for compatibility with UnivariateSpline."""
        # For R splines, we don't have direct access to knots
        # Return a reasonable approximation based on degrees of freedom
        num_knots = (
            max(2, int(self.df)) if hasattr(self, "df") and self.df is not None else 10
        )
        # Generate evenly spaced knots across the x range
        return np.asarray(np.linspace(np.min(self.x), np.max(self.x), num_knots))


class SplineFitter:
    """
    Custom spline fitting class to replicate R's smooth.spline with cv=TRUE.

    This class implements cross-validation approaches to closely match
    R's generalized cross-validation for smoothing parameter selection.
    If rpy2 is available, it can also directly call R's smooth.spline function.
    """

    def __init__(self, verbose: Optional[bool] = None) -> None:
        """
        Initialize SplineFitter.

        Args:
            verbose: Whether to print verbose diagnostics during fitting.
                    If None, uses the THERMOGRAM_FORGE_VERBOSE environment variable.
        """
        # Use environment variable if verbose is not explicitly set
        self.verbose = ENV_VERBOSE if verbose is None else verbose
        self._r_available = rpy2_available

        if self.verbose:
            if self._r_available:
                print("rpy2 is available - can use R's smooth.spline directly")
            else:
                print("rpy2 is not available - using Python implementation")

    def fit_with_gcv(
        self,
        x: np.ndarray,
        y: np.ndarray,
        spar: Optional[float] = None,
        use_r: Optional[bool] = None,
    ) -> Union[interpolate.UnivariateSpline, "RSpline"]:
        """
        Fit a spline with generalized cross-validation to mimic
        R's smooth.spline(cv=TRUE).

        Args:
            x: Array of x coordinates.
            y: Array of y coordinates.
            spar: Optional smoothing parameter (0-1) similar to R's spar.
                If None, determined automatically via GCV.
            use_r: Whether to use R's smooth.spline if available (via rpy2).
                If None, uses the THERMOGRAM_FORGE_USE_R environment variable.

        Returns:
            A UnivariateSpline object with optimal smoothing parameter.
            If using R, returns an RSpline object that wraps the R spline.
        """
        # Use environment variable if use_r is not explicitly set
        if use_r is None:
            use_r = ENV_USE_R

        # Log inputs if verbose
        if self.verbose:
            print(f"fit_with_gcv called with {len(x)} data points")
            print(f"spar: {spar}, use_r: {use_r}")

        # Validate inputs
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        # Sort data (required for spline fitting)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Use R's smooth.spline if available and requested
        if self._r_available and use_r:
            if self.verbose:
                print("Using R's smooth.spline")
            return self._fit_with_r(x_sorted, y_sorted, spar)

        # Fall back to Python implementation
        if self.verbose:
            print("Using Python implementation")
        return self._fit_with_python(x_sorted, y_sorted, spar)

    def _fit_with_r(
        self, x: np.ndarray, y: np.ndarray, spar: Optional[float] = None
    ) -> "RSpline":
        """
        Fit a spline using R's smooth.spline via rpy2.

        Args:
            x: Sorted array of x coordinates.
            y: Corresponding array of y coordinates.
            spar: Optional smoothing parameter (0-1).
                If None, determined automatically via GCV.

        Returns:
            An RSpline object that wraps the R spline.
        """
        # Convert numpy arrays to R vectors
        r_x = robjects.FloatVector(x)
        r_y = robjects.FloatVector(y)

        # Prepare call to smooth.spline
        kwargs = {"x": r_x, "y": r_y, "cv": True}

        # Add spar if provided
        if spar is not None:
            kwargs["spar"] = spar

        # Call R's smooth.spline
        r_spline = stats.smooth_spline(**kwargs)

        # Extract fitted values and create RSpline
        r_fitted = np.array(r_spline.rx2("y"))
        r_spar = float(r_spline.rx2("spar")[0])
        r_df = float(r_spline.rx2("df")[0])
        r_lambda = (
            float(r_spline.rx2("lambda")[0]) if "lambda" in r_spline.names else None
        )

        if self.verbose:
            print(f"R fit complete. spar: {r_spar}, df: {r_df}, lambda: {r_lambda}")

        # Create and return RSpline object
        return RSpline(x, r_fitted, r_spline, r_spar, r_df, r_lambda)

    def _fit_with_python(
        self, x: np.ndarray, y: np.ndarray, spar: Optional[float] = None
    ) -> interpolate.UnivariateSpline:
        """
        Fit a spline using Python's implementation.

        Args:
            x: Sorted array of x coordinates.
            y: Corresponding array of y coordinates.
            spar: Optional smoothing parameter (0-1).
                If None, determined automatically via GCV.

        Returns:
            A UnivariateSpline object with optimal smoothing parameter.
        """
        # Handle very small datasets
        if len(x) < 4:
            if self.verbose:
                print("Very few data points provided. Using simple interpolation.")

            # For very small datasets, use a simple spline with minimal smoothing
            s = max(0.001, len(x) * 0.05)
            spline = interpolate.UnivariateSpline(x, y, s=s)
            spline.s_opt = s  # Store the optimal smoothing parameter
            spline.cv_score = None  # No CV score available
            return spline

        # Handle repeated x values by averaging y values
        x_unique, indices = np.unique(x, return_inverse=True)
        if len(x_unique) < len(x):
            if self.verbose:
                print(
                    f"Found {len(x) - len(x_unique)} duplicate x values, ",
                    "averaging y values",
                )

            # If there are duplicates, average y values for same x
            y_averaged = np.zeros_like(x_unique, dtype=float)
            for i in range(len(x_unique)):
                mask = indices == i
                y_averaged[i] = np.mean(y[mask])

            # Use the averaged data
            x_sorted = x_unique
            y_sorted = y_averaged
        else:
            x_sorted = x
            y_sorted = y

        # Check for constant y values (edge case)
        if np.all(np.abs(y_sorted - y_sorted[0]) < 1e-10):
            if self.verbose:
                print("Constant y values detected, using high smoothing")

            # For constant data, use a spline with high smoothing
            s = len(x_sorted) * 10
            spline = interpolate.UnivariateSpline(x_sorted, y_sorted, s=s)
            spline.s_opt = s  # Store the optimal smoothing parameter
            spline.cv_score = None  # No CV score available
            return spline

        # If spar is provided directly, convert to scipy's s parameter
        if spar is not None:
            # Convert spar (0-1) to scipy's s parameter
            n = len(x_sorted)

            # Improved mapping based on empirical testing
            # This is a heuristic mapping that attempts to match R's behavior
            if spar <= 0:
                s = 0.0001  # Almost interpolation
            elif spar >= 1:
                s = n * 10  # Very smooth
            else:
                # Exponential mapping between spar and s
                # This is still a heuristic and might need refinement
                log_min_s = np.log(0.0001)
                log_max_s = np.log(n * 10)
                log_s = log_min_s + (1 - spar) * (log_max_s - log_min_s)
                s = np.exp(log_s)

            if self.verbose:
                print(f"Using provided spar {spar}, converted to s={s}")

            # Fit with the derived smoothing parameter
            spline = interpolate.UnivariateSpline(x_sorted, y_sorted, s=s)
            spline.s_opt = s
            spline.cv_score = None  # No CV score since we didn't use GCV

            # Store the spar value for reference
            spline.spar_input = spar
            return spline

        # Use GCV to find optimal smoothing parameter
        if self.verbose:
            print("Using GCV to find optimal smoothing parameter")

        best_s, best_score = self._find_optimal_smoothing(x_sorted, y_sorted)

        if self.verbose:
            print(f"GCV optimization complete. Best s: {best_s}, score: {best_score}")

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
            # Inverse of the exponential mapping
            log_min_s = np.log(0.0001)
            log_max_s = np.log(n * 10)
            log_s = np.log(best_s)
            spar_approx = 1 - (log_s - log_min_s) / (log_max_s - log_min_s)
            spar_approx = max(0, min(1, spar_approx))  # Clamp to [0, 1]

        if self.verbose:
            print(f"Estimated spar: {spar_approx}")

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
        if self.verbose:
            print("Starting GCV optimization")

        # Define function to calculate GCV score for a given smoothing parameter
        def gcv_score(s_param: Union[float, np.ndarray]) -> float:
            # Use the first element of s_param (minimize expects array-like input)
            s = s_param[0] if isinstance(s_param, np.ndarray) else s_param

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

                if (
                    self.verbose and np.random.random() < 0.1
                ):  # Log only ~10% of evaluations
                    print(f"GCV score for s={s}: {gcv}, df={df}, rss={rss}")

                return gcv
            except Exception as e:
                if self.verbose:
                    print(f"Error in GCV calculation for s={s}: {str(e)}")
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

        if self.verbose:
            print(f"Trying {len(initial_s_values)} initial s values")

        for s_init in initial_s_values:
            try:
                # Calculate initial score
                initial_score = gcv_score(s_init)

                if self.verbose:
                    print(f"Initial s={s_init}, score={initial_score}")

                if initial_score < best_score:
                    best_s = s_init
                    best_score = initial_score

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

                    if self.verbose:
                        print(f"New best: s={best_s}, score={best_score}")
            except Exception as e:
                if self.verbose:
                    print(f"Optimization failed for s_init={s_init}: {str(e)}")
                continue

        # If optimization failed, use a default value
        if best_s is None or not np.isfinite(best_s):
            # Use a default smoothing parameter based on data size
            # The formula is a common heuristic: roughly n/10
            best_s = max(0.1, len(x) * 0.1)
            best_score = np.inf

            if self.verbose:
                print(f"Optimization failed, using default s={best_s}")

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
        # If rpy2 is available, use R for exact comparison
        if self._r_available:
            # Fit using direct R call
            r_spline = self._fit_with_r(x, y, r_spar)
            py_fitted_r = r_spline.predict(x)

            # Compare R's output with provided r_fitted
            mse_r = np.mean((r_fitted - py_fitted_r) ** 2)

            if self.verbose:
                print(f"R-R comparison MSE: {mse_r}")

            # Should be very close to zero
            assert mse_r < 1e-10, "R fitting doesn't match provided r_fitted"

        # Fit using our Python implementation
        if r_spar is not None:
            # Direct spar mapping approach
            spline_direct = self._fit_with_python(x, y, spar=r_spar)
            py_fitted_direct = spline_direct(x)

            # Use the direct mapping for comparison
            py_fitted = py_fitted_direct
            spline = spline_direct
        else:
            # GCV-only approach
            spline = self._fit_with_python(x, y)
            py_fitted = spline(x)
            py_fitted_direct = py_fitted

        # Calculate differences
        abs_diff = np.abs(py_fitted - r_fitted)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        # Calculate relative differences (avoid division by zero)
        mask = np.abs(r_fitted) > 1e-6
        rel_diff = np.zeros_like(r_fitted)
        rel_diff[mask] = abs_diff[mask] / np.abs(r_fitted[mask])
        max_rel_diff = np.max(rel_diff) * 100  # as percentage
        mean_rel_diff = np.mean(rel_diff) * 100  # as percentage

        return {
            "mean_squared_error": float(np.mean((r_fitted - py_fitted) ** 2)),
            "mean_absolute_error": float(mean_abs_diff),
            "max_absolute_difference": float(max_abs_diff),
            "mean_relative_difference_percent": float(mean_rel_diff),
            "max_relative_difference_percent": float(max_rel_diff),
            "optimal_smoothing_parameter": float(getattr(spline, "s_opt", 0.0)),
            "r_spar": float(r_spar if r_spar is not None else 0.0),
            "python_spar_approx": float(getattr(spline, "spar_approx", 0.0)),
        }
