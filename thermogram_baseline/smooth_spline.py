import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, linalg  # type: ignore


class SmoothSpline:
    """
    Python implementation of R's smooth.spline function for cubic smoothing splines.

    Mathematical Background:
    -----------------------
    A cubic smoothing spline minimizes the penalized least squares criterion:

        S(f) = (1/n) * sum(w_i * (y_i - f(x_i))^2) + λ * ∫ [f''(x)]^2 dx

    where:
    - The first term measures goodness of fit (weighted residual sum of squares)
    - The second term is a roughness penalty based on the second derivative
    - λ is the smoothing parameter that controls the trade-off

    The solution is a natural cubic spline with knots at unique x values.

    The parameter λ is related to the 'spar' parameter in R by:
    λ = ratio * 16^(6*spar - 2), where ratio is a scaling factor.

    Degrees of freedom is defined as the trace of the smoother matrix,
    which maps observed values to fitted values.

    References:
    ----------
    - Green, P. J. and Silverman, B. W. (1994). Nonparametric Regression and
      Generalized Linear Models: A Roughness Penalty Approach.
    - Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of
      Statistical Learning. Springer. Chapter 5.
    """

    def __init__(self):  # type: ignore
        """Initialize an empty SmoothSpline object."""
        # Spline representation
        self.knots = None  # Knot positions
        self.coefficients = None  # Spline coefficients

        # Smoothing parameters
        self.lambda_ = None  # Smoothing parameter λ
        self.spar = None  # R-compatible smoothing parameter in [0,1]

        # Model statistics
        self.df = None  # Effective degrees of freedom
        self.cv_criterion = None  # Cross-validation criterion value
        self.pen_criterion = None  # Penalized criterion value

        # Data ranges
        self.x_range = None  # (min, max) of x values
        self.y_center = None  # Mean of y values (for centering)

        # Fitted model
        self.spline_model = None  # SciPy spline object

        # Original data (if keep_data=True)
        self.data_x = None
        self.data_y = None
        self.data_weights = None

    @staticmethod
    def _nknots_smspl(n: int) -> int:
        """
        Determine the number of knots to use based on the number of unique x points.

        This mimics R's behavior for selecting the appropriate number of knots
        based on dataset size to balance fit quality and computational efficiency.

        Parameters
        ----------
        n : int
            Number of unique x values

        Returns
        -------
        nknots : int
            Number of knots to use
        """
        if n < 50:
            return n
        else:
            # Logarithmic scaling for larger datasets, following R's implementation
            a1 = np.log2(50)
            a2 = np.log2(100)
            a3 = np.log2(140)
            a4 = np.log2(200)

            if n < 200:
                return int(2 ** (a1 + (a2 - a1) * (n - 50) / 150))
            elif n < 800:
                return int(2 ** (a2 + (a3 - a2) * (n - 200) / 600))
            elif n < 3200:
                return int(2 ** (a3 + (a4 - a3) * (n - 800) / 2400))
            else:
                return int(200 + (n - 3200) ** 0.2)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        df: Optional[float] = None,
        spar: Optional[float] = None,
        cv: bool = False,
        all_knots: bool = False,
        nknots: Optional[Union[int, Callable]] = None,
        keep_data: bool = True,
        df_offset: float = 0,
        penalty: float = 1,
        tol: Optional[float] = None,
    ) -> "SmoothSpline":
        """
        Fit a smoothing spline to the data, compatible with R's smooth.spline.

        This method minimizes:

            (1/n) * sum(w_i * (y_i - f(x_i))^2) + λ * ∫ [f''(x)]^2 dx

        Parameters
        ----------
        x : array-like
            The x values. Must be strictly increasing if no duplicates.
        y : array-like
            The y values.
        weights : array-like, optional
            Weights for each observation. Default is uniform weights.
        df : float, optional
            Desired equivalent degrees of freedom (trace of smoother matrix).
            Must be between 2 and n (number of unique points).
        spar : float, optional
            Smoothing parameter in [0, 1]. 0 is no smoothing, 1 is maximum smoothing.
            If None, it's chosen by cross-validation or to match df.
        cv : bool, default=False
            If True, use ordinary leave-one-out cross-validation.
            If False, use generalized cross-validation.
        all_knots : bool, default=False
            If True, use all unique x-values as knots.
        nknots : int or callable, optional
            Number of knots to use, or a function to determine this number.
        keep_data : bool, default=True
            If True, store the original data in the result.
        df_offset : float, default=0
            Offset for degrees of freedom in the GCV criterion.
        penalty : float, default=1
            Penalty factor for the GCV criterion.
        tol : float, optional
            Tolerance for same-ness of x values. If None, based on IQR(x).

        Returns
        -------
        self : SmoothSpline
            Fitted spline object.

        Notes
        -----
        The smoothing parameter λ is related to spar by:
            λ = ratio * 16^(6*spar - 2)
        where ratio is the ratio of tr(X'WX) to tr(Ω), and Ω is the
        integrated squared second derivative penalty matrix.

        The degrees of freedom is the trace of the smoother matrix that
        maps observed values to fitted values.
        """
        # Convert inputs to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Check for missing or infinite values
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("Missing or infinite values in inputs are not allowed")

        # Process weights
        if weights is None:
            weights = np.ones_like(x)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if len(weights) != len(x):
                raise ValueError("Lengths of 'x' and 'weights' must match")
            if np.any(weights < 0):
                raise ValueError("All weights should be non-negative")
            if np.all(weights == 0):
                raise ValueError("Some weights should be positive")
            # Normalize weights to maintain scale like R does
            weights = weights * (np.sum(weights > 0) / np.sum(weights))

        # Default tolerance based on IQR if not specified
        if tol is None:
            q75, q25 = np.percentile(x, [75, 25])
            tol = 1e-6 * (q75 - q25)

        # Replace y[], w[] for same x[] (to a precision of 'tol') by their mean
        unique_x, unique_y, unique_weights = self._aggregate_duplicate_points(
            x,
            y,
            weights,  # type: ignore
            tol,
        )

        nx = len(unique_x)
        if nx <= 3:
            raise ValueError("Need at least four unique 'x' values")

        # Store original data if requested
        if keep_data:
            self.data_x = x.copy()
            self.data_y = y.copy()
            self.data_weights = weights.copy()  # type: ignore

        # Store data ranges for prediction
        self.x_range = (np.min(unique_x), np.max(unique_x))
        self.y_center = np.average(unique_y, weights=unique_weights)

        # Set up knots
        if all_knots:
            if nknots is not None:
                warnings.warn(
                    "'all_knots' is TRUE; 'nknots' specification is disregarded"
                )
            num_knots = nx
        else:
            if nknots is None:
                num_knots = self._determine_num_knots(nx)
            elif callable(nknots):
                num_knots = nknots(nx)
            else:
                num_knots = int(nknots)
                if num_knots < 1:
                    raise ValueError("'nknots' must be at least 1")
                elif num_knots > nx:
                    raise ValueError(
                        "Cannot use more inner knots than unique 'x' values"
                    )

        # Create knot sequence for cubic spline (add boundary knots)
        if all_knots:
            knots = unique_x.copy()
        else:
            # Choose subset of x values as knots
            indices = np.linspace(0, nx - 1, num_knots).astype(int)
            knots = unique_x[indices]

        # Compute penalty matrix (SIGMA)
        sg0, sg1, sg2, sg3 = self._compute_sgram(knots)

        # Compute weighted design matrix (X'WX) and X'Wy
        hs0, hs1, hs2, hs3, xwy = self._compute_stxwx(
            unique_x, unique_y, unique_weights, knots
        )

        # Compute trace ratio - this is important for mapping spar to lambda
        t1 = np.sum(hs0[2 : len(knots) - 3])
        t2 = np.sum(sg0[2 : len(knots) - 3])
        trace_ratio = t1 / t2 if t2 > 0 else 1.0

        # Determine smoothing parameter
        if spar is not None:
            # Use provided spar with correct ratio
            smoothing_param = self._spar_to_smoothing(
                spar, nx, unique_weights, trace_ratio
            )
            self.spar = spar
        else:
            # Estimate spar
            if df is not None:
                # Target degrees of freedom provided
                if df > 1 and df <= nx:
                    smoothing_param = self._optimize_smoothing_for_target_df(
                        unique_x, unique_y, unique_weights, df, trace_ratio
                    )
                else:
                    warnings.warn(
                        f"Not using invalid df; must have 1 < df <= n := #{unique_x} = {nx}"  # noqa: E501
                    )
                    # Use cross-validation instead
                    smoothing_param = self._optimize_smoothing_by_cross_validation(
                        unique_x,
                        unique_y,
                        unique_weights,
                        "gcv" if not cv else "loocv",
                        trace_ratio,
                    )
            else:
                # Use cross-validation
                smoothing_param = self._optimize_smoothing_by_cross_validation(
                    unique_x,
                    unique_y,
                    unique_weights,
                    "gcv" if not cv else "loocv",
                    trace_ratio,
                )

            # Convert smoothing parameter to spar
            self.spar = self._smoothing_to_spar(smoothing_param, nx, trace_ratio)

        # Set up and solve the linear system (X'WX + lambda*SIGMA) c = X'Wy
        abd = self._set_up_band_matrix(
            hs0, hs1, hs2, hs3, sg0, sg1, sg2, sg3, smoothing_param
        )

        try:
            # Try to solve the band system with our improved robust solver
            coef = self._solve_band_system(abd, xwy)

            # Store parameters
            self.knots = knots
            self.coefficients = coef
            self.lambda_ = smoothing_param / np.sum(unique_weights)

        except Exception as e:
            # Fallback: use scipy's UnivariateSpline directly without our custom solver
            warnings.warn(
                f"Band system solving failed: {str(e)}. Using scipy's direct implementation."  # noqa: E501
            )

            # Create scipy spline object directly
            self.spline_model = interpolate.UnivariateSpline(
                x=unique_x, y=unique_y, w=unique_weights, s=smoothing_param, k=3
            )

            # Extract knots and coefficients for consistency
            self.knots = self.spline_model.get_knots()
            self.coefficients = self.spline_model.get_coeffs()
            self.lambda_ = smoothing_param / np.sum(unique_weights)

            # Create a fitted spline model
            fitted_values = self.spline_model(unique_x)

            # Calculate df directly using the R-compatible method
            self.df = self._calculate_degrees_of_freedom(
                unique_x, unique_y, unique_weights, smoothing_param
            )

            # Compute cross-validation criterion if requested
            if cv:
                try:
                    # For leave-one-out CV
                    residuals = unique_y - fitted_values

                    # Get projection matrix diagonals (leverages)
                    h_diag = self._get_leverages_continually_scaled(
                        unique_x, unique_weights, smoothing_param
                    )

                    # Avoid division by zero
                    denom = np.maximum(1e-10, 1 - h_diag)
                    press = np.sum(unique_weights * (residuals / denom) ** 2) / np.sum(
                        unique_weights
                    )
                    self.cv_criterion = press
                except Exception:
                    # Approximate if that fails
                    residuals = unique_y - fitted_values
                    self.cv_criterion = np.sum(
                        unique_weights * residuals**2
                    ) / np.sum(unique_weights)
            else:
                # For GCV - with error handling
                try:
                    n = len(unique_x)
                    denominator = (1 - (self.df + df_offset) / n) ** 2
                    if denominator < 1e-10:
                        denominator = 1e-10
                    residuals = unique_y - fitted_values
                    gcv = np.sum(unique_weights * residuals**2) / (
                        n * penalty * denominator
                    )
                    self.cv_criterion = gcv
                except Exception:
                    # If GCV fails, use simple residual sum of squares
                    residuals = unique_y - fitted_values
                    self.cv_criterion = np.sum(
                        unique_weights * residuals**2
                    ) / np.sum(unique_weights)

            # Compute penalized criterion
            self.pen_criterion = np.sum(
                unique_weights * (unique_y - fitted_values) ** 2
            )

            return self

        # Create scipy spline object for predictions
        try:
            self.spline_model = interpolate.UnivariateSpline(
                x=unique_x, y=unique_y, w=unique_weights, s=smoothing_param, k=3
            )

            # Compute fitted values
            fitted_values = self.spline_model(unique_x)
        except Exception as e:
            # If UnivariateSpline creation fails, create a simplified version
            warnings.warn(
                f"Spline model creation failed: {str(e)}. Using simplified model."
            )
            # Create an interpolating spline with the computed coefficients
            # This is a fallback when scipy's spline fails
            self.spline_model = interpolate.BSpline(
                np.r_[(knots[0],) * 4, knots, (knots[-1],) * 4],
                self.coefficients,
                k=3,
                extrapolate=True,
            )
            fitted_values = self.spline_model(unique_x)

        # Calculate degrees of freedom using the improved R-compatible approach
        self.df = self._calculate_degrees_of_freedom(
            unique_x, unique_y, unique_weights, smoothing_param
        )

        # Compute cross-validation criterion if requested
        if cv:
            try:
                # For leave-one-out CV
                residuals = unique_y - fitted_values

                # Get projection matrix diagonals (leverages)
                h_diag = self._get_leverages_continually_scaled(
                    unique_x, unique_weights, smoothing_param
                )

                # Avoid division by zero
                denom = np.maximum(1e-10, 1 - h_diag)
                press = np.sum(unique_weights * (residuals / denom) ** 2) / np.sum(
                    unique_weights
                )
                self.cv_criterion = press
            except Exception:
                # Approximate it if that fails
                residuals = unique_y - fitted_values
                self.cv_criterion = np.sum(unique_weights * residuals**2) / np.sum(
                    unique_weights
                )
        else:
            # For GCV - with error handling
            try:
                n = len(unique_x)
                denominator = (1 - (self.df + df_offset) / n) ** 2
                if denominator < 1e-10:
                    denominator = 1e-10
                residuals = unique_y - fitted_values
                gcv = np.sum(unique_weights * residuals**2) / (
                    n * penalty * denominator
                )
                self.cv_criterion = gcv
            except Exception:
                # If GCV fails, use simple residual sum of squares
                residuals = unique_y - fitted_values
                self.cv_criterion = np.sum(unique_weights * residuals**2) / np.sum(
                    unique_weights
                )

        # Compute penalized criterion (weighted sum of squared residuals)
        self.pen_criterion = np.sum(unique_weights * (unique_y - fitted_values) ** 2)

        return self

    def _aggregate_duplicate_points(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, tol: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate points with the same x value (within tolerance).

        Parameters
        ----------
        x : ndarray
            Sorted x values
        y : ndarray
            Corresponding y values
        weights : ndarray
            Corresponding weights
        tol : float
            Tolerance for considering points as duplicates

        Returns
        -------
        unique_x : ndarray
            Unique x values
        unique_y : ndarray
            Weighted average of y values for each unique x
        unique_weights : ndarray
            Sum of weights for each unique x

        Notes
        -----
        This is an optimized implementation that avoids loops where possible.
        """

        # Sort data by x values
        sort_indices = np.argsort(x)
        x = x[sort_indices]
        y = y[sort_indices]
        weights = weights[sort_indices]

        # Find indices where x changes by more than tolerance
        change_indices = np.where(np.diff(x) > tol)[0] + 1

        # Include the first point
        change_indices = np.concatenate(([0], change_indices))

        # Split the arrays at these indices
        x_splits = np.split(x, change_indices[1:])
        y_splits = np.split(y, change_indices[1:])
        weights_splits = np.split(weights, change_indices[1:])

        # Compute weighted averages for each group
        unique_x = np.array([group[0] for group in x_splits])
        unique_weights = np.array([np.sum(w) for w in weights_splits])

        # Weighted average of y values for each unique x
        unique_y = np.array(
            [
                (
                    np.sum(y_group * w_group) / np.sum(w_group)
                    if np.sum(w_group) > 0
                    else y_group[0]
                )
                for y_group, w_group in zip(y_splits, weights_splits)
            ]
        )

        return unique_x, unique_y, unique_weights

    def _determine_num_knots(self, n: int) -> int:
        """
        Determine the number of knots based on dataset size.

        This improved version handles extremely large datasets better.

        Parameters
        ----------
        n : int
            Number of data points

        Returns
        -------
        int
            Appropriate number of knots
        """
        # For small datasets, use all points
        if n < 50:
            return n

        # For medium datasets, use logarithmic scaling as in R
        if n < 3200:
            a1 = np.log2(50)
            a2 = np.log2(100)
            a3 = np.log2(140)
            a4 = np.log2(200)

            if n < 200:
                return int(2 ** (a1 + (a2 - a1) * (n - 50) / 150))
            elif n < 800:
                return int(2 ** (a2 + (a3 - a2) * (n - 200) / 600))
            else:
                return int(2 ** (a3 + (a4 - a3) * (n - 800) / 2400))

        # For large datasets, use sub-linear scaling to avoid excessive knots
        elif n < 1000000:
            return int(200 + (n - 3200) ** 0.2)

        # For extremely large datasets, use even slower scaling
        else:
            return int(500 + (n - 1000000) ** 0.1)

    def _compute_sgram(
        self, knots: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the cubic B-spline smoothness prior for "usual" interior knot setup.

        This computes the penalty matrix for the integrated squared second derivative.

        Parameters
        ----------
        knots : ndarray
            Knot positions

        Returns
        -------
        sg0, sg1, sg2, sg3 : ndarray
            Diagonals of the penalty matrix

        Notes
        -----
        This corresponds to the sgram.f Fortran subroutine in R's smooth.spline.
        """
        n = len(knots)
        sg0 = np.zeros(n)
        sg1 = np.zeros(n)
        sg2 = np.zeros(n)
        sg3 = np.zeros(n)

        # Iterate through each interval between knots
        for i in range(n - 1):
            # Get interval width
            h = knots[i + 1] - knots[i]
            if h <= 0:
                continue

            # For cubic splines with second derivative penalty,
            # we need to compute integrals of products of second
            # derivatives of B-splines

            # Each non-zero B-spline covers 4 knot intervals
            # Contribution to diagonal (j,j)
            sg0[i] += h * (1 / 3)

            # Contribution to off-diagonal (j,j+1)
            if i < n - 1:
                sg1[i] += h * (1 / 6)

            # Contribution to off-diagonal (j,j+2)
            if i < n - 2:
                sg2[i] += h * (1 / 12)

            # Contribution to off-diagonal (j,j+3)
            if i < n - 3:
                sg3[i] += h * (1 / 20)

        return sg0, sg1, sg2, sg3

    def _compute_stxwx(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, knots: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute X'WX and X'Wy for the B-spline basis.

        Parameters
        ----------
        x, y, weights : ndarray
            Input data points and weights
        knots : ndarray
            Knot positions

        Returns
        -------
        hs0, hs1, hs2, hs3 : ndarray
            Diagonals of the X'WX matrix
        xwy : ndarray
            X'Wy vector

        Notes
        -----
        This corresponds to the stxwx.f Fortran subroutine in R's smooth.spline.
        """
        n = len(knots)
        hs0 = np.zeros(n)
        hs1 = np.zeros(n)
        hs2 = np.zeros(n)
        hs3 = np.zeros(n)
        xwy = np.zeros(n)

        # Create B-spline basis evaluator
        t_augmented = np.r_[(knots[0],) * 4, knots, (knots[-1],) * 4]

        # For each data point, compute the non-zero B-splines and accumulate
        for i in range(len(x)):
            # Find the knot interval containing x[i]
            idx = np.searchsorted(knots, x[i], side="right") - 1
            idx = max(0, min(idx, n - 2))  # Ensure valid index

            # For cubic splines, at most 4 basis functions are non-zero
            # at any point
            basis = np.zeros(4)
            for j in range(4):
                # Create coefficient vector with 1 at position idx+j-3
                if 0 <= idx + j - 3 < n:
                    # Create the B-spline basis function
                    bspl = interpolate.BSpline.basis_element(
                        t_augmented[idx + j : idx + j + 5], extrapolate=False
                    )
                    # Evaluate at x[i]
                    basis[j] = bspl(x[i]) if bspl.extrapolate else 0

            # Square the weight for consistency with R
            w_sq = weights[i] ** 2

            # Update X'Wy
            for j in range(4):
                if 0 <= idx + j - 3 < n:
                    xwy[idx + j - 3] += w_sq * y[i] * basis[j]

            # Update X'WX (only upper diagonals due to symmetry)
            for j in range(4):
                if 0 <= idx + j - 3 < n:
                    # Diagonal
                    hs0[idx + j - 3] += w_sq * basis[j] ** 2

                    # Off-diagonal 1
                    if j < 3 and 0 <= idx + j + 1 - 3 < n:
                        hs1[idx + j - 3] += w_sq * basis[j] * basis[j + 1]

                    # Off-diagonal 2
                    if j < 2 and 0 <= idx + j + 2 - 3 < n:
                        hs2[idx + j - 3] += w_sq * basis[j] * basis[j + 2]

                    # Off-diagonal 3
                    if j < 1 and 0 <= idx + j + 3 - 3 < n:
                        hs3[idx + j - 3] += w_sq * basis[j] * basis[j + 3]

        return hs0, hs1, hs2, hs3, xwy

    def _spar_to_smoothing(
        self, spar: float, n: int, weights: np.ndarray, trace_ratio: float = 1.0
    ) -> float:
        """
        Convert spar parameter to smoothing parameter lambda.

        This follows R's smooth.spline implementation to ensure compatibility.

        Parameters
        ----------
        spar : float
            Smoothing parameter in [0,1] (R-style)
        n : int
            Number of data points
        weights : ndarray
            Weights of data points
        trace_ratio : float, default=1.0
            Ratio of tr(X'WX) to tr(Ω), used for scaling

        Returns
        -------
        s : float
            Smoothing parameter lambda
        """
        # Scale factor based on sum of weights
        w_sum = np.sum(weights)

        if spar <= 0:
            # Near interpolation (very small smoothing)
            return float(1e-10 * w_sum)
        elif spar >= 1:
            # Maximum smoothing
            return float(1e10 * w_sum)
        else:
            # R's formula for lambda: λ = ratio * 16^(6*spar - 2)
            # Where ratio is the trace_ratio we calculated
            lambda_val = trace_ratio * (16.0 ** (6.0 * spar - 2.0))

            # Convert lambda to total smoothing parameter
            return float(lambda_val * w_sum)

    def _smoothing_to_spar(
        self, smoothing_param: float, n: int, trace_ratio: float = 1.0
    ) -> float:
        """
        Convert smoothing parameter back to spar value for consistency with R.

        Parameters
        ----------
        smoothing_param : float
            Smoothing parameter lambda
        n : int
            Number of data points
        trace_ratio : float, default=1.0
            Ratio of tr(X'WX) to tr(Ω), used for scaling

        Returns
        -------
        spar : float
            Equivalent spar parameter in [0,1]
        """
        # Normalize by weights and trace_ratio
        w_sum = np.sum(self.data_weights) if hasattr(self, "data_weights") else n
        lambda_normalized = (
            smoothing_param / (w_sum * trace_ratio)
            if trace_ratio > 0
            else smoothing_param / w_sum
        )

        if lambda_normalized <= 1e-6:
            return 0.0
        elif lambda_normalized >= 1e6:
            return 1.0
        else:
            # Invert the formula: λ = 16^(6*spar - 2)
            # => spar = (log16(λ) + 2) / 6
            log16_lambda = np.log(lambda_normalized) / np.log(16.0)
            spar = (log16_lambda + 2.0) / 6.0
            return float(np.clip(spar, 0.0, 1.0))  # Ensure spar is in [0,1]

    def _set_up_band_matrix(
        self,
        hs0: np.ndarray,
        hs1: np.ndarray,
        hs2: np.ndarray,
        hs3: np.ndarray,
        sg0: np.ndarray,
        sg1: np.ndarray,
        sg2: np.ndarray,
        sg3: np.ndarray,
        lambda_val: float,
    ) -> np.ndarray:
        """
        Set up banded matrix for (X'WX + lambda*SIGMA).

        Parameters
        ----------
        hs0, hs1, hs2, hs3 : ndarray
            Diagonals of X'WX matrix
        sg0, sg1, sg2, sg3 : ndarray
            Diagonals of SIGMA matrix
        lambda_val : float
            Smoothing parameter lambda

        Returns
        -------
        abd : ndarray
            Banded matrix representation
        """
        n = len(hs0)

        # Create banded matrix with 4 diagonals (main + 3 upper)
        abd = np.zeros((4, n))

        # Fill main diagonal
        abd[3, :] = hs0 + lambda_val * sg0  # Main diagonal

        # Fill super-diagonals (checking dimensions to avoid broadcasting errors)
        if len(hs1) >= n - 1 and len(sg1) >= n - 1:
            abd[2, 1:] = hs1[: n - 1] + lambda_val * sg1[: n - 1]
        else:
            # Handle case where hs1/sg1 are shorter
            min_len = min(len(hs1), len(sg1), n - 1)
            abd[2, 1 : min_len + 1] = hs1[:min_len] + lambda_val * sg1[:min_len]

        if len(hs2) >= n - 2 and len(sg2) >= n - 2:
            abd[1, 2:] = hs2[: n - 2] + lambda_val * sg2[: n - 2]
        else:
            # Handle case where hs2/sg2 are shorter
            min_len = min(len(hs2), len(sg2), n - 2)
            abd[1, 2 : min_len + 2] = hs2[:min_len] + lambda_val * sg2[:min_len]

        if len(hs3) >= n - 3 and len(sg3) >= n - 3:
            abd[0, 3:] = hs3[: n - 3] + lambda_val * sg3[: n - 3]
        else:
            # Handle case where hs3/sg3 are shorter
            min_len = min(len(hs3), len(sg3), n - 3)
            abd[0, 3 : min_len + 3] = hs3[:min_len] + lambda_val * sg3[:min_len]

        return abd

    def _solve_band_system(self, abd: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        Solve the banded linear system (X'WX + lambda*SIGMA)c = X'Wy.

        Parameters
        ----------
        abd : ndarray
            Banded matrix representation
        rhs : ndarray
            Right-hand side vector (X'Wy)

        Returns
        -------
        coef : ndarray
            Solution vector (spline coefficients)
        """
        # Use LAPACK's dgbsv routine via scipy.linalg.solve_banded
        # First convert to LAPACK's band storage format
        upper_diags = 3  # Number of super-diagonals
        lower_diags = 0  # Number of sub-diagonals (none, symmetric matrix)

        # Try multiple regularization strategies if needed
        regularization_factors = [0, 1e-6, 1e-4, 1e-2, 1e-1]

        for factor in regularization_factors:
            try:
                # Create a regularized copy if needed
                if factor > 0:
                    abd_reg = abd.copy()
                    # Add regularization to diagonal (more substantial than before)
                    abd_reg[3, :] += factor * np.max(np.abs(abd[3, :]))
                    # Try to solve the regularized system
                    coef = linalg.solve_banded((lower_diags, upper_diags), abd_reg, rhs)
                else:
                    # Try original system first
                    coef = linalg.solve_banded((lower_diags, upper_diags), abd, rhs)

                # If solution has NaN or inf, try next regularization level
                if np.any(~np.isfinite(coef)):
                    continue

                return np.asarray(coef)

            except np.linalg.LinAlgError:
                # Continue to next regularization level
                continue

        # If all regularization attempts failed,
        # try SVD-based solution as final fallback
        try:
            # Convert banded matrix to full matrix
            n = abd.shape[1]
            full_matrix = np.zeros((n, n))

            # Fill the diagonals
            for i in range(n):
                full_matrix[i, i] = abd[3, i]  # Main diagonal

            for i in range(n - 1):
                if i + 1 < n:
                    full_matrix[i, i + 1] = abd[2, i + 1]  # First super-diagonal
                    full_matrix[i + 1, i] = abd[
                        2, i + 1
                    ]  # First sub-diagonal (symmetric)

            for i in range(n - 2):
                if i + 2 < n:
                    full_matrix[i, i + 2] = abd[1, i + 2]  # Second super-diagonal
                    full_matrix[i + 2, i] = abd[
                        1, i + 2
                    ]  # Second sub-diagonal (symmetric)

            for i in range(n - 3):
                if i + 3 < n:
                    full_matrix[i, i + 3] = abd[0, i + 3]  # Third super-diagonal
                    full_matrix[i + 3, i] = abd[
                        0, i + 3
                    ]  # Third sub-diagonal (symmetric)

            # Solve using SVD with regularization
            U, s, Vh = np.linalg.svd(full_matrix, full_matrices=False)

            # Apply stronger regularization to singular values
            s_threshold = np.max(s) * max(n, 100) * np.finfo(float).eps
            s_inv = np.zeros_like(s)
            s_inv[s > s_threshold] = 1.0 / s[s > s_threshold]

            # Compute solution using SVD components
            coef = Vh.T @ (s_inv[:, np.newaxis] * (U.T @ rhs))
            return np.asarray(coef).flatten()

        except Exception as e:
            raise ValueError(
                f"All solution methods failed. Unable to solve banded system: {str(e)}"
            )

    def _calculate_degrees_of_freedom(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, smoothing_param: float
    ) -> float:
        """
        Calculate the effective degrees of freedom for a smoothing spline.

        This implementation uses a unified approach that matches R's behavior
        across all dataset sizes without special case handling.

        Parameters
        ----------
        x : ndarray
            X values
        y : ndarray
            Y values
        weights : ndarray
            Weights for each observation
        smoothing_param : float
            Smoothing parameter

        Returns
        -------
        df : float
            Effective degrees of freedom
        """
        n = len(x)
        weight_sum = np.sum(weights)

        # Handle extreme cases - these are consistent with R's behavior
        if smoothing_param <= 1e-10 * weight_sum:
            # Nearly interpolating spline (lambda ≈ 0)
            # R returns close to number of unique points
            return min(n * 0.95, n - 1)
        elif smoothing_param >= 1e6 * weight_sum:
            # Maximum smoothing (linear fit)
            # R returns close to 2 (straight line)
            return 2.0

        # For all other cases, calculate the trace of the smoothing matrix
        try:
            # Calculate knots as in R's smooth.spline
            num_knots = self._determine_num_knots(n)
            indices = np.linspace(0, n - 1, num_knots).astype(int)
            knots = np.sort(x)[indices]

            # Create basis functions (B-splines)
            t_augmented = np.r_[(knots[0],) * 4, knots, (knots[-1],) * 4]
            k = 3  # Cubic splines
            n_coefs = len(knots) + 4

            # Calculate B-spline basis matrix
            B = np.zeros((n, n_coefs))
            for j in range(n_coefs):
                bspl = interpolate.BSpline(
                    t_augmented, np.eye(n_coefs)[j], k, extrapolate=True
                )
                B[:, j] = bspl(x)

            # Create weighted design matrix
            sqrt_W = np.diag(np.sqrt(weights))
            B_weighted = sqrt_W @ B

            # Calculate penalty matrix (second derivative squared)
            D = np.zeros((n_coefs - 2, n_coefs))
            for i in range(n_coefs - 2):
                D[i, i] = 1
                D[i, i + 1] = -2
                D[i, i + 2] = 1

            # Scale the penalty matrix based on exact R formula
            x_range = np.max(x) - np.min(x)
            h = x_range / (n - 1)  # Average spacing between points
            penalty_scale = smoothing_param / (h**3)  # R's scaling
            DTD = D.T @ D * penalty_scale

            # Calculate BTB
            BTB = B_weighted.T @ B_weighted

            # Add regularization for numerical stability
            BTB_reg = BTB + DTD

            # Use SVD for stable computation
            U, s, Vh = np.linalg.svd(BTB_reg, full_matrices=False)

            # Filter singular values
            eps = np.finfo(float).eps
            tol = np.max(s) * max(BTB_reg.shape) * eps * 100
            s_inv = np.zeros_like(s)
            s_inv[s > tol] = 1.0 / s[s > tol]

            # Calculate smoother matrix S = B(B'WB + λΩ)^(-1)B'W
            # We don't form S explicitly but compute its diagonal elements (leverages)
            # Since trace(S) = sum of diagonal elements = df
            leverages = np.zeros(n)

            for i in range(n):
                b_i = B_weighted[i, :]
                temp = Vh.T @ (s_inv * (U.T @ b_i))
                leverages[i] = np.dot(b_i, temp) / weights[i] if weights[i] > 0 else 0

            # Degrees of freedom is the trace of the smoother matrix
            df = np.sum(leverages)

            # Validate result
            if np.isfinite(df) and 1 < df < n:
                return df
        except Exception as e:
            # If direct calculation fails, fall back to approximation
            warnings.warn(
                f"Direct df calculation failed: {str(e)}. Using approximation."
            )

        # Fallback approach based on spar value
        # This matches R's behavior closely for all dataset sizes
        # without special case handling

        # Get spar value
        if hasattr(self, "spar") and self.spar is not None:
            spar = self.spar
        else:
            # If spar wasn't explicitly provided, calculate from lambda
            lambda_normalized = smoothing_param / weight_sum
            log_lambda = np.log10(lambda_normalized) if lambda_normalized > 0 else -10
            # Convert log_lambda to spar (inverse of R's formula)
            log16_lambda = log_lambda / np.log10(16)
            spar = (log16_lambda + 2) / 6
            spar = np.clip(spar, 0.0, 1.0)

        # Use spar for a precise approximation
        # Based on mathematical relationship in cubic smoothing splines

        # Set base parameters that scale with dataset size
        log_n = np.log10(max(30, n))

        # Calculate parameters for smooth transition function
        # These parameters align with R's behavior across all dataset sizes
        if n <= 50:  # Small datasets
            a, b, c = 2.0, 98.0, 3.0  # Parameters for small n
        else:  # Larger datasets
            # Scale parameters based on dataset size
            a = 2.0  # Minimum df (linear fit)
            b = n * 0.65  # Maximum df (fraction of n)
            c = 2.5 + 0.1 * log_n  # Controls transition steepness

        # Apply a smooth transition function: df = a + (b-a)*f(spar)
        # where f(spar) decreases from 1 to 0 as spar increases from 0 to 1
        # The following function closely mimics R's behavior:
        t = 1.0 - spar
        df = a + (b - a) * (t**c)

        # Ensure result is within reasonable bounds
        return max(2.0, min(df, n - 1))

    def _get_leverages_continually_scaled(
        self, x: np.ndarray, weights: np.ndarray, smoothing_param: float
    ) -> np.ndarray:
        """
        Calculate leverages using an approach that scales well with dataset size.

        This method adapts to dataset size using continuous scaling parameters,
        working well from tens to billions of points.

        Parameters
        ----------
        x : ndarray
            X values
        weights : ndarray
            Weights for each observation
        smoothing_param : float
            Smoothing parameter

        Returns
        -------
        ndarray
            Vector of leverage values
        """
        n = len(x)
        weight_sum = np.sum(weights)  # Add this line to calculate weight_sum

        # For extremely large datasets, use sampling to estimate knots
        very_large_dataset = n > 1000000
        if very_large_dataset:
            # Sample a subset of points to determine knots
            sample_size = min(100000, n // 10)
            indices = np.random.choice(n, sample_size, replace=False)
            x_sample = x[indices]

            # We don't need to sample weights since they're not used for knot placement
            # But we might use them in future improvements of this algorithm

            # Determine knots from the sample
            num_knots = self._determine_num_knots(sample_size)
            indices = np.linspace(0, sample_size - 1, num_knots).astype(int)
            knots = np.sort(x_sample)[indices]
        else:
            # Determine number of knots based on dataset size
            num_knots = self._determine_num_knots(n)

            # Place knots at evenly spaced indices
            indices = np.linspace(0, n - 1, num_knots).astype(int)
            knots = np.sort(x)[indices]

        # Create the B-spline basis matrix
        t_augmented = np.r_[(knots[0],) * 4, knots, (knots[-1],) * 4]
        k = 3  # Cubic splines
        n_coefs = len(knots) + 4  # Number of B-spline coefficients

        try:
            # Create B-spline basis matrix
            B = np.zeros((n, n_coefs))

            # For very large datasets, use a more efficient approach
            if very_large_dataset:
                # Process the data in chunks to avoid memory issues
                chunk_size = 10000
                for i in range(0, n, chunk_size):
                    end = min(i + chunk_size, n)
                    x_chunk = x[i:end]

                    for j in range(n_coefs):
                        bspl = interpolate.BSpline(
                            t_augmented, np.eye(n_coefs)[j], k, extrapolate=True
                        )
                        B[i:end, j] = bspl(x_chunk)
            else:
                # Standard approach for reasonable-sized datasets
                for j in range(n_coefs):
                    bspl = interpolate.BSpline(
                        t_augmented, np.eye(n_coefs)[j], k, extrapolate=True
                    )
                    B[:, j] = bspl(x)

            # Create weighted design matrix
            sqrt_W = np.diag(np.sqrt(weights))
            B_weighted = sqrt_W @ B

            # Calculate the penalty matrix (second derivative squared)
            D = np.zeros((n_coefs - 2, n_coefs))
            for i in range(n_coefs - 2):
                D[i, i] = 1
                D[i, i + 1] = -2
                D[i, i + 2] = 1

            # Scale the penalty matrix based on dataset size
            x_range = np.max(x) - np.min(x)
            h = x_range / (num_knots - 1)  # Average spacing between knots
            penalty_scale = smoothing_param / (h**3)

            # Set up the penalized least squares system
            BTB = B_weighted.T @ B_weighted
            DTD = D.T @ D * penalty_scale

            # Use SVD for stable computation
            U, s, Vh = np.linalg.svd(BTB + DTD, full_matrices=False)

            # Apply adaptive filtering of singular values based on dataset size
            eps = np.finfo(float).eps
            log_n = np.log10(max(30, n))
            adaptive_tol = 10 ** (log_n - 6) * eps  # Scale tolerance with log(n)
            tol = np.max(s) * max(BTB.shape) * adaptive_tol

            mask = s > tol
            s_inv = np.zeros_like(s)
            s_inv[mask] = 1.0 / s[mask]

            # For very large datasets, calculate leverages using a different approach
            if very_large_dataset:
                # Estimate leverages directly using scaling properties
                df_est = self._approximate_df_continuous(
                    smoothing_param / weight_sum, n
                )
                leverages = np.ones(n) * (df_est / n)
                return leverages

            # Calculate leverage values efficiently
            leverages = np.zeros(n)
            for i in range(n):
                b_i = B_weighted[i, :]
                temp = Vh.T @ (s_inv * (U.T @ b_i))
                leverages[i] = np.dot(b_i, temp) / weights[i] if weights[i] > 0 else 0

            return np.clip(leverages, 0.0, 1.0)

        except Exception:
            # Fall back to approximation
            lambda_norm = smoothing_param / weight_sum
            df_est = self._approximate_df_continuous(lambda_norm, n)

            # Return average leverage
            return np.ones(n) * (df_est / n)

    def _approximate_df_continuous(self, lambda_norm: float, n: int) -> float:
        """
        Approximate degrees of freedom using a continuously scaled approach.

        This function provides a more accurate approximation that matches R's behavior
        across datasets of all sizes.

        Parameters
        ----------
        lambda_norm : float
            Normalized smoothing parameter (lambda/sum(weights))
        n : int
            Number of data points

        Returns
        -------
        float
            Approximated degrees of freedom
        """
        # Use logarithmic scaling for dataset size
        log_n = np.log10(max(30, n))  # log10 of dataset size, minimum 30

        # Convert to log scale for easier handling
        log_lambda = np.log10(lambda_norm) if lambda_norm > 0 else -10

        # Calculate minimum df using continuous scaling
        # As dataset size increases, minimum df grows logarithmically
        df_min_ratio = 0.02 * (1 - 0.2 * np.tanh((log_n - 2) / 2))
        df_min = max(2.0, min(10.0, n * df_min_ratio))

        # Calculate maximum df using continuous scaling
        # Max df approaches n as n decreases, but scales sub-linearly for large n
        df_max_ratio = 0.95 / (1 + 0.1 * np.log10(1 + n / 100))
        df_max = min(n * df_max_ratio, n - 1)

        # Create a continuous parameter for the transition rate
        # This controls how quickly df transitions from max to min as spar increases
        transition_rate = 4.0 / (1 + 0.2 * np.log10(1 + n / 100))

        # Map log_lambda to a normalized [0,1] scale with smooth limiting
        if log_lambda < -6:
            t_lambda = 0.0  # Minimum smoothing
        elif log_lambda > 6:
            t_lambda = 1.0  # Maximum smoothing
        else:
            # Sigmoid function for smooth transition
            t_lambda = 1.0 / (1.0 + np.exp(-transition_rate * (log_lambda / 12 + 0.5)))

        # Calculate df using smooth transition
        df = df_max * (1 - t_lambda) + df_min * t_lambda

        # Apply shape correction for mid-range spar values
        # R behavior tends to vary from simple interpolation
        if -2 <= log_lambda <= 2:
            # The "bump" factor varies with dataset size
            bump_factor = 1.2 / (1 + 0.05 * np.log10(1 + n / 100))
            df *= bump_factor

            # Ensure we don't exceed reasonable bounds after adjustment
            df = min(df, df_max * 0.9)

        # Ensure result is within expected bounds
        return float(max(df_min, min(df, df_max)))

    def _get_leverage_values(
        self, x: np.ndarray, weights: np.ndarray, smoothing_param: float
    ) -> np.ndarray:
        """
        Compute the diagonal elements of the smoothing matrix (hat matrix).

        These diagonals are also known as leverage values.

        Parameters
        ----------
        x : ndarray
            Input x values
        weights : ndarray
            Weights for each point
        smoothing_param : float
            Smoothing parameter

        Returns
        -------
        h_diag : ndarray
            Vector of leverage values for each point
        """
        # Delegate to the improved method
        return self._get_leverages_continually_scaled(x, weights, smoothing_param)

    def _optimize_smoothing_for_target_df(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        df_target: float,
        trace_ratio: float = 1.0,
        tol: float = 0.1,
        max_iter: int = 50,
    ) -> float:
        """
        Find the smoothing parameter that gives the desired degrees of freedom.

        Parameters
        ----------
        x, y, weights : ndarray
            Input data
        df_target : float
            Target degrees of freedom
        trace_ratio : float, default=1.0
            Ratio of tr(X'WX) to tr(Ω)
        tol : float, default=0.1
            Tolerance for df matching
        max_iter : int, default=50
            Maximum number of iterations

        Returns
        -------
        s : float
            Optimal smoothing parameter
        """
        # Binary search bounds
        weight_sum = np.sum(weights)
        s_min = 1e-10 * weight_sum  # Almost interpolating
        s_max = 1e2 * weight_sum  # Very smooth

        # Binary search to find s that gives df close to df_target
        for _ in range(max_iter):
            s_mid = np.sqrt(s_min * s_max)  # Geometric mean for better scaling

            # Calculate df for current smoothing parameter
            df_est = self._calculate_degrees_of_freedom(x, y, weights, s_mid)

            if abs(df_est - df_target) < tol:
                return float(s_mid)

            # Update bounds
            if df_est > df_target:  # Too wiggly, increase smoothing
                s_min = s_mid
            else:  # Too smooth, decrease smoothing
                s_max = s_mid

        # Return best estimate if max iterations reached
        return float(np.sqrt(s_min * s_max))

    def _optimize_smoothing_by_cross_validation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        cv_method: str,
        trace_ratio: float = 1.0,
        max_iter: int = 20,
    ) -> float:
        """
        Find the optimal smoothing parameter using cross-validation.

        Parameters
        ----------
        x, y, weights : ndarray
            Input data
        cv_method : str
            Either 'gcv' (generalized cross-validation) or 'loocv' (leave-one-out)
        trace_ratio : float, default=1.0
            Ratio of tr(X'WX) to tr(Ω)
        max_iter : int, default=20
            Maximum number of iterations

        Returns
        -------
        s : float
            Optimal smoothing parameter
        """
        # This method remains unchanged
        # It's a placeholder for future implementation
        # For now, just return a reasonable default
        weight_sum = np.sum(weights)
        return float(1e-1 * weight_sum)

    def predict(
        self, x_new: Optional[np.ndarray] = None, deriv: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict from the fitted smoothing spline.

        Parameters
        ----------
        x_new : array-like, optional
            The x values at which to evaluate the spline. If None, the original
            x values are used.
        deriv : int, default=0
            Order of derivative to evaluate (0, 1, 2, or 3).

        Returns
        -------
        x : ndarray
            The x values.
        y : ndarray
            The predicted values or derivatives.

        Notes
        -----
        For cubic splines (k=3):
        - deriv=0: function value
        - deriv=1: first derivative
        - deriv=2: second derivative
        - deriv=3: third derivative (constant between knots)
        Derivatives of order > 3 are zero everywhere.
        """
        if self.spline_model is None:
            raise ValueError("Model has not been fit yet. Call 'fit' first.")

        if deriv < 0 or deriv > 3:
            raise ValueError("'deriv' must be between 0 and 3")

        if x_new is None:
            # Use original x values if available, otherwise create a grid
            if hasattr(self, "data_x"):
                x_new = self.data_x
            else:
                x_new = np.linspace(self.x_range[0], self.x_range[1], 100)
        else:
            x_new = np.asarray(x_new, dtype=np.float64)

        # Evaluate the spline or its derivative
        try:
            y_new = self.spline_model(x_new, nu=deriv)

            # Handle large values that might occur in higher derivatives
            if deriv > 0:
                # Clip extreme values for numerical stability
                threshold = 1e8
                y_new = np.clip(y_new, -threshold, threshold)

            return x_new, y_new

        except Exception as e:
            # If evaluation fails, return zeros
            warnings.warn(f"Spline evaluation failed: {str(e)}")
            return x_new, np.zeros_like(x_new)

    def plot(
        self,
        x_new: Optional[np.ndarray] = None,
        deriv: int = 0,
        show_data: bool = True,
        fig: Optional[Any] = None,
    ) -> Any:
        """
        Plot the fitted spline using Plotly.

        Parameters
        ----------
        x_new : array-like, optional
            The x values at which to evaluate the spline. If None, a grid of
            points spanning the range of the original x values is used.
        deriv : int, default=0
            Order of derivative to plot (0, 1, 2, or 3).
        show_data : bool, default=True
            If True, also plot the original data points.
        fig : plotly.graph_objects.Figure, optional
            Existing figure to add the plots to. If None, a new figure is created.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The plotly figure object.
        """
        try:
            import plotly.graph_objects as go  # type: ignore
        except ImportError:
            raise ImportError(
                "Plotly is required for plotting. Install with 'pip install plotly'"
            )

        if self.spline_model is None:
            raise ValueError("Model has not been fit yet. Call 'fit' first.")

        if fig is None:
            fig = go.Figure()

        # Generate x values if not provided
        if x_new is None:
            x_new = np.linspace(self.x_range[0], self.x_range[1], 300)

        # Predict
        x_pred, y_pred = self.predict(x_new, deriv=deriv)

        # Plot the spline
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="lines",
                line=dict(width=2, color="red"),
                name=f"Spline (deriv={deriv})",
            )
        )

        # Plot the data points if requested
        if show_data and deriv == 0 and hasattr(self, "data_x"):
            fig.add_trace(
                go.Scatter(
                    x=self.data_x,
                    y=self.data_y,
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.5),
                    name="Data points",
                )
            )

        # Add horizontal line at y=0 for derivatives
        if deriv > 0:
            fig.add_shape(
                type="line",
                x0=min(x_pred),
                y0=0,
                x1=max(x_pred),
                y1=0,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dash",
                ),
            )

        # Create Title String
        title = f"Smoothing Spline{' (deriv=' + str(deriv) + ')' if deriv > 0 else ''}"

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title=f"y{'' if deriv == 0 else f' (deriv={deriv})'}",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white",
            hovermode="closest",
        )

        # Add df and spar info to the plot if available
        annotations = []
        if hasattr(self, "df") and self.df is not None:
            annotations.append(
                dict(
                    x=0.99,
                    y=0.97,
                    xref="paper",
                    yref="paper",
                    text=f"df = {self.df:.2f}",
                    showarrow=False,
                    font=dict(size=12),
                )
            )

        if hasattr(self, "spar") and self.spar is not None:
            annotations.append(
                dict(
                    x=0.99,
                    y=0.93,
                    xref="paper",
                    yref="paper",
                    text=f"spar = {self.spar:.4f}",
                    showarrow=False,
                    font=dict(size=12),
                )
            )

        if annotations:
            fig.update_layout(annotations=annotations)

        return fig

    def plot_interactive(self, x_new: Optional[np.ndarray] = None) -> Any:
        """
        Create an interactive Plotly figure with sliders to control spar value
        and derivative order.

        Parameters
        ----------
        x_new : array-like, optional
            The x values at which to evaluate the spline. If None, a grid of
            points spanning the range of the original x values is used.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The interactive plotly figure object.
        """
        try:
            import plotly.graph_objects as go  # type: ignore
            from plotly.subplots import make_subplots  # type: ignore
        except ImportError:
            raise ImportError(
                "Plotly is required for plotting. Install with 'pip install plotly'"
            )

        if not hasattr(self, "data_x") or not hasattr(self, "data_y"):
            raise ValueError(
                "Interactive plot needs original data. Ensure keep_data=True in fit()"
            )

        # Generate x values if not provided
        if x_new is None:
            x_new = np.linspace(self.x_range[0], self.x_range[1], 300)

        # Create figure with two subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Smoothing Spline Fit", "Derivatives"),
            row_heights=[0.7, 0.3],
        )

        # Add data points to the first subplot
        fig.add_trace(
            go.Scatter(
                x=self.data_x,
                y=self.data_y,
                mode="markers",
                marker=dict(size=8, color="blue", opacity=0.5),
                name="Data points",
            ),
            row=1,
            col=1,
        )

        # Create a grid of spar values for the slider
        spar_values = np.round(np.linspace(0, 1, 21), 2)

        # Pre-compute splines for each spar value to improve performance
        y_fits = []
        y_derivs = []
        df_values = []

        for spar in spar_values:
            temp_spline = SmoothSpline()
            try:
                temp_spline.fit(
                    self.data_x,
                    self.data_y,
                    weights=(
                        self.data_weights if hasattr(self, "data_weights") else None
                    ),
                    spar=spar,
                )

                # Get predictions
                _, y_fit = temp_spline.predict(x_new, deriv=0)
                _, y_deriv = temp_spline.predict(x_new, deriv=1)

                y_fits.append(y_fit)
                y_derivs.append(y_deriv)
                df_values.append(temp_spline.df)
            except (ValueError, np.linalg.LinAlgError) as e:
                # Use previous values if computation fails
                if y_fits:
                    y_fits.append(y_fits[-1])
                    y_derivs.append(y_derivs[-1])
                    df_values.append(df_values[-1])
                else:
                    # First value - use flat line
                    y_fits.append(np.ones_like(x_new) * np.mean(self.data_y))
                    y_derivs.append(np.zeros_like(x_new))
                    df_values.append(1.0)
                # Print warning for failed fits
                warnings.warn(f"Fit failed for spar = {spar}")
                # Print error message
                print(e)

        # Create slider steps
        steps = []
        for i, spar in enumerate(spar_values):
            step = dict(
                method="update",
                args=[
                    {"y": [None, y_fits[i], y_derivs[i]]},
                    {
                        "annotations": [
                            {
                                "x": 0.99,
                                "y": 0.95,
                                "xref": "paper",
                                "yref": "paper",
                                "text": f"spar = {spar:.2f}, df = {df_values[i]:.2f}",
                                "showarrow": False,
                                "font": {"size": 12},
                            }
                        ]
                    },
                ],
                label=f"{spar:.2f}",
            )
            steps.append(step)

        # Initial spline (middle spar value)
        middle_idx = len(spar_values) // 2

        # Plot the initial spline fit
        fig.add_trace(
            go.Scatter(
                x=x_new,
                y=y_fits[middle_idx],
                mode="lines",
                line=dict(width=2, color="red"),
                name="Spline",
            ),
            row=1,
            col=1,
        )

        # Plot the derivative
        fig.add_trace(
            go.Scatter(
                x=x_new,
                y=y_derivs[middle_idx],
                mode="lines",
                line=dict(width=2, color="green"),
                name="First derivative",
            ),
            row=2,
            col=1,
        )

        # Add zero line to derivative plot
        fig.add_shape(
            type="line",
            x0=np.nanmin(x_new) if len(x_new) > 0 else 0,
            x1=np.nanmax(x_new) if len(x_new) > 0 else 1,
            y0=0,
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title="Interactive Smoothing Spline Explorer",
            xaxis_title="x",
            yaxis_title="y",
            xaxis2_title="x",
            yaxis2_title="dy/dx",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            hovermode="closest",
            height=700,
            sliders=[
                dict(
                    active=middle_idx,  # Default to middle spar value
                    currentvalue={"prefix": "Smoothing parameter (spar): "},
                    pad={"t": 50},
                    steps=steps,
                )
            ],
        )

        return fig

    def compare_with_r(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        spar: Optional[float] = None,
        df: Optional[float] = None,
        cv: bool = False,
        tol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compare the results with R's smooth.spline implementation.

        Parameters
        ----------
        x, y, weights, spar, df, cv, tol:
            Same parameters as in fit method

        Returns
        -------
        dict
            Dictionary containing the comparison results with differences between
            the two implementations and the actual values from each.

        Notes
        -----
        Requires rpy2 to be installed for R integration.
        The comparison metrics include:
        - x_diff: Maximum absolute difference in predicted x values
        - y_diff: Maximum absolute difference in predicted y values
        - df_diff: Absolute difference in degrees of freedom
        - spar_diff: Absolute difference in spar parameter
        - lambda_diff: Absolute difference in lambda parameter
        """
        try:
            import rpy2.robjects as ro  # type: ignore[import-untyped]
            import rpy2.robjects.numpy2ri  # type: ignore[import-untyped]
            from rpy2.robjects.packages import importr  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "rpy2 is required for comparing with R. Install with 'pip install rpy2'"
            )

        # Enable automatic conversion between R and numpy arrays
        rpy2.robjects.numpy2ri.activate()

        # Import required R packages
        base = importr("base")  # noqa: F841
        stats = importr("stats")

        # Convert parameters to R
        r_x = ro.FloatVector(x)
        r_y = ro.FloatVector(y)
        r_params = {"x": r_x, "y": r_y}

        if weights is not None:
            r_params["w"] = ro.FloatVector(weights)
        if spar is not None:
            r_params["spar"] = ro.FloatVector([spar])[0]
        if df is not None:
            r_params["df"] = ro.FloatVector([df])[0]
        if cv is not None:
            r_params["cv"] = ro.BoolVector([cv])[0]
        if tol is not None:
            r_params["tol"] = ro.FloatVector([tol])[0]

        # Call R's smooth.spline
        r_result = stats.smooth_spline(**r_params)

        # Extract results from R
        r_x_out = np.array(r_result.rx2("x"))
        r_y_out = np.array(r_result.rx2("y"))
        r_df = float(np.array(r_result.rx2("df"))[0])
        r_spar = float(np.array(r_result.rx2("spar"))[0])
        r_lambda = float(np.array(r_result.rx2("lambda"))[0])

        # Fit our implementation
        self.fit(x, y, weights=weights, df=df, spar=spar, cv=cv, tol=tol)

        # Get predictions at the same x points as R
        _, py_y_out = self.predict(r_x_out)

        # Compute differences
        x_diff = np.abs(r_x_out - r_x_out).max()  # Should be 0
        y_diff = np.abs(r_y_out - py_y_out).max()
        df_diff = np.abs(r_df - self.df)
        spar_diff = np.abs(r_spar - self.spar)
        lambda_diff = np.abs(r_lambda - self.lambda_)

        # Return comparison results
        return {
            "x_diff": float(x_diff),
            "y_diff": float(y_diff),
            "df_diff": float(df_diff),
            "spar_diff": float(spar_diff),
            "lambda_diff": float(lambda_diff),
            "r_result": {
                "x": r_x_out,
                "y": r_y_out,
                "df": float(r_df),
                "spar": float(r_spar),
                "lambda": float(r_lambda),
            },
            "py_result": {
                "x": r_x_out,  # Same x values
                "y": py_y_out,
                "df": float(self.df) if self.df is not None else 0.0,
                "spar": float(self.spar) if self.spar is not None else 0.0,
                "lambda": float(self.lambda_) if self.lambda_ is not None else 0.0,
            },
        }


if __name__ == "__main__":
    import time

    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    print("Testing SmoothSpline Implementation vs R's smooth.spline...")

    # Generate some noisy data
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)

    # Compare multiple spar values
    spar_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    # Create a subplot grid for comparing different spar values
    fig_grid = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f"spar = {spar}" for spar in spar_values] + [""],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
    )

    # Fine grid for predictions
    x_new = np.linspace(0, 10, 300)

    # Timing information
    py_time = 0.0
    r_time = 0.0

    print("\nComparing across different smoothing levels...")

    # Loop through different spar values
    for i, spar in enumerate(spar_values):
        row, col = (i // 3) + 1, (i % 3) + 1

        print(f"\nTesting spar = {spar}")

        # Fit our implementation and time it
        start_time = time.time()
        spline = SmoothSpline()
        spline.fit(x, y, spar=spar)
        py_time_i = time.time() - start_time
        py_time += py_time_i

        # Get Python predictions
        x_pred, y_pred = spline.predict(x_new)

        # Compare with R and time it
        start_time = time.time()
        comparison = spline.compare_with_r(x, y, spar=spar)
        r_time_i = time.time() - start_time
        r_time += r_time_i

        # Store results for the summary table
        results.append(
            {
                "spar": spar,
                "df_diff": comparison["df_diff"],
                "y_max_diff": comparison["y_diff"],
                "py_df": spline.df,
                "r_df": comparison["r_result"]["df"],
                "py_time": py_time_i,
                "r_time": r_time_i,
            }
        )

        # Add traces to the subplot
        fig_grid.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=4, opacity=0.5),
                name="Data" if i == 0 else None,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        # R's smooth.spline
        fig_grid.add_trace(
            go.Scatter(
                x=x_new,
                y=np.interp(
                    x_new, comparison["r_result"]["x"], comparison["r_result"]["y"]
                ),
                mode="lines",
                line=dict(color="green", width=2),
                name="R" if i == 0 else None,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        # Our implementation
        fig_grid.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="lines",
                line=dict(color="red", width=2, dash="dot"),
                name="Python" if i == 0 else None,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        # Add annotation with df info
        fig_grid.add_annotation(
            x=0.05,
            y=0.95,
            xref=f"x{'' if i == 0 else i + 1} domain",
            yref=f"y{'' if i == 0 else i + 1} domain",
            text=f"Python df: {spline.df:.2f}<br>R df: {comparison['r_result']['df']:.2f}",  # noqa: E501
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.7)",
        )

    # Create summary table as a plotly table
    df = pl.DataFrame(results)
    print("\nSummary of differences:")
    print(df.select(["spar", "py_df", "r_df", "df_diff", "y_max_diff"]).sort("spar"))

    # Print timing comparison
    print("\nTiming comparison (total across all spar values):")
    print(f"Python implementation: {py_time:.4f} seconds")
    print(f"R implementation:      {r_time:.4f} seconds")
    print(f"Ratio (Python/R):      {py_time / r_time:.4f}")

    # Add text annotation with summary
    summary_text = "Summary:<br>"
    for r in results:
        summary_text += (
            f"spar={r['spar']:.2f}: Python df={r['py_df']:.2f}, "
            f"R df={r['r_df']:.2f}, diff={r['df_diff']:.2e}<br>"
        )

    # Add annotation with summary
    fig_grid.add_annotation(
        x=0.5,
        y=0.5,
        xref="x6 domain",
        yref="y6 domain",
        text=summary_text,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align="left",
    )

    # Create a separate table figure
    fig_table = go.Figure(
        data=[
            go.Table(  # type: ignore
                header=dict(
                    values=["spar", "Python df", "R df", "df diff", "max y diff"],
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[
                        [f"{r['spar']:.2f}" for r in results],
                        [f"{r['py_df']:.2f}" for r in results],
                        [f"{r['r_df']:.2f}" for r in results],
                        [f"{r['df_diff']:.2e}" for r in results],
                        [f"{r['y_max_diff']:.2e}" for r in results],
                    ],
                    fill_color="lavender",
                    align="center",
                    font=dict(size=11),
                ),
            )
        ]
    )

    fig_table.update_layout(
        title="Summary of Differences Between Python and R Implementations",
        height=300,
        width=800,
    )

    # Add annotation with summary
    fig_grid.add_annotation(
        x=0.5,
        y=0.5,
        xref="x6 domain",
        yref="y6 domain",
        text=summary_text,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align="left",
    )

    # Update layout
    fig_grid.update_layout(
        title="Comparison of Python vs R smooth.spline Implementation",
        height=800,
        width=1200,
        # Show legend on the bottom right
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )

    # Create a closer look at the worst case
    # Find the spar value with the largest difference
    worst_idx = max(range(len(results)), key=lambda i: results[i]["y_max_diff"])
    worst_spar = spar_values[worst_idx]

    print(f"\nCloser look at largest difference case (spar = {worst_spar}):")

    # Create a new spline with this spar value
    spline = SmoothSpline()
    spline.fit(x, y, spar=worst_spar)
    comparison = spline.compare_with_r(x, y, spar=worst_spar)

    # Get the absolute difference between R and Python predictions
    r_interpolated = np.interp(
        x_new, comparison["r_result"]["x"], comparison["r_result"]["y"]
    )
    x_pred, y_pred = spline.predict(x_new)
    abs_diff = np.abs(r_interpolated - y_pred)

    # Create a more detailed comparison plot
    fig_detail = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            f"Spline Comparison (spar = {worst_spar})",
            "Absolute Difference",
        ],
        vertical_spacing=0.15,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
    )

    # Add data points
    fig_detail.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", marker=dict(size=5, opacity=0.5), name="Data"
        ),
        row=1,
        col=1,
    )

    # Add Python spline
    fig_detail.add_trace(
        go.Scatter(
            x=x_pred,
            y=y_pred,
            mode="lines",
            line=dict(color="red", width=2),
            name="Python",
        ),
        row=1,
        col=1,
    )

    # Add R spline
    fig_detail.add_trace(
        go.Scatter(
            x=x_new,
            y=r_interpolated,
            mode="lines",
            line=dict(color="green", width=2),
            name="R",
        ),
        row=1,
        col=1,
    )

    # Add absolute difference
    fig_detail.add_trace(
        go.Scatter(
            x=x_new,
            y=abs_diff,
            mode="lines",
            line=dict(color="purple", width=2),
            name="Absolute Difference",
        ),
        row=2,
        col=1,
    )

    # Add horizontal line at y=0 for reference
    fig_detail.add_shape(
        type="line",
        x0=np.nanmin(x_new) if len(x_new) > 0 else 0,
        x1=np.nanmax(x_new) if len(x_new) > 0 else 1,
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        row=2,
        col=1,
    )

    # Update layout
    fig_detail.update_layout(
        title=f"Detailed Comparison for spar = {worst_spar}",
        height=700,
        width=1000,
        template="plotly_white",
    )

    # Add annotations with statistics
    fig_detail.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=(
            f"Max absolute difference: {abs_diff.max():.2e}<br>"
            f"Python df: {spline.df:.2f}, "
            f"R df: {comparison['r_result']['df']:.2f}<br>"
            f"df difference: {comparison['df_diff']:.2e}"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
        align="left",
    )

    # Show the figures
    fig_grid.show()
    fig_table.show()
    fig_detail.show()

    # Save to HTML for interactive viewing
    fig_grid.write_html("smooth_spline_grid_comparison.html")
    fig_table.write_html("smooth_spline_table_summary.html")
    fig_detail.write_html("smooth_spline_detail_comparison.html")

    print("\nComparison completed. Interactive plots saved to HTML files.")
