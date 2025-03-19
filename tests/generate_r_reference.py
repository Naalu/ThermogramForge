#!/usr/bin/env python3
"""
Generate reference data from R's smooth.spline for validation testing.

This script creates reference data files comparing outputs from R's smooth.spline
with the Python implementation for verification and testing.
"""

from pathlib import Path

import numpy as np
import polars as pl

# Try to import rpy2 for R integration
try:
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr

    # Activate numpy to R conversion
    rpy2.robjects.numpy2ri.activate()

    # Import stats package which contains smooth.spline
    stats = importr("stats")

    HAS_R = True
except ImportError:
    HAS_R = False
    print("Warning: rpy2 not available. Cannot generate R reference data.")


def generate_test_pattern(pattern, n=100, noise=0.1):
    """Generate test data with different patterns."""
    x = np.linspace(0, 10, n)

    if pattern == "sine":
        y = np.sin(x) + noise * np.random.randn(n)
    elif pattern == "exp":
        y = np.exp(x / 5) + noise * np.random.randn(n)
    elif pattern == "peaks":
        # Create thermogram-like pattern with multiple peaks
        y = (
            0.3 * np.exp(-0.5 * ((x - 2) / 0.5) ** 2)
            + 0.5 * np.exp(-0.5 * ((x - 5) / 0.8) ** 2)
            + 0.2 * np.exp(-0.5 * ((x - 8) / 0.6) ** 2)
            + noise * np.random.randn(n)
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return x, y


def generate_r_reference(pattern, output_dir, n=100, noise=0.1):
    """Generate reference data from R's smooth.spline."""
    if not HAS_R:
        print(f"Skipping {pattern} (R not available)")
        return

    print(f"Generating reference data for {pattern} pattern...")

    # Generate test data
    x, y = generate_test_pattern(pattern, n, noise)

    # Create R vectors
    r_x = ro.FloatVector(x)
    r_y = ro.FloatVector(y)

    # Call R's smooth.spline with cross-validation
    r_spline = stats.smooth_spline(x=r_x, y=r_y, cv=True)

    # Extract results
    r_fitted = np.array(r_spline.rx2("y"))
    r_spar = float(r_spline.rx2("spar")[0])
    r_df = float(r_spline.rx2("df")[0])
    r_lambda = float(r_spline.rx2("lambda")[0])

    # Save input data
    input_df = pl.DataFrame({"x": x, "y": y})
    input_df.write_csv(output_dir / f"{pattern}_input.csv")

    # Save fitted values
    fitted_df = pl.DataFrame({"x": x, "fitted": r_fitted})
    fitted_df.write_csv(output_dir / f"{pattern}_fitted.csv")

    # Save parameters
    params_df = pl.DataFrame({"spar": [r_spar], "df": [r_df], "lambda": [r_lambda]})
    params_df.write_csv(output_dir / f"{pattern}_params.csv")

    print(f"  - Created reference data with spar={r_spar:.4f}, df={r_df:.2f}")


def main():
    """Main function to generate all reference data."""
    # Create output directory
    output_dir = Path("tests/data/r_reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate reference data for different patterns
    for pattern in ["sine", "exp", "peaks"]:
        generate_r_reference(pattern, output_dir)

    # Generate data with different sizes
    generate_r_reference("sine", output_dir, n=20, noise=0.05)

    print("Reference data generation complete")


if __name__ == "__main__":
    main()
