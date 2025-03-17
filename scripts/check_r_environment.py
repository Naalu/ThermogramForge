#!/usr/bin/env python3
"""
Utility to check if the R environment is correctly configured for ThermogramForge.

This script verifies:
1. R is installed and accessible
2. rpy2 is installed and working
3. Required R packages are available
"""

import subprocess
import sys


def check_r_installation() -> bool:
    """Check if R is installed and accessible."""
    print("Checking R installation...")
    try:
        r_version = subprocess.run(
            ["R", "--version"], capture_output=True, text=True, check=True
        )
        version_str = r_version.stdout.split("\n")[0]
        print(f"✅ R is installed: {version_str}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ R is not installed or not in PATH")
        print("   Please install R from https://cran.r-project.org/")
        return False


def check_r_packages() -> bool:
    """Check if required R packages are installed."""
    print("\nChecking required R packages...")
    required_packages = ["stats"]  # Add more as needed

    try:
        # Create a temporary R script to check packages
        script = "installed <- rownames(installed.packages())\n"
        for pkg in required_packages:
            script += f'cat("{pkg}:", "{pkg}" %in% installed, "\\n")\n'

        # Run the script
        result = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, check=True
        )

        # Parse the output
        all_installed = True
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines:
            if ":" in line:
                pkg, status = line.split(":", 1)
                is_installed = "TRUE" in status
                if is_installed:
                    print(f"✅ Package {pkg} is installed")
                else:
                    print(f"❌ Package {pkg} is NOT installed")
                    all_installed = False

        return all_installed
    except subprocess.SubprocessError:
        print("❌ Failed to check R packages")
        return False


def check_rpy2_installation() -> bool:
    """Check if rpy2 is installed and working."""
    print("\nChecking rpy2 installation...")
    try:
        import rpy2  # type: ignore
        import rpy2.robjects  # type: ignore

        print(f"✅ rpy2 is installed (version {rpy2.__version__})")

        # Try to create an R session
        r = rpy2.robjects.r
        r_version = r["R.version.string"]  # type: ignore
        print(f"✅ R session created via rpy2: {r_version[0]}")

        return True
    except ImportError:
        print("❌ rpy2 is not installed")
        print("   Install with: pip install rpy2")
        return False
    except Exception as e:
        print(f"❌ rpy2 is installed but encountered an error: {str(e)}")
        return False


def check_integration_example() -> bool:
    """Run a simple integration example to verify everything works."""
    print("\nRunning a simple integration test...")
    try:
        import numpy as np
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri  # type: ignore
        from rpy2.robjects.packages import importr  # type: ignore

        # Activate conversion
        rpy2.robjects.numpy2ri.activate()

        # Import stats
        stats = importr("stats")

        # Create simple test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)

        # Convert to R vectors
        r_x = robjects.FloatVector(x)
        r_y = robjects.FloatVector(y)

        # Call smooth.spline
        r_spline = stats.smooth_spline(x=r_x, y=r_y, cv=True)

        # Extract fitted values
        r_fitted = np.array(r_spline.rx2("y"))

        print("✅ Successfully ran R smooth.spline via rpy2 on test data")
        print(f"   Fitted {len(r_fitted)} points with spline")

        return True
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False


def main() -> None:
    """Run all checks and report overall status."""
    print("=" * 60)
    print("ThermogramForge R Environment Check")
    print("=" * 60)

    r_ok = check_r_installation()
    if not r_ok:
        print("\n❌ R installation check failed - fix this first")
        sys.exit(1)

    packages_ok = check_r_packages()
    if not packages_ok:
        print("\n⚠️ Some required R packages are missing")
        print("   You can install them in R with: install.packages(c('package_name'))")

    rpy2_ok = check_rpy2_installation()
    if not rpy2_ok:
        print("\n❌ rpy2 installation check failed")
        print("   Install with: pip install rpy2")
        sys.exit(1)

    if r_ok and rpy2_ok:
        integration_ok = check_integration_example()
        if not integration_ok:
            print("\n❌ Integration test failed - see error above")
            sys.exit(1)

    print("\n" + "=" * 60)
    if r_ok and packages_ok and rpy2_ok and integration_ok:
        print("✅ All checks passed. R environment is correctly configured.")
        print("   You can use the R integration features of ThermogramForge.")
    else:
        print("⚠️ Some checks failed. R integration may not work correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
