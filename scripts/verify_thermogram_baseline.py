"""
Verification script for thermogram_baseline package.

This script compares the results of the Python implementation against the R implementation
for baseline subtraction and generates an HTML report with visualizations that closely
match the original R workflow demonstrated in the ThermogramBaseline R package.

Usage:
    python verify_thermogram_baseline.py [options]

Options:
    --python-data PATH   Path to Python-processed data
    --r-data PATH        Path to R-processed data
    --raw-data PATH      Path to raw thermogram data
    --output PATH        Output HTML report path
    --r-sample NAME      Sample column to use from R data
    --verbose            Print verbose output
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import jinja2
import numpy as np
import plotly.graph_objects as go  # type: ignore
import polars as pl
from plotly.subplots import make_subplots  # type: ignore
from rich.console import Console  # type: ignore
from rich.progress import Progress, SpinnerColumn, TextColumn  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

# Add the project root to Python path to enable imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from thermogram_baseline import (
    ThermogramData,
    detect_endpoints,
    interpolate_sample,
    subtract_baseline,
)

# Initialize rich console
console = Console()

# Default file paths
TEMPLATE_DIR = PROJECT_ROOT / "scripts" / "templates"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DEFAULT = DATA_DIR / "raw" / "example_thermogram.csv"
R_DATA_DEFAULT = DATA_DIR / "reference" / "r_processed.csv"
PYTHON_DATA_DEFAULT = DATA_DIR / "processed" / "python_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "verification_results"
OUTPUT_DEFAULT = OUTPUT_DIR / "verification_report.html"


# Helper functions
def _safe_interpolate(x, y, target_x):
    """Safely interpolate data, handling edge cases.

    Args:
        x: Source x values
        y: Source y values
        target_x: Target x values for interpolation

    Returns:
        np.ndarray: Interpolated y values
    """
    try:
        # Ensure x is strictly increasing (requirement for interp1d)
        if not np.all(np.diff(x) > 0):
            # Sort values if not already in order
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]

        # Restrict target_x to the valid range of x
        valid_range = (target_x >= np.min(x)) & (target_x <= np.max(x))
        valid_target_x = target_x[valid_range]

        # Perform interpolation
        interpolator = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
        result = interpolator(valid_target_x)

        # Create full result array initialized with NaN
        full_result = np.full_like(target_x, np.nan, dtype=float)

        # Fill in valid values
        full_result[valid_range] = result

        return full_result
    except Exception as e:
        console.print(f"[red]Error during interpolation: {str(e)}[/]")
        # Return array of NaNs with same shape as target_x
        return np.full_like(target_x, np.nan, dtype=float)


def _plot_endpoints(
    fig,
    temperatures,
    values,
    lower_temp,
    upper_temp,
    color="red",
    size=10,
    name="Endpoints",
):
    """Helper function to add endpoint markers to a plot.

    Args:
        fig: Plotly figure
        temperatures: Array of temperature values
        values: Array of dCp values
        lower_temp: Lower temperature endpoint
        upper_temp: Upper temperature endpoint
        color: Color for markers
        size: Size of markers
        name: Name for markers in legend

    Returns:
        fig: Updated Plotly figure
    """
    try:
        # Convert numpy arrays to standard Python lists if needed
        if isinstance(temperatures, np.ndarray):
            temperatures = temperatures.tolist()
        if isinstance(values, np.ndarray):
            values = values.tolist()

        # Find the dCp values at the endpoints - using Python int conversion to handle numpy types
        lower_idx = int(np.abs(np.array(temperatures) - lower_temp).argmin())
        upper_idx = int(np.abs(np.array(temperatures) - upper_temp).argmin())

        # Ensure indices are valid
        if 0 <= lower_idx < len(values) and 0 <= upper_idx < len(values):
            lower_dcp = values[lower_idx]
            upper_dcp = values[upper_idx]

            fig.add_trace(
                go.Scatter(
                    x=[lower_temp, upper_temp],
                    y=[lower_dcp, upper_dcp],
                    mode="markers",
                    name=name,
                    marker=dict(size=size, color=color),
                )
            )
    except Exception as e:
        console.print(f"[yellow]Warning: Could not add endpoint markers: {str(e)}[/]")

    return fig


# Data format detection and conversion
def detect_data_format(df):
    """Detect the format of the data (ThermogramBaseline or tlbparam format).

    Args:
        df (pl.DataFrame): DataFrame to analyze

    Returns:
        str: Detected format, either "thermogram_baseline", "tlbparam", or "r_multi_sample"

    Raises:
        ValueError: If the data format can't be determined
    """
    # Check for standard thermogram_baseline format
    if "Temperature" in df.columns and "dCp" in df.columns:
        return "thermogram_baseline"

    # Check for R output with Temperature + multiple sample columns
    if "Temperature" in df.columns and len(df.columns) > 1:
        # If there's a Temperature column and at least one other column,
        # assume it's multi-sample R output
        return "r_multi_sample"

    # Check for temperature columns like "T45", "T45.1", etc.
    temp_cols = [
        col
        for col in df.columns
        if col.startswith("T") and col[1:].replace(".", "", 1).isdigit()
    ]
    if temp_cols:
        return "tlbparam"

    raise ValueError(
        "Unknown data format. Expected 'Temperature'/'dCp' columns, multiple sample columns, or 'T{number}' columns"
    )


def convert_tlbparam_format(df):
    """Convert tlbparam format to thermogram_baseline format.

    Args:
        df (pl.DataFrame): DataFrame in tlbparam format (with T{number} columns)

    Returns:
        pl.DataFrame: Converted DataFrame with Temperature and dCp columns
    """
    # Find temperature columns (starting with T)
    temp_cols = [
        col
        for col in df.columns
        if col.startswith("T") and col[1:].replace(".", "", 1).isdigit()
    ]

    if not temp_cols:
        raise ValueError("No valid temperature columns found in tlbparam format")

    # Sort them by temperature
    temp_cols.sort(key=lambda x: float(x[1:]))

    # If SampleCode or similar column exists, use one sample, otherwise use first row
    if "SampleCode" in df.columns:
        sample_col = "SampleCode"
        sample_id = df[sample_col][0]
        sample_data = df.filter(pl.col(sample_col) == sample_id)
    else:
        sample_data = df.head(1)

    # Extract temperatures and values for the first sample
    try:
        temperatures = [float(col[1:]) for col in temp_cols]
        dcp_values = sample_data.select(temp_cols).to_numpy()[0]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error extracting data from tlbparam format: {str(e)}")

    # Create new DataFrame in thermogram_baseline format
    new_df = pl.DataFrame({"Temperature": temperatures, "dCp": dcp_values})

    return new_df


def convert_r_multi_sample_format(df, sample_column=None):
    """Convert R multi-sample format to thermogram_baseline format.

    Args:
        df (pl.DataFrame): DataFrame in R format with Temperature and multiple sample columns
        sample_column (str, optional): Specific sample column to use. If None, uses the first available.

    Returns:
        pl.DataFrame: Converted DataFrame with Temperature and dCp columns

    Raises:
        ValueError: If no sample columns are found or data extraction fails
    """
    if df is None or df.is_empty():
        raise ValueError("Empty or null DataFrame provided")

    # Get all non-Temperature columns (these are sample columns)
    sample_columns = [col for col in df.columns if col != "Temperature"]

    if not sample_columns:
        raise ValueError("No sample columns found in R data format")

    # Choose which sample column to use
    selected_column = None
    if sample_column is not None:
        if sample_column in sample_columns:
            selected_column = sample_column
        else:
            console.print(
                f"[yellow]Warning: Requested sample column '{sample_column}' not found. Available columns: {', '.join(sample_columns)}[/]"
            )
            console.print(f"[yellow]Using '{sample_columns[0]}' instead.[/]")
            selected_column = sample_columns[0]
    else:
        selected_column = sample_columns[0]
        console.print(f"[blue]Using sample column '{selected_column}' from R data[/]")

    try:
        # Create new DataFrame with Temperature and dCp columns
        new_df = pl.DataFrame(
            {"Temperature": df["Temperature"], "dCp": df[selected_column]}
        )

        # Verify data validity
        if new_df.is_empty() or new_df.null_count().sum() > 0:
            console.print(
                "[yellow]Warning: Converted data contains empty or null values[/]"
            )

        return new_df

    except Exception as e:
        raise ValueError(f"Failed to extract data from R format: {str(e)}")


# Setup functions
def ensure_directories_exist():
    """Create all necessary directories if they don't exist.

    Creates the following directory structure:
        - data/
            - raw/
            - reference/
            - processed/
        - verification_results/
        - scripts/templates/
    """
    directories = [
        DATA_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "reference",
        DATA_DIR / "processed",
        OUTPUT_DIR,
        TEMPLATE_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments containing:
            python_data (str): Path to Python-processed data
            r_data (str): Path to R-processed data
            raw_data (str): Path to raw thermogram data
            output (str): Output HTML report path
            verbose (bool): Whether to print verbose output
            r_sample (str): Specific sample column to use from R data
    """
    parser = argparse.ArgumentParser(
        description="Verify thermogram_baseline against R implementation"
    )
    parser.add_argument(
        "--python-data",
        type=str,
        default=str(PYTHON_DATA_DEFAULT),
        help=f"Path to Python-processed data (CSV), default: {PYTHON_DATA_DEFAULT}",
    )
    parser.add_argument(
        "--r-data",
        type=str,
        default=str(R_DATA_DEFAULT),
        help=f"Path to R-processed data (CSV), default: {R_DATA_DEFAULT}",
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        default=str(RAW_DATA_DEFAULT),
        help=f"Path to raw thermogram data (CSV), default: {RAW_DATA_DEFAULT}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DEFAULT),
        help=f"Output HTML report path, default: {OUTPUT_DEFAULT}",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--r-sample",
        type=str,
        default=None,
        help="Specific sample column to use from R data (if multiple samples are present)",
    )
    return parser.parse_args()


# Processing functions
def load_data(python_path, r_path, raw_path, verbose=False, sample_column=None):
    """Load data from Python, R implementations and raw data.

    Args:
        python_path (str): Path to Python-processed data
        r_path (str): Path to R-processed data
        raw_path (str): Path to raw thermogram data
        verbose (bool): Whether to print verbose output
        sample_column (str, optional): Specific sample column to use from R data

    Returns:
        tuple: Contains:
            - Python processed data (pl.DataFrame or None)
            - R processed data (pl.DataFrame or None)
            - Raw thermogram data (pl.DataFrame or None)
    """
    data = {}

    # Load and check existence of files
    for name, path in [("python", python_path), ("r", r_path), ("raw", raw_path)]:
        if path and os.path.exists(path):
            try:
                with console.status(
                    f"Loading {name} data from {path}...", spinner="dots"
                ):
                    df = pl.read_csv(path)

                # Detect format and convert if needed
                format_type = detect_data_format(df)
                if verbose:
                    console.print(
                        f"Detected {name} data format: [blue]{format_type}[/]"
                    )

                if format_type == "tlbparam":
                    df = convert_tlbparam_format(df)
                    if verbose:
                        console.print(f"Converted {name} data from tlbparam format")

                elif format_type == "r_multi_sample":
                    df = convert_r_multi_sample_format(df, sample_column)
                    if verbose:
                        console.print(
                            f"Converted {name} data from R multi-sample format"
                        )

                data[name] = df
                if verbose:
                    console.print(
                        f"Successfully loaded {name} data from {path}", style="green"
                    )
            except Exception as e:
                console.print(
                    f"Error loading {name} data from {path}: {str(e)}", style="red"
                )
                data[name] = None
        else:
            if verbose and path:
                console.print(f"File not found: {path}", style="yellow")
            data[name] = None

    return data.get("python"), data.get("r"), data.get("raw")


def process_sample(raw_data, sample_id=None, verbose=False):
    """Process a sample using the Python implementation, capturing all stages.

    Args:
        raw_data (pl.DataFrame or ThermogramData): Raw thermogram data
        sample_id (str, optional): Sample identifier
        verbose (bool): Whether to print verbose output

    Returns:
        dict: Dictionary containing processing results at each stage:
            - endpoints: Detected endpoints
            - baseline_result: Result of baseline subtraction
            - interpolated_result: Final interpolated result
    """
    # Convert to ThermogramData if needed
    if isinstance(raw_data, pl.DataFrame):
        data = ThermogramData.from_dataframe(raw_data)
    else:
        data = raw_data

    with console.status(
        "[bold green]Processing thermogram...", spinner="dots"
    ) as status:
        if verbose:
            status.update("[bold green]Step 1: Detecting endpoints...")

        # 1. Detect endpoints
        endpoints = detect_endpoints(
            data=data,
            window_size=90,
            exclusion_lower=60.0,
            exclusion_upper=80.0,
            point_selection="innermost",
            verbose=verbose,
        )

        if verbose:
            console.print(
                f"Endpoints detected: lower={endpoints.lower:.5f}, upper={endpoints.upper:.5f}"
            )
            status.update("[bold green]Step 2: Subtracting baseline...")

        # 2. Subtract baseline
        baseline_result = subtract_baseline(
            data=data,
            lower_temp=endpoints.lower,
            upper_temp=endpoints.upper,
            method="innermost",
            plot=False,
        )

        if verbose:
            console.print("Baseline subtracted successfully", style="green")
            status.update("[bold green]Step 3: Interpolating to uniform grid...")

        # 3. Interpolate to uniform grid
        interpolated_result = interpolate_sample(
            data=baseline_result,
            grid_temp=np.arange(45, 90.1, 0.1),
            plot=False,
        )

        if verbose:
            console.print("Processing complete", style="green bold")

    return {
        "endpoints": endpoints,
        "baseline_result": baseline_result,
        "interpolated_result": interpolated_result,
    }


def compare_results(python_results, r_results):
    """Compare Python and R implementation results.

    Args:
        python_results (dict): Dictionary with temperature and dCp arrays from Python
        r_results (dict): Dictionary with temperature and dCp arrays from R

    Returns:
        dict: Comparison metrics including:
            - temperature_range: Range of temperatures used for comparison
            - dcp_mse: Mean squared error between implementations
            - max_diff: Maximum absolute difference
            - correlation: Correlation coefficient
            - mean_python/r: Mean values from each implementation
            - std_python/r: Standard deviations from each implementation
            - python/r_area: Area under the curve for each implementation
    """
    try:
        # Ensure temperature ranges match by interpolating to common grid
        min_temp = max(
            min(python_results["temperature"]), min(r_results["temperature"])
        )
        max_temp = min(
            max(python_results["temperature"]), max(r_results["temperature"])
        )

        # Create common temperature grid
        common_temp = np.linspace(min_temp, max_temp, 451)

        # Interpolate both datasets to common grid using safe interpolation
        python_interp = _safe_interpolate(
            python_results["temperature"], python_results["dcp"], common_temp
        )

        r_interp = _safe_interpolate(
            r_results["temperature"], r_results["dcp"], common_temp
        )

        # Filter out invalid points
        valid_indices = np.isfinite(python_interp) & np.isfinite(r_interp)
        valid_temp = common_temp[valid_indices]
        valid_python = python_interp[valid_indices]
        valid_r = r_interp[valid_indices]

        if len(valid_temp) == 0:
            return {
                "temperature_range": f"{min_temp:.1f} - {max_temp:.1f} °C",
                "error": "No overlapping valid data points found",
                "valid_points": 0,
            }

        # Calculate comparison metrics
        comparison = {
            "temperature_range": f"{min_temp:.1f} - {max_temp:.1f} °C",
            "valid_points": len(valid_temp),
            "dcp_mse": np.mean((valid_python - valid_r) ** 2),
            "max_diff": np.max(np.abs(valid_python - valid_r)),
            "correlation": np.corrcoef(valid_python, valid_r)[0, 1],
            "mean_python": np.mean(valid_python),
            "mean_r": np.mean(valid_r),
            "std_python": np.std(valid_python),
            "std_r": np.std(valid_r),
        }

        # Calculate area under the curve if we have enough points
        if len(valid_temp) > 3:
            comparison["python_area"] = np.trapz(valid_python, valid_temp)
            comparison["r_area"] = np.trapz(valid_r, valid_temp)

        return comparison

    except Exception as e:
        console.print(f"[red]Error comparing results: {str(e)}[/]")
        return {"temperature_range": "Unknown", "error": str(e)}


def create_visualizations(python_data, r_data, raw_data, python_results, verbose=False):
    """Create visualization figures comparing Python and R results.

    Args:
        python_data (pl.DataFrame): Python-processed data
        r_data (pl.DataFrame): R-processed data
        raw_data (pl.DataFrame): Raw thermogram data
        python_results (dict): Results from Python processing
        verbose (bool): Whether to print verbose output

    Returns:
        dict: Dictionary of Plotly figure JSONs for different visualizations
    """
    figures = {}

    if verbose:
        console.print(
            "Creating visualizations for the thermogram processing workflow...",
            style="blue",
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        viz_task = progress.add_task("[green]Creating visualizations...", total=5)

        # 1. Endpoint Detection Visualization
        if raw_data is not None and "endpoints" in python_results:
            progress.update(
                viz_task, description="[green]Creating endpoint detection plot..."
            )

            fig_endpoints = make_subplots(rows=1, cols=1)

            # Plot original data
            fig_endpoints.add_trace(
                go.Scatter(
                    x=raw_data["Temperature"],
                    y=raw_data["dCp"],
                    mode="lines",
                    name="Raw Data",
                )
            )

            # Add endpoint markers
            fig_endpoints = _plot_endpoints(
                fig=fig_endpoints,
                temperatures=raw_data["Temperature"].to_numpy(),
                values=raw_data["dCp"].to_numpy(),
                lower_temp=python_results["endpoints"].lower,
                upper_temp=python_results["endpoints"].upper,
                name="Detected Endpoints",
            )

            fig_endpoints.update_layout(
                title="Raw Data with Detected Endpoints",
                xaxis_title="Temperature (°C)",
                yaxis_title="dCp",
                legend_title="Data",
            )

            figures["endpoints"] = fig_endpoints.to_json()

            progress.update(viz_task, advance=1)

        # 2. Baseline Subtraction Visualization (Raw Curve with Spline Overlay)
        if "baseline_result" in python_results:
            progress.update(
                viz_task, description="[green]Creating baseline overlay plot..."
            )

            fig_baseline = make_subplots(rows=1, cols=1)

            # Plot original data
            fig_baseline.add_trace(
                go.Scatter(
                    x=python_results["baseline_result"].original.temperature,
                    y=python_results["baseline_result"].original.dcp,
                    mode="lines",
                    name="Original Data",
                )
            )

            # Plot baseline
            fig_baseline.add_trace(
                go.Scatter(
                    x=python_results["baseline_result"].baseline.temperature,
                    y=python_results["baseline_result"].baseline.dcp,
                    mode="lines",
                    name="Calculated Baseline",
                    line=dict(color="red", dash="dash"),
                )
            )

            # Add endpoint markers using helper function
            fig_baseline = _plot_endpoints(
                fig=fig_baseline,
                temperatures=python_results["baseline_result"].original.temperature,
                values=python_results["baseline_result"].original.dcp,
                lower_temp=python_results["endpoints"].lower,
                upper_temp=python_results["endpoints"].upper,
                name="Endpoints",
            )

            fig_baseline.update_layout(
                title="Raw Curve with Spline Overlay",
                xaxis_title="Temperature (°C)",
                yaxis_title="dCp",
                legend_title="Data",
            )

            figures["baseline_overlay"] = fig_baseline.to_json()

            progress.update(viz_task, advance=1)

        # 3. Baseline Subtracted Result
        if "baseline_result" in python_results:
            progress.update(
                viz_task, description="[green]Creating baseline subtracted plot..."
            )

            fig_subtracted = make_subplots(rows=1, cols=1)

            fig_subtracted.add_trace(
                go.Scatter(
                    x=python_results["baseline_result"].subtracted.temperature,
                    y=python_results["baseline_result"].subtracted.dcp,
                    mode="lines",
                    name="Baseline Subtracted",
                )
            )

            fig_subtracted.update_layout(
                title="Baseline Subtracted Sample",
                xaxis_title="Temperature (°C)",
                yaxis_title="dCp",
            )

            figures["baseline_subtracted"] = fig_subtracted.to_json()

            progress.update(viz_task, advance=1)

        # 4. Interpolated Result
        if "interpolated_result" in python_results:
            progress.update(
                viz_task, description="[green]Creating interpolated result plot..."
            )

            fig_interpolated = make_subplots(rows=1, cols=1)

            fig_interpolated.add_trace(
                go.Scatter(
                    x=python_results["interpolated_result"].data.temperature,
                    y=python_results["interpolated_result"].data.dcp,
                    mode="lines+markers",
                    name="Interpolated Result",
                    marker=dict(size=3),
                )
            )

            fig_interpolated.update_layout(
                title="Interpolated Result",
                xaxis_title="Temperature (°C)",
                yaxis_title="dCp",
            )

            figures["interpolated"] = fig_interpolated.to_json()

            progress.update(viz_task, advance=1)

        # 5. Comparison with R implementation (if available)
        if r_data is not None and "interpolated_result" in python_results:
            progress.update(
                viz_task, description="[green]Creating comparison visualizations..."
            )

            if verbose:
                console.print(
                    "Creating comparison visualizations with R implementation...",
                    style="blue",
                )

            # 5.1 Python vs R processed data
            fig_comparison = make_subplots(rows=1, cols=1)

            fig_comparison.add_trace(
                go.Scatter(
                    x=python_results["interpolated_result"].data.temperature,
                    y=python_results["interpolated_result"].data.dcp,
                    mode="lines",
                    name="Python Implementation",
                    line=dict(color="blue"),
                )
            )

            fig_comparison.add_trace(
                go.Scatter(
                    x=r_data["Temperature"],
                    y=r_data["dCp"],
                    mode="lines",
                    name="R Implementation",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_comparison.update_layout(
                title="Python vs R Implementation Comparison",
                xaxis_title="Temperature (°C)",
                yaxis_title="dCp",
                legend_title="Implementation",
            )

            figures["python_vs_r"] = fig_comparison.to_json()

            # 5.2 Difference plot (Python - R)
            try:
                # Get temperature ranges
                min_temp = max(
                    min(python_results["interpolated_result"].data.temperature),
                    min(r_data["Temperature"]),
                )
                max_temp = min(
                    max(python_results["interpolated_result"].data.temperature),
                    max(r_data["Temperature"]),
                )

                # Create common temperature grid
                common_temp = np.linspace(min_temp, max_temp, 451)

                # Interpolate both datasets to common grid using safe interpolation
                python_interp = _safe_interpolate(
                    python_results["interpolated_result"].data.temperature,
                    python_results["interpolated_result"].data.dcp,
                    common_temp,
                )

                r_interp = _safe_interpolate(
                    r_data["Temperature"].to_numpy(),
                    r_data["dCp"].to_numpy(),
                    common_temp,
                )

                # Filter out invalid points
                valid_indices = np.isfinite(python_interp) & np.isfinite(r_interp)
                valid_temp = common_temp[valid_indices]
                valid_python = python_interp[valid_indices]
                valid_r = r_interp[valid_indices]

                if len(valid_temp) > 0:
                    # Calculate difference
                    diff = valid_python - valid_r

                    fig_diff = make_subplots(rows=1, cols=1)

                    fig_diff.add_trace(
                        go.Scatter(
                            x=valid_temp,
                            y=diff,
                            mode="lines",
                            name="Difference (Python - R)",
                            line=dict(color="purple"),
                        )
                    )

                    fig_diff.add_shape(
                        type="line",
                        x0=min(valid_temp),
                        y0=0,
                        x1=max(valid_temp),
                        y1=0,
                        line=dict(color="black", dash="dash"),
                    )

                    fig_diff.update_layout(
                        title="Difference Between Python and R Results",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Difference in dCp",
                        showlegend=True,
                    )

                    figures["difference"] = fig_diff.to_json()
                else:
                    console.print(
                        "[yellow]Warning: Could not create difference plot, no valid points[/]"
                    )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not create difference plot: {str(e)}[/]"
                )

            progress.update(viz_task, advance=1)

    return figures


def generate_report(
    python_data, r_data, raw_data, python_results, figures, output_path, verbose=False
):
    """Generate an HTML report with the verification results.

    Args:
        python_data (pl.DataFrame): Python-processed data
        r_data (pl.DataFrame): R-processed data
        raw_data (pl.DataFrame): Raw thermogram data
        python_results (dict): Results from Python processing
        figures (dict): Dictionary of Plotly figure JSONs
        output_path (str): Path to save the HTML report
        verbose (bool): Whether to print verbose output

    Returns:
        str: Path to the generated report
    """
    # Ensure output directory exists (this is a safeguard in case output_path is custom)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        console.print(f"Generating HTML report to [blue]{output_path}[/]")
        console.print(f"Using template directory: [blue]{TEMPLATE_DIR}[/]")

    # Create template loader
    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        # Create default template if it doesn't exist
        with open(TEMPLATE_DIR / "verification_report.html", "w") as f:
            f.write(DEFAULT_TEMPLATE)

    # Validate output path extension
    if not output_path.endswith(".html"):
        original_path = output_path
        output_path = f"{output_path}.html"
        console.print(
            f"[yellow]Adding .html extension to output path: {original_path} -> {output_path}[/]"
        )

    # Load template
    with console.status("[green]Rendering HTML template...", spinner="dots"):
        template_loader = jinja2.FileSystemLoader(searchpath=TEMPLATE_DIR)
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("verification_report.html")

        # Prepare data for template
        template_data = {
            "title": "Thermogram Baseline Verification Report",
            "figures": figures,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "script_version": "1.0.0",  # Add version for tracking
        }

        # Add python results if available
        if "endpoints" in python_results:
            template_data["python_results"] = {
                "endpoints": {
                    "lower": python_results["endpoints"].lower,
                    "upper": python_results["endpoints"].upper,
                    "method": python_results["endpoints"].method,
                },
            }

        # If we have R data, add comparison metrics
        if r_data is not None and "interpolated_result" in python_results:
            python_processed = {
                "temperature": python_results["interpolated_result"].data.temperature,
                "dcp": python_results["interpolated_result"].data.dcp,
            }
            r_processed = {
                "temperature": r_data["Temperature"].to_numpy(),
                "dcp": r_data["dCp"].to_numpy(),
            }
            template_data["comparison"] = compare_results(python_processed, r_processed)

        # Render template
        try:
            html_content = template.render(**template_data)
        except Exception as e:
            console.print(f"[red]Error rendering template: {str(e)}[/]")
            # Create a very simple report instead
            html_content = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>Thermogram Baseline Verification Report (Error)</title>
            </head>
            <body>
                <h1>Thermogram Baseline Verification Report</h1>
                <p>Error generating report: {str(e)}</p>
                <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </body>
            </html>"""

    # Write to file
    try:
        with open(output_path, "w") as f:
            f.write(html_content)
    except Exception as e:
        console.print(f"[red]Error writing report to {output_path}: {str(e)}[/]")
        # Try writing to a fallback location
        fallback_path = Path("verification_report_fallback.html")
        console.print(f"[yellow]Trying fallback path: {fallback_path}[/]")
        with open(fallback_path, "w") as f:
            f.write(html_content)
        output_path = fallback_path

    if verbose:
        console.print(f"Report generated successfully at [green]{output_path}[/]")

    return output_path


# Main function
def main():
    """Run the verification script."""
    # Ensure all required directories exist
    ensure_directories_exist()

    args = parse_args()
    verbose = args.verbose

    if verbose:
        print("Running verification with following parameters:")
        print(f"  Python data: {args.python_data}")
        print(f"  R data: {args.r_data}")
        print(f"  Raw data: {args.raw_data}")
        print(f"  Output path: {args.output}")

    # Load data
    python_data, r_data, raw_data = load_data(
        args.python_data, args.r_data, args.raw_data, verbose=verbose
    )

    # Initialize results
    python_results = {}

    # If raw data is provided, process it
    if raw_data is not None:
        if verbose:
            print("Processing raw data with Python implementation...")
        python_results = process_sample(raw_data, verbose=verbose)
    elif python_data is not None:
        # If only processed data is available, we can still compare
        if verbose:
            print("Using pre-processed Python data (no raw data processing)")
        python_results["interpolated_result"] = ThermogramData(
            temperature=python_data["Temperature"].to_numpy(),
            dcp=python_data["dCp"].to_numpy(),
        )

    # Create visualizations
    figures = create_visualizations(
        python_data, r_data, raw_data, python_results, verbose=verbose
    )

    # Generate report
    report_path = generate_report(
        python_data,
        r_data,
        raw_data,
        python_results,
        figures,
        args.output,
        verbose=verbose,
    )

    print(f"Verification report generated at: {report_path}")


# Default HTML template
DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        header {
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0;
            color: #333;
        }
        .timestamp {
            color: #6c757d;
            font-size: 0.9em;
        }
        section {
            margin-bottom: 30px;
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h2 {
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .plot {
            width: 100%;
            height: 500px;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .metric-value {
            font-weight: bold;
        }
        .workflow-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .workflow-step {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .workflow-step h3 {
            margin-top: 0;
            color: #495057;
        }
        .overview-text {
            margin-bottom: 20px;
            color: #495057;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <header>
        <h1>{{ title }}</h1>
        <p class="timestamp">Generated on: {{ timestamp }}</p>
    </header>

    <section>
        <h2>Thermogram Processing Workflow</h2>
        <p class="overview-text">
            This report visualizes the complete thermogram processing workflow from the Python implementation,
            mirroring the process demonstrated in the R ThermogramBaseline package. Each step of the process
            is visualized below, from raw data to the final interpolated result.
        </p>
        
        <div class="workflow-container">
            {% if figures.endpoints is defined %}
            <div class="workflow-step">
                <h3>Step 1: Endpoint Detection</h3>
                <p>
                    The first step is to detect suitable endpoints for baseline subtraction. The algorithm identifies 
                    regions of low variance outside the exclusion zone (typically 60-80°C). These endpoints will be 
                    used to anchor the baseline.
                </p>
                <div id="endpoints-plot" class="plot"></div>
                {% if python_results is defined and python_results.endpoints is defined %}
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Lower Endpoint</td>
                        <td class="metric-value">{{ python_results.endpoints.lower|round(3) }} °C</td>
                    </tr>
                    <tr>
                        <td>Upper Endpoint</td>
                        <td class="metric-value">{{ python_results.endpoints.upper|round(3) }} °C</td>
                    </tr>
                    <tr>
                        <td>Selection Method</td>
                        <td class="metric-value">{{ python_results.endpoints.method }}</td>
                    </tr>
                </table>
                {% endif %}
            </div>
            {% endif %}
            
            {% if figures.baseline_overlay is defined %}
            <div class="workflow-step">
                <h3>Step 2: Baseline Calculation</h3>
                <p>
                    Next, a baseline is calculated using the detected endpoints. This involves fitting splines to the regions 
                    outside the endpoints and connecting them with a straight line through the transition region. The red 
                    dashed line shows the calculated baseline.
                </p>
                <div id="baseline-overlay-plot" class="plot"></div>
            </div>
            {% endif %}
            
            {% if figures.baseline_subtracted is defined %}
            <div class="workflow-step">
                <h3>Step 3: Baseline Subtraction</h3>
                <p>
                    The calculated baseline is subtracted from the original data to isolate the signal of interest.
                    This removes the background trend and highlights the thermal transitions.
                </p>
                <div id="baseline-subtracted-plot" class="plot"></div>
            </div>
            {% endif %}
            
            {% if figures.interpolated is defined %}
            <div class="workflow-step">
                <h3>Step 4: Interpolation to Uniform Grid</h3>
                <p>
                    Finally, the baseline-subtracted data is interpolated onto a uniform temperature grid
                    (typically 45-90°C with 0.1°C steps) for easier analysis and comparison. This is the final
                    processed thermogram.
                </p>
                <div id="interpolated-plot" class="plot"></div>
            </div>
            {% endif %}
        </div>
    </section>

    {% if comparison is defined %}
    <section>
        <h2>Comparison with R Implementation</h2>
        <p class="overview-text">
            The following analysis compares the Python implementation with the reference R implementation
            to verify consistency and accuracy. Visual comparisons and quantitative metrics are provided.
        </p>
        
        {% if figures.python_vs_r is defined %}
        <h3>Visual Comparison</h3>
        <p>Direct comparison of the final results from both implementations:</p>
        <div id="python-vs-r-plot" class="plot"></div>
        {% endif %}
        
        {% if figures.difference is defined %}
        <h3>Difference Analysis</h3>
        <p>The difference between Python and R results (Python minus R):</p>
        <div id="difference-plot" class="plot"></div>
        {% endif %}
        
        <h3>Quantitative Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Temperature Range</td>
                <td class="metric-value">{{ comparison.temperature_range }}</td>
            </tr>
            <tr>
                <td>Mean Squared Error (dCp)</td>
                <td class="metric-value">{{ comparison.dcp_mse|round(6) }}</td>
            </tr>
            <tr>
                <td>Maximum Absolute Difference</td>
                <td class="metric-value">{{ comparison.max_diff|round(6) }}</td>
            </tr>
            <tr>
                <td>Correlation Coefficient</td>
                <td class="metric-value">{{ comparison.correlation|round(6) }}</td>
            </tr>
            <tr>
                <td>Area Under Curve (Python)</td>
                <td class="metric-value">{{ comparison.python_area|round(4) }}</td>
            </tr>
            <tr>
                <td>Area Under Curve (R)</td>
                <td class="metric-value">{{ comparison.r_area|round(4) }}</td>
            </tr>
            <tr>
                <td>Mean Value (Python)</td>
                <td class="metric-value">{{ comparison.mean_python|round(6) }}</td>
            </tr>
            <tr>
                <td>Mean Value (R)</td>
                <td class="metric-value">{{ comparison.mean_r|round(6) }}</td>
            </tr>
            <tr>
                <td>Standard Deviation (Python)</td>
                <td class="metric-value">{{ comparison.std_python|round(6) }}</td>
            </tr>
            <tr>
                <td>Standard Deviation (R)</td>
                <td class="metric-value">{{ comparison.std_r|round(6) }}</td>
            </tr>
        </table>
    </section>
    {% endif %}

    <script>
        {% if figures.endpoints is defined %}
        var endpointsPlot = JSON.parse('{{ figures.endpoints|safe }}');
        Plotly.newPlot('endpoints-plot', endpointsPlot.data, endpointsPlot.layout);
        {% endif %}
        
        {% if figures.baseline_overlay is defined %}
        var baselineOverlayPlot = JSON.parse('{{ figures.baseline_overlay|safe }}');
        Plotly.newPlot('baseline-overlay-plot', baselineOverlayPlot.data, baselineOverlayPlot.layout);
        {% endif %}
        
        {% if figures.baseline_subtracted is defined %}
        var baselineSubtractedPlot = JSON.parse('{{ figures.baseline_subtracted|safe }}');
        Plotly.newPlot('baseline-subtracted-plot', baselineSubtractedPlot.data, baselineSubtractedPlot.layout);
        {% endif %}
        
        {% if figures.interpolated is defined %}
        var interpolatedPlot = JSON.parse('{{ figures.interpolated|safe }}');
        Plotly.newPlot('interpolated-plot', interpolatedPlot.data, interpolatedPlot.layout);
        {% endif %}
        
        {% if figures.python_vs_r is defined %}
        var pythonVsRPlot = JSON.parse('{{ figures.python_vs_r|safe }}');
        Plotly.newPlot('python-vs-r-plot', pythonVsRPlot.data, pythonVsRPlot.layout);
        {% endif %}
        
        {% if figures.difference is defined %}
        var differencePlot = JSON.parse('{{ figures.difference|safe }}');
        Plotly.newPlot('difference-plot', differencePlot.data, differencePlot.layout);
        {% endif %}
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
