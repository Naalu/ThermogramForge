#!/usr/bin/env python3
"""
CLI script for testing thermogram baseline and endpoint detection.

This script provides a simple command-line interface for testing the
core functionality of the thermogram_baseline package.
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.subplots as sp  # type: ignore
import polars as pl

from thermogram_baseline.endpoint_detection import Endpoints, detect_endpoints
from thermogram_baseline.spline_fitter import SplineFitter

# Define default paths
DEFAULT_OUTPUT_DIR = Path("data/generated/html")


def load_data(file_path: str) -> pl.DataFrame:
    """Load thermogram data from a CSV or Excel file."""
    path = Path(file_path)

    # Check file existence
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    # Load based on file extension
    if path.suffix.lower() == ".csv":
        try:
            data = pl.read_csv(path)
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            sys.exit(1)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            data = pl.read_excel(path)
        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")
            sys.exit(1)
    else:
        print(f"Unsupported file format: {path.suffix}")
        sys.exit(1)

    # Check for required columns
    required_columns = ["Temperature", "dCp"]
    if not all(col in data.columns for col in required_columns):
        print(f"Error: Input file must contain columns: {required_columns}")
        print(f"Found columns: {data.columns}")
        sys.exit(1)

    return data


def detect_endpoints_cli(args: argparse.Namespace) -> None:
    """Run endpoint detection on input file."""
    # Load data
    data = load_data(args.input)

    # Run endpoint detection
    print("Running endpoint detection with:")
    print(f"  Window size (w): {args.window}")
    print(f"  Exclusion zone: [{args.exclusion_lower}, {args.exclusion_upper}]")
    print(f"  Point selection: {args.point_selection}")

    endpoints: Endpoints = detect_endpoints(
        data,
        w=args.window,
        exclusion_lwr=args.exclusion_lower,
        exclusion_upr=args.exclusion_upper,
        point_selection=args.point_selection,
        explicit=True,
    )

    # Print results
    print("\nResults:")
    print(f"  Lower endpoint: {endpoints.lower:.2f}")
    print(f"  Upper endpoint: {endpoints.upper:.2f}")
    print(f"  Method: {endpoints.method}")

    # Visualize if requested
    if args.visualize:
        fig = visualize_endpoints(data, endpoints)

        # Create output directory if it doesn't exist
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Define output file path
        if args.output:
            output_path = args.output
        else:
            filename = f"endpoints_{Path(args.input).stem}.html"
            output_path = str(DEFAULT_OUTPUT_DIR / filename)

        fig.write_html(output_path)
        print(f"Visualization saved to {output_path}")

        # Open in browser if requested
        if args.open_browser:
            webbrowser.open(f"file://{Path(output_path).resolve()}")


def fit_spline_cli(args: argparse.Namespace) -> None:
    """Run spline fitting on input file."""
    # Load data
    data = load_data(args.input)

    # Extract data
    temps = data.select("Temperature").to_numpy().flatten()
    dcps = data.select("dCp").to_numpy().flatten()

    # Run spline fitting
    print("Running spline fitting...")

    fitter = SplineFitter()
    spline = fitter.fit_with_gcv(temps, dcps)

    # Calculate fitted values
    fitted = spline(temps)

    # Print results
    print("\nResults:")
    print(f"  Optimal smoothing parameter: {spline.s_opt:.4f}")
    print(f"  Number of knots: {len(spline.get_knots())}")

    # Calculate R²
    ss_total = np.sum((dcps - np.mean(dcps)) ** 2)
    ss_residual = np.sum((dcps - fitted) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"  R²: {r_squared:.4f}")

    # Calculate residual standard error
    n = len(temps)
    p = len(spline.get_knots())
    if n > p:
        rse = np.sqrt(ss_residual / (n - p))
        print(f"  Residual standard error: {rse:.6f}")

    # Visualize if requested
    if args.visualize:
        fig = visualize_spline(temps, dcps, fitted)

        # Create output directory if it doesn't exist
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Define output file path
        if args.output:
            output_path = args.output
        else:
            filename = f"spline_{Path(args.input).stem}.html"
            output_path = str(DEFAULT_OUTPUT_DIR / filename)

        fig.write_html(output_path)
        print(f"Visualization saved to {output_path}")

        # Open in browser if requested
        if args.open_browser:
            webbrowser.open(f"file://{Path(output_path).resolve()}")


def visualize_endpoints(data: pl.DataFrame, endpoints: Endpoints) -> go.Figure:
    """Visualize thermogram data with detected endpoints.

    Args:
        data: DataFrame containing Temperature and dCp columns
        endpoints: Detected endpoints object

    Returns:
        Plotly figure object
    """
    # Sort data by temperature
    data = data.sort("Temperature")
    temps = data.select("Temperature").to_numpy().flatten()
    dcps = data.select("dCp").to_numpy().flatten()

    # Create figure
    fig = go.Figure()

    # Plot data points
    fig.add_trace(
        go.Scatter(
            x=temps,
            y=dcps,
            mode="markers",
            marker=dict(size=5, opacity=0.5),
            name="Data",
        )
    )

    # Plot endpoints
    lower_temp = endpoints.lower
    upper_temp = endpoints.upper

    # Find y values at endpoints
    lower_idx = np.abs(temps - lower_temp).argmin()
    upper_idx = np.abs(temps - upper_temp).argmin()

    fig.add_trace(
        go.Scatter(
            x=[lower_temp, upper_temp],
            y=[dcps[lower_idx], dcps[upper_idx]],
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name="Endpoints",
        )
    )

    # Add exclusion zone shading
    exclusion_lwr = endpoints.lower
    exclusion_upr = endpoints.upper

    # Add a rectangle for the exclusion zone
    fig.add_shape(
        type="rect",
        x0=exclusion_lwr,
        x1=exclusion_upr,
        y0=min(dcps) - 0.05 * (max(dcps) - min(dcps)),
        y1=max(dcps) + 0.05 * (max(dcps) - min(dcps)),
        fillcolor="gray",
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    # Add annotation for exclusion zone
    fig.add_annotation(
        x=(exclusion_lwr + exclusion_upr) / 2,
        y=min(dcps) - 0.07 * (max(dcps) - min(dcps)),
        text="Exclusion Zone",
        showarrow=False,
        yshift=-10,
    )

    # Customize layout
    fig.update_layout(
        title="Thermogram with Detected Endpoints",
        xaxis_title="Temperature (°C)",
        yaxis_title="Excess Heat Capacity (dCp)",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        width=900,
        height=600,
        template="plotly_white",
    )

    fig.update_xaxes(
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig


def visualize_spline(
    temps: np.ndarray, dcps: np.ndarray, fitted: np.ndarray
) -> go.Figure:
    """Visualize original data with fitted spline.

    Args:
        temps: Array of temperature values
        dcps: Array of heat capacity values
        fitted: Array of fitted values from spline

    Returns:
        Plotly figure with two subplots
    """
    # Sort by temperature
    sort_idx = np.argsort(temps)
    temps_sorted = temps[sort_idx]
    dcps_sorted = dcps[sort_idx]
    fitted_sorted = fitted[sort_idx]

    # Calculate residuals
    residuals = dcps_sorted - fitted_sorted

    # Create figure with subplots
    fig = sp.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Thermogram with Fitted Spline", "Residuals"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
    )

    # Add data trace to top plot
    fig.add_trace(
        go.Scatter(
            x=temps_sorted,
            y=dcps_sorted,
            mode="markers",
            marker=dict(size=5, opacity=0.5),
            name="Data",
        ),
        row=1,
        col=1,
    )

    # Add fitted line to top plot
    fig.add_trace(
        go.Scatter(
            x=temps_sorted,
            y=fitted_sorted,
            mode="lines",
            line=dict(color="red", width=2),
            name="Fitted Spline",
        ),
        row=1,
        col=1,
    )

    # Add residuals to bottom plot
    fig.add_trace(
        go.Scatter(
            x=temps_sorted,
            y=residuals,
            mode="markers",
            marker=dict(size=5, opacity=0.5),
            name="Residuals",
        ),
        row=2,
        col=1,
    )

    # Add zero line to residuals plot
    fig.add_shape(
        type="line",
        x0=min(temps_sorted),
        x1=max(temps_sorted),
        y0=0,
        y1=0,
        line=dict(color="red", width=1, dash="dash"),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title="Spline Fitting Results",
        showlegend=True,
        width=900,
        height=800,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
    )

    # Update axes
    fig.update_xaxes(
        title_text="Temperature (°C)",
        row=1,
        col=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="Excess Heat Capacity (dCp)",
        row=1,
        col=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_xaxes(
        title_text="Temperature (°C)",
        row=2,
        col=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="Residuals",
        row=2,
        col=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig


def main() -> None:
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Thermogram Baseline Analysis CLI")
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Endpoint detection command
    endpoint_parser = subparsers.add_parser(
        "endpoints", help="Detect endpoints for baseline subtraction"
    )
    endpoint_parser.add_argument("input", help="Input file (CSV or Excel)")
    endpoint_parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=90,
        help="Window size for variance calculation",
    )
    endpoint_parser.add_argument(
        "--exclusion-lower",
        "-el",
        type=float,
        default=60,
        help="Lower bound of exclusion window",
    )
    endpoint_parser.add_argument(
        "--exclusion-upper",
        "-eu",
        type=float,
        default=80,
        help="Upper bound of exclusion window",
    )
    endpoint_parser.add_argument(
        "--point-selection",
        "-p",
        choices=["innermost", "outermost", "mid"],
        default="innermost",
        help="Method for endpoint selection",
    )
    endpoint_parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize results"
    )
    endpoint_parser.add_argument(
        "--output", "-o", help="Output file for visualization (HTML)"
    )
    endpoint_parser.add_argument(
        "--open-browser",
        "-b",
        action="store_false",  # Changed to store_false so default is True
        help="Disable opening visualization in browser",
    )

    # Spline fitting command
    spline_parser = subparsers.add_parser(
        "spline", help="Fit spline to thermogram data"
    )
    spline_parser.add_argument("input", help="Input file (CSV or Excel)")
    spline_parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize results"
    )
    spline_parser.add_argument(
        "--output", "-o", help="Output file for visualization (HTML)"
    )
    spline_parser.add_argument(
        "--open-browser",
        "-b",
        action="store_false",  # Changed to store_false so default is True
        help="Disable opening visualization in browser",
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "endpoints":
        detect_endpoints_cli(args)
    elif args.command == "spline":
        fit_spline_cli(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
