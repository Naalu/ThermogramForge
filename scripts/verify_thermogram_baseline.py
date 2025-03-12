#!/usr/bin/env python
"""
Verification script for thermogram_baseline package.

This script compares the results of the Python implementation against the R implementation
for baseline subtraction and generates an HTML report with visualizations.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

# Add the project root to Python path to enable imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jinja2
import numpy as np
import plotly.graph_objects as go  # type: ignore
import polars as pl
from plotly.subplots import make_subplots  # type: ignore
from thermogram_baseline import (
    ThermogramData,
    detect_endpoints,
    interpolate_sample,
    subtract_baseline,
)

# Default file paths
TEMPLATE_DIR = PROJECT_ROOT / "scripts" / "templates"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DEFAULT = DATA_DIR / "raw" / "example_thermogram.csv"
R_DATA_DEFAULT = DATA_DIR / "reference" / "r_processed.csv"
PYTHON_DATA_DEFAULT = DATA_DIR / "processed" / "python_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "verification_results"
OUTPUT_DEFAULT = OUTPUT_DIR / "verification_report.html"


def parse_args():
    """Parse command line arguments."""
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
    return parser.parse_args()


def detect_data_format(df):
    """Detect the format of the data (ThermogramBaseline or tlbparam format)."""
    if "Temperature" in df.columns and "dCp" in df.columns:
        return "thermogram_baseline"

    # Check for temperature columns like "T45", "T45.1", etc.
    temp_cols = [
        col
        for col in df.columns
        if col.startswith("T") and col[1:].replace(".", "", 1).isdigit()
    ]
    if temp_cols:
        return "tlbparam"

    raise ValueError(
        "Unknown data format. Expected 'Temperature'/'dCp' columns or 'T{number}' columns"
    )


def convert_tlbparam_format(df):
    """Convert tlbparam format to thermogram_baseline format."""
    # Find temperature columns (starting with T)
    temp_cols = [
        col
        for col in df.columns
        if col.startswith("T") and col[1:].replace(".", "", 1).isdigit()
    ]

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
    temperatures = [float(col[1:]) for col in temp_cols]
    dcp_values = sample_data.select(temp_cols).to_numpy()[0]

    # Create new DataFrame in thermogram_baseline format
    new_df = pl.DataFrame({"Temperature": temperatures, "dCp": dcp_values})

    return new_df


def load_data(python_path, r_path, raw_path, verbose=False):
    """Load data from Python, R implementations and raw data."""
    data = {}

    # Load and check existence of files
    for name, path in [("python", python_path), ("r", r_path), ("raw", raw_path)]:
        if path and os.path.exists(path):
            try:
                df = pl.read_csv(path)

                # Detect format and convert if needed
                format_type = detect_data_format(df)
                if verbose:
                    print(f"Detected {name} data format: {format_type}")

                if format_type == "tlbparam":
                    df = convert_tlbparam_format(df)
                    if verbose:
                        print(f"Converted {name} data from tlbparam format")

                data[name] = df
                if verbose:
                    print(f"Successfully loaded {name} data from {path}")
            except Exception as e:
                print(f"Error loading {name} data from {path}: {e}")
                data[name] = None
        else:
            if verbose and path:
                print(f"File not found: {path}")
            data[name] = None

    return data.get("python"), data.get("r"), data.get("raw")


def process_sample(raw_data, sample_id=None, verbose=False):
    """Process a sample using the Python implementation."""
    # Convert to ThermogramData if needed
    if isinstance(raw_data, pl.DataFrame):
        data = ThermogramData.from_dataframe(raw_data)
    else:
        data = raw_data

    if verbose:
        print("Detecting endpoints...")

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
        print(f"Endpoints detected: lower={endpoints.lower}, upper={endpoints.upper}")
        print("Subtracting baseline...")

    # 2. Subtract baseline
    baseline_result = subtract_baseline(
        data=data,
        lower_temp=endpoints.lower,
        upper_temp=endpoints.upper,
        method="innermost",
        plot=False,
    )

    if verbose:
        print("Baseline subtracted successfully")
        print("Interpolating to uniform grid...")

    # 3. Interpolate to uniform grid
    interpolated_result = interpolate_sample(
        data=baseline_result,
        grid_temp=np.arange(45, 90.1, 0.1),
        plot=False,
    )

    if verbose:
        print("Processing complete")

    return {
        "endpoints": endpoints,
        "baseline_result": baseline_result,
        "interpolated_result": interpolated_result,
    }


def compare_results(python_results, r_results):
    """Compare Python and R implementation results."""
    # Ensure temperature ranges match by interpolating to common grid
    min_temp = max(min(python_results["temperature"]), min(r_results["temperature"]))
    max_temp = min(max(python_results["temperature"]), max(r_results["temperature"]))

    # Create common temperature grid
    common_temp = np.linspace(min_temp, max_temp, 451)

    # Interpolate both datasets to common grid
    from scipy.interpolate import interp1d  # type: ignore

    python_interp = interp1d(python_results["temperature"], python_results["dcp"])(
        common_temp
    )
    r_interp = interp1d(r_results["temperature"], r_results["dcp"])(common_temp)

    # Calculate comparison metrics
    comparison = {
        "temperature_range": f"{min_temp:.1f} - {max_temp:.1f} °C",
        "dcp_mse": np.mean((python_interp - r_interp) ** 2),
        "max_diff": np.max(np.abs(python_interp - r_interp)),
        "correlation": np.corrcoef(python_interp, r_interp)[0, 1],
        "mean_python": np.mean(python_interp),
        "mean_r": np.mean(r_interp),
        "std_python": np.std(python_interp),
        "std_r": np.std(r_interp),
    }
    return comparison


def create_visualizations(python_data, r_data, raw_data, python_results, verbose=False):
    """Create visualization figures comparing Python and R results."""
    figures = {}

    if verbose:
        print("Creating visualization: Raw vs Python processed")

    # 1. Raw data vs Python processed data
    if raw_data is not None and python_results.get("interpolated_result") is not None:
        fig1 = make_subplots(rows=1, cols=1)
        fig1.add_trace(
            go.Scatter(
                x=raw_data["Temperature"],
                y=raw_data["dCp"],
                mode="lines",
                name="Raw Data",
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=python_results["interpolated_result"].data.temperature,
                y=python_results["interpolated_result"].data.dcp,
                mode="lines",
                name="Python Processed",
            )
        )
        fig1.update_layout(
            title="Raw Data vs Python Processed",
            xaxis_title="Temperature (°C)",
            yaxis_title="dCp",
            legend_title="Data Source",
        )
        figures["raw_vs_python"] = fig1.to_json()

    if verbose and r_data is not None:
        print("Creating visualization: Python vs R processed")

    # 2. Python vs R processed data
    if r_data is not None and python_results.get("interpolated_result") is not None:
        fig2 = make_subplots(rows=1, cols=1)
        fig2.add_trace(
            go.Scatter(
                x=python_results["interpolated_result"].data.temperature,
                y=python_results["interpolated_result"].data.dcp,
                mode="lines",
                name="Python Processed",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=r_data["Temperature"],
                y=r_data["dCp"],
                mode="lines",
                name="R Processed",
            )
        )
        fig2.update_layout(
            title="Python vs R Processed Data",
            xaxis_title="Temperature (°C)",
            yaxis_title="dCp",
            legend_title="Implementation",
        )
        figures["python_vs_r"] = fig2.to_json()

    if verbose and "baseline_result" in python_results:
        print("Creating visualization: Baseline detection")

    # 3. Baseline visualization
    if "baseline_result" in python_results and "endpoints" in python_results:
        fig3 = make_subplots(rows=1, cols=1)
        fig3.add_trace(
            go.Scatter(
                x=python_results["baseline_result"].original.temperature,
                y=python_results["baseline_result"].original.dcp,
                mode="lines",
                name="Original Data",
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=python_results["baseline_result"].baseline.temperature,
                y=python_results["baseline_result"].baseline.dcp,
                mode="lines",
                name="Calculated Baseline",
                line=dict(dash="dash"),
            )
        )
        # Add endpoint markers
        lower_temp = python_results["endpoints"].lower
        upper_temp = python_results["endpoints"].upper

        # Find the dCp values at the endpoints
        lower_idx = np.abs(
            python_results["baseline_result"].original.temperature - lower_temp
        ).argmin()
        upper_idx = np.abs(
            python_results["baseline_result"].original.temperature - upper_temp
        ).argmin()
        lower_dcp = python_results["baseline_result"].original.dcp[lower_idx]
        upper_dcp = python_results["baseline_result"].original.dcp[upper_idx]

        fig3.add_trace(
            go.Scatter(
                x=[lower_temp, upper_temp],
                y=[lower_dcp, upper_dcp],
                mode="markers",
                name="Endpoints",
                marker=dict(size=10),
            )
        )
        fig3.update_layout(
            title="Baseline Detection",
            xaxis_title="Temperature (°C)",
            yaxis_title="dCp",
            legend_title="Data",
        )
        figures["baseline"] = fig3.to_json()

    return figures


def generate_report(
    python_data, r_data, raw_data, python_results, figures, output_path, verbose=False
):
    """Generate an HTML report with the verification results."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Generating HTML report to {output_path}")
        print(f"Using template directory: {TEMPLATE_DIR}")

    # Create template loader
    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        # Create default template if it doesn't exist
        with open(TEMPLATE_DIR / "verification_report.html", "w") as f:
            f.write(DEFAULT_TEMPLATE)

    # Load template
    template_loader = jinja2.FileSystemLoader(searchpath=TEMPLATE_DIR)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("verification_report.html")

    # Prepare data for template
    template_data = {
        "title": "Thermogram Baseline Verification Report",
        "figures": figures,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    html_content = template.render(**template_data)

    # Write to file
    with open(output_path, "w") as f:
        f.write(html_content)

    if verbose:
        print(f"Report generated successfully at {output_path}")

    return output_path


def main():
    """Run the verification script."""
    args = parse_args()
    verbose = args.verbose

    if verbose:
        print("Running verification with following parameters:")
        print(f"  Python data: {args.python_data}")
        print(f"  R data: {args.r_data}")
        print(f"  Raw data: {args.raw_data}")
        print(f"  Output path: {args.output}")

    # Create directories if they don't exist
    for directory in [
        DATA_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "reference",
        DATA_DIR / "processed",
        OUTPUT_DIR,
        TEMPLATE_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    # Load data
    python_data, r_data, raw_data = load_data(
        args.python_data, args.r_data, args.raw_data, verbose=verbose
    )

    # Initialize results
    python_results = {}

    # If raw data is provided, process it
    if raw_data is not None:
        if verbose:
            print("Processing raw data with Python implementation")
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
    </style>
</head>
<body>
    <header>
        <h1>{{ title }}</h1>
        <p class="timestamp">Generated on: {{ timestamp }}</p>
    </header>

    {% if python_results is defined and python_results.endpoints is defined %}
    <section>
        <h2>Baseline Detection</h2>
        <p>The Python implementation detected the following baseline endpoints:</p>
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
        
        {% if figures.baseline is defined %}
        <div id="baseline-plot" class="plot"></div>
        {% endif %}
    </section>
    {% endif %}

    {% if comparison is defined %}
    <section>
        <h2>Comparison with R Implementation</h2>
        <p>The following metrics compare the Python implementation with the R implementation:</p>
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
        
        {% if figures.python_vs_r is defined %}
        <div id="comparison-plot" class="plot"></div>
        {% endif %}
    </section>
    {% endif %}

    {% if figures.raw_vs_python is defined %}
    <section>
        <h2>Raw vs. Processed Data</h2>
        <p>Comparison between the raw thermogram data and the output of the Python implementation:</p>
        <div id="raw-vs-python-plot" class="plot"></div>
    </section>
    {% endif %}

    <script>
        {% if figures.baseline is defined %}
        // Load the baseline plot
        var baselinePlot = JSON.parse('{{ figures.baseline|safe }}');
        Plotly.newPlot('baseline-plot', baselinePlot.data, baselinePlot.layout);
        {% endif %}
        
        {% if figures.raw_vs_python is defined %}
        // Load the raw vs python plot
        var rawVsPythonPlot = JSON.parse('{{ figures.raw_vs_python|safe }}');
        Plotly.newPlot('raw-vs-python-plot', rawVsPythonPlot.data, rawVsPythonPlot.layout);
        {% endif %}
        
        {% if figures.python_vs_r is defined %}
        // Load the comparison plot
        var comparisonPlot = JSON.parse('{{ figures.python_vs_r|safe }}');
        Plotly.newPlot('comparison-plot', comparisonPlot.data, comparisonPlot.layout);
        {% endif %}
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
