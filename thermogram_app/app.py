import base64
import io
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
import numpy as np
import plotly.graph_objects as go  # type: ignore
import polars as pl
from dash import Input, Output, State, callback, dcc, html  # type: ignore
from dash.development.base_component import Component  # type: ignore

import thermogram_baseline
from thermogram_baseline.baseline import subtract_baseline
from thermogram_baseline.endpoint_detection import detect_endpoints
from thermogram_baseline.interpolation import interpolate_thermogram
from tlbparam.metrics import ThermogramAnalyzer
from tlbparam.peak_detection import PeakDetector
from tlbparam.visualization import plot_thermogram, plot_with_baseline, plot_with_peaks


def main() -> None:
    """Run the application."""
    app.run_server(debug=True)


"""
Web application for thermogram analysis.

This module provides a Dash-based web application for uploading, processing,
and visualizing thermogram data.
"""

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ThermogramForge",
    suppress_callback_exceptions=True,
)

# Disable R integration to avoid rpy2 issues
os.environ["THERMOGRAM_FORGE_USE_R"] = "0"

# Create layout separately and then assign to avoid the "read-only" mypy error
layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("ThermogramForge", className="my-4"),
                        html.P(
                            f"Version {thermogram_baseline.__version__}",
                            className="lead",
                        ),
                        html.Hr(),
                    ],
                    width=12,
                )
            ]
        ),
        # Main content
        dbc.Row(
            [
                # Left column - Upload and controls
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data Input"),
                                dbc.CardBody(
                                    [
                                        # File upload component
                                        dcc.Upload(
                                            id="upload-data",
                                            children=html.Div(
                                                [
                                                    "Drag and Drop or ",
                                                    html.A("Select CSV or Excel Files"),
                                                ]
                                            ),
                                            style={
                                                "width": "100%",
                                                "height": "60px",
                                                "lineHeight": "60px",
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "5px",
                                                "textAlign": "center",
                                                "margin": "10px 0",
                                            },
                                            multiple=True,
                                        ),
                                        html.Div(id="upload-status"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Analysis Controls"),
                                dbc.CardBody(
                                    [
                                        html.P("Baseline Subtraction Range:"),
                                        dcc.RangeSlider(
                                            id="baseline-range",
                                            min=45,
                                            max=90,
                                            step=0.5,
                                            marks={
                                                i: f"{i}°C" for i in range(45, 91, 5)
                                            },
                                            value=[55, 85],
                                            className="mb-4",
                                        ),
                                        html.P("Thermogram Processing:"),
                                        dbc.Checklist(
                                            id="processing-options",
                                            options=[
                                                {
                                                    "label": "Auto-detect baseline",
                                                    "value": "auto-baseline",
                                                },
                                                {
                                                    "label": (
                                                        "Interpolate to regular grid"
                                                    ),
                                                    "value": "interpolate",
                                                },
                                                {
                                                    "label": "Detect peaks",
                                                    "value": "detect-peaks",
                                                },
                                                {
                                                    "label": "Calculate metrics",
                                                    "value": "calculate-metrics",
                                                },
                                            ],
                                            value=[
                                                "auto-baseline",
                                                "interpolate",
                                                "detect-peaks",
                                            ],
                                            className="mb-4",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Process Thermograms",
                                                    id="process-button",
                                                    color="primary",
                                                    className="mt-2",
                                                ),
                                                html.Div(
                                                    id="processing-status",
                                                    className="mt-2",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=4,
                ),
                # Right column - Results and visualization
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    [
                                        dcc.Loading(
                                            id="loading-thermogram",
                                            type="circle",
                                            children=[
                                                dcc.Graph(
                                                    id="thermogram-plot",
                                                    style={"height": "600px"},
                                                ),
                                            ],
                                        ),
                                    ],
                                    label="Visualization",
                                ),
                                dbc.Tab(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Peak Information"),
                                                dbc.CardBody(id="peak-info"),
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Detected Metrics"),
                                                dbc.CardBody(id="metrics-info"),
                                            ]
                                        ),
                                    ],
                                    label="Metrics",
                                ),
                                dbc.Tab(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Data Preview"),
                                                dbc.CardBody(id="data-preview"),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Download Processed Data",
                                                    id="btn-download",
                                                    color="success",
                                                    className="mt-3",
                                                ),
                                                dcc.Download(id="download-data"),
                                            ],
                                            className="mt-3",
                                        ),
                                    ],
                                    label="Data",
                                ),
                                # Comparison tab
                                dbc.Tab(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Thermogram Comparison"),
                                                dbc.CardBody(
                                                    [
                                                        html.P(
                                                            "Compare multiple "
                                                            "thermograms:"
                                                        ),
                                                        dbc.RadioItems(
                                                            id="comparison-type",
                                                            options=[
                                                                {
                                                                    "label": "Raw Data",
                                                                    "value": "raw",
                                                                },
                                                                {
                                                                    "label": (
                                                                        "Baseline "
                                                                        "Subtracted"
                                                                    ),
                                                                    "value": "baseline",
                                                                },
                                                                {
                                                                    "label": (
                                                                        "Peak Heights"
                                                                    ),
                                                                    "value": "peaks",
                                                                },
                                                            ],
                                                            value="baseline",
                                                            className="mb-3",
                                                        ),
                                                        dcc.Graph(
                                                            id="comparison-plot",
                                                            style={"height": "500px"},
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="mb-4",
                                        ),
                                    ],
                                    label="Comparison",
                                ),
                            ]
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
        # Footer
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(),
                        html.P(
                            "© 2024 ThermogramForge Project",
                            className="text-center text-muted",
                        ),
                        html.Div(id="debug-info", style={"display": "none"}),
                    ],
                    width=12,
                )
            ]
        ),
        # Data stores
        dcc.Store(id="thermogram-data"),
        dcc.Store(id="processed-data"),
        dcc.Store(id="peaks-data"),
        dcc.Store(id="metrics-data"),
    ],
    fluid=True,
)

# Assign layout to app
app.layout = layout

# Add loading component to provide global loading feedback
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            document.body.style.cursor = 'wait';
            return "Processing...";
        } else {
            document.body.style.cursor = 'default';
            return "";
        }
    }
    """,
    Output("processing-status", "children", allow_duplicate=True),
    Input("process-button", "n_clicks"),
    prevent_initial_call=True,
)


# Callbacks


@callback(
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_upload_status(
    contents: Optional[List[str]], filenames: Optional[List[str]]
) -> Component:
    """Update upload status after files are uploaded."""
    if not contents or not filenames:
        return html.Div("No files uploaded yet.")

    return html.Div(
        [
            html.P(f"Uploaded {len(contents)} file(s):"),
            html.Ul([html.Li(filename) for filename in filenames]),
        ]
    )


@callback(
    [
        Output("thermogram-plot", "figure"),
        Output("peak-info", "children"),
        Output("metrics-info", "children"),
        Output("data-preview", "children"),
        Output("thermogram-data", "data"),
        Output("processed-data", "data"),
        Output("peaks-data", "data"),
        Output("metrics-data", "data"),
        Output("processing-status", "children"),
    ],
    Input("process-button", "n_clicks"),
    [
        State("upload-data", "contents"),
        State("upload-data", "filename"),
        State("baseline-range", "value"),
        State("processing-options", "value"),
    ],
    prevent_initial_call=True,
)
def process_thermograms(
    n_clicks: int,
    contents: Optional[List[str]],
    filenames: Optional[List[str]],
    baseline_range: List[float],
    processing_options: List[str],
) -> Tuple[
    go.Figure,
    Union[Component, str],
    Union[Component, str],
    Union[Component, List[Component]],
    Optional[Dict],
    Optional[Dict],
    Optional[Dict],
    Optional[Dict],
]:
    """Process uploaded thermograms and update results."""
    if not contents or not filenames:
        return (
            go.Figure(),
            html.Div("No data to process."),
            html.Div("No metrics available."),
            html.Div("No data available."),
            None,
            None,
            None,
            None,
        )

    # Process the first file for now
    content_type, content_string = contents[0].split(",")
    decoded = base64.b64decode(content_string)

    # Create empty figure with proper layout
    fig = go.Figure()
    fig.update_layout(
        title="Thermogram Analysis",
        xaxis_title="Temperature (°C)",
        yaxis_title="dCp (kJ/mol·K)",
        template="plotly_white",
        height=600,
    )

    try:
        # Determine file type and read data
        if filenames[0].endswith(".csv"):
            # Read CSV
            df = pl.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filenames[0].endswith((".xls", ".xlsx")):
            # Read Excel
            with tempfile.NamedTemporaryFile(suffix=filenames[0].split(".")[-1]) as tmp:
                tmp.write(decoded)
                tmp.flush()
                df = pl.read_excel(tmp.name)
        else:
            return (
                fig,
                html.Div(f"Unsupported file type: {filenames[0]}"),
                html.Div("No metrics available."),
                html.Div("No data available."),
                None,
                None,
                None,
                None,
            )

        # Check if the data has the right columns
        if "Temperature" not in df.columns or "dCp" not in df.columns:
            # Try to guess columns - first column might be temp, second might be dCp
            if len(df.columns) >= 2:
                df = df.rename({df.columns[0]: "Temperature", df.columns[1]: "dCp"})
            else:
                return (
                    fig,
                    html.Div("Data must have Temperature and dCp columns."),
                    html.Div("No metrics available."),
                    html.Div("No data available."),
                    None,
                    None,
                    None,
                    None,
                )

        # Process data based on selected options
        processed_df = df
        peaks = {}
        metrics = {}

        # Baseline subtraction
        if "auto-baseline" in processing_options:
            # Auto-detect endpoints if requested
            endpoints = detect_endpoints(df)
            lower_temp, upper_temp = endpoints.lower, endpoints.upper
        else:
            # Use manual endpoints
            lower_temp, upper_temp = baseline_range

        # Perform baseline subtraction
        baseline_result = subtract_baseline(df, lower_temp, upper_temp, use_r=False)

        # Handle the case where subtract_baseline returns either DataFrame or tuple
        if isinstance(baseline_result, tuple):
            baseline_subtracted = baseline_result[0]
        else:
            baseline_subtracted = baseline_result

        processed_df = baseline_subtracted

        # Interpolation
        if "interpolate" in processing_options:
            grid_temp = np.arange(45, 90.1, 0.1)
            interpolated = interpolate_thermogram(baseline_subtracted, grid_temp)
            processed_df = interpolated

        # Peak detection
        if "detect-peaks" in processing_options:
            detector = PeakDetector()
            peaks = detector.detect_peaks(processed_df)

        # Metrics calculation
        if "calculate-metrics" in processing_options:
            analyzer = ThermogramAnalyzer()
            metrics = analyzer.calculate_metrics(processed_df)

        # Create appropriate visualization based on processing options
        if "detect-peaks" in processing_options:
            # Use plot_with_peaks if peaks were detected
            fig = plot_with_peaks(
                processed_df, peaks, title=f"Thermogram Analysis - {filenames[0]}"
            )
        elif baseline_subtracted is not None:
            # Use plot_with_baseline if baseline subtraction was performed
            fig = plot_with_baseline(
                df,
                baseline_subtracted,
                lower_temp,
                upper_temp,
                title=f"Thermogram Analysis - {filenames[0]}",
            )
        else:
            # Use basic plot for raw data
            fig = plot_thermogram(df, title=f"Raw Thermogram - {filenames[0]}")

        # Create peak info display
        peak_info_children: Union[Component, str] = html.Div(
            "No peak information available."
        )
        if peaks:
            peak_info_table: Component = html.Table(
                # Header
                [html.Tr([html.Th(col) for col in ["Peak", "Height", "Temperature"]])] +
                # Rows
                [
                    html.Tr(
                        [
                            html.Td(peak_name),
                            html.Td(
                                f"{peak_info['peak_height']:.4f}"
                                if peak_name != "FWHM"
                                else ""
                            ),
                            html.Td(
                                f"{peak_info['peak_temp']:.2f}°C"
                                if peak_name != "FWHM"
                                else f"{peak_info['fwhm']:.2f}°C"
                            ),
                        ]
                    )
                    for peak_name, peak_info in peaks.items()
                ],
                className="table table-striped table-sm",
            )
            peak_info_children = peak_info_table

        # Create metrics info
        metrics_info_children: Union[Component, str] = html.Div(
            "No metrics information available."
        )
        if metrics:
            # Create a more detailed metrics display
            metrics_info_table = html.Table(
                # Header
                [html.Tr([html.Th("Metric"), html.Th("Value")])] +
                # Rows - exclude SampleID if present
                [
                    html.Tr(
                        [
                            html.Td(k),
                            html.Td(f"{v:.4f}" if isinstance(v, float) else str(v)),
                        ]
                    )
                    for k, v in metrics.items()
                    if k != "SampleID"
                ],
                className="table table-striped table-sm",
            )
            metrics_info_children = metrics_info_table

        # Create data preview with better formatting
        data_preview_children: Union[Component, List[Component]] = html.Div(
            [
                html.P(
                    f"Data shape: {processed_df.shape[0]} rows × "
                    f"{processed_df.shape[1]} columns"
                ),
                html.Div(
                    dbc.Table.from_dataframe(
                        processed_df.head(10).to_pandas(),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                        size="sm",
                    )
                ),
            ]
        )

        # Prepare data for storage (convert to JSON-serializable format)
        # For simplicity, we'll just store filenames and parameters here
        thermogram_data = {
            "filename": filenames[0],
            "endpoints": {"lower": lower_temp, "upper": upper_temp},
        }

        processed_data = {
            "filename": filenames[0],
            "processed": True,
            "operations": processing_options,
        }

        peaks_data = {
            "filename": filenames[0],
            "peaks": peaks,
        }

        metrics_data = {
            "filename": filenames[0],
            "metrics": metrics,
        }

        return (  # type: ignore
            fig,
            peak_info_children,
            metrics_info_children,
            data_preview_children,
            thermogram_data,
            processed_data,
            peaks_data,
            metrics_data,
            html.Div("Processing complete!", className="text-success"),
        )

    except Exception as e:
        return (  # type: ignore
            fig,
            html.Div(f"Error processing file: {str(e)}"),
            html.Div("No metrics available."),
            html.Div("No data available."),
            None,
            None,
            None,
            None,
            html.Div(f"Error: {str(e)}", className="text-danger"),
        )


@callback(
    Output("thermogram-plot", "figure", allow_duplicate=True),
    [Input("baseline-range", "value")],
    [
        State("upload-data", "contents"),
        State("upload-data", "filename"),
        State("processing-options", "value"),
    ],
    prevent_initial_call=True,
)
def update_endpoints(
    baseline_range: List[float],
    contents: Optional[List[str]],
    filenames: Optional[List[str]],
    processing_options: List[str],
) -> go.Figure:
    """Update visualization when endpoints are manually changed."""
    if not contents or not filenames:
        # Nothing to update
        return go.Figure()

    try:
        # Process the first file for now
        content_type, content_string = contents[0].split(",")
        decoded = base64.b64decode(content_string)

        # Read data
        if filenames[0].endswith(".csv"):
            df = pl.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filenames[0].endswith((".xls", ".xlsx")):
            with tempfile.NamedTemporaryFile(suffix=filenames[0].split(".")[-1]) as tmp:
                tmp.write(decoded)
                tmp.flush()
                df = pl.read_excel(tmp.name)
        else:
            # Return empty figure for unsupported file types
            return go.Figure()

        # Check and fix columns if needed
        if "Temperature" not in df.columns or "dCp" not in df.columns:
            if len(df.columns) >= 2:
                df = df.rename({df.columns[0]: "Temperature", df.columns[1]: "dCp"})
            else:
                return go.Figure()

        # Use manual endpoints
        lower_temp, upper_temp = baseline_range

        # Perform baseline subtraction with manual endpoints
        baseline_result = subtract_baseline(df, lower_temp, upper_temp, use_r=False)

        # Extract the result
        if isinstance(baseline_result, tuple):
            baseline_subtracted = baseline_result[0]
        else:
            baseline_subtracted = baseline_result

        # Create visualization
        fig = plot_with_baseline(
            df,
            baseline_subtracted,
            lower_temp,
            upper_temp,
            title=f"Manual Endpoint Adjustment - {filenames[0]}",
        )

        return fig

    except Exception as e:
        print(f"Error updating endpoints: {str(e)}")
        return go.Figure()


@callback(
    Output("comparison-plot", "figure"),
    [Input("comparison-type", "value")],
    [
        State("thermogram-data", "data"),
        State("processed-data", "data"),
        State("peaks-data", "data"),
        State("metrics-data", "data"),
    ],
    prevent_initial_call=True,
)
def update_comparison_plot(
    comparison_type: str,
    thermogram_data: Optional[Dict],
    processed_data: Optional[Dict],
    peaks_data: Optional[Dict],
    metrics_data: Optional[Dict],
) -> go.Figure:
    """Update the comparison plot based on selected comparison type."""
    # Create a basic empty figure
    fig = go.Figure()
    fig.update_layout(
        title="Thermogram Comparison",
        xaxis_title="Temperature (°C)",
        yaxis_title="dCp (kJ/mol·K)",
        template="plotly_white",
    )

    # If no data available, return empty figure
    if not thermogram_data or not processed_data:
        fig.add_annotation(
            text="No data available for comparison. Process thermograms first.",
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    # For now, just return a placeholder visualization
    # In a real implementation, we would load and display the actual data
    if comparison_type == "raw":
        fig.add_annotation(
            text="Raw thermogram comparison would be displayed here.",
            showarrow=False,
            font=dict(size=14),
        )
    elif comparison_type == "baseline":
        fig.add_annotation(
            text="Baseline-subtracted comparison would be displayed here.",
            showarrow=False,
            font=dict(size=14),
        )
    elif comparison_type == "peaks":
        if peaks_data and peaks_data.get("peaks"):
            # Create a simple bar chart of peak heights
            peaks = peaks_data.get("peaks", {})
            peak_names = []
            peak_heights = []

            for name, info in peaks.items():
                if name != "FWHM" and "peak_height" in info:
                    peak_names.append(name)
                    peak_heights.append(info["peak_height"])

            fig = go.Figure(
                data=[go.Bar(x=peak_names, y=peak_heights, marker_color="blue")]  # type: ignore
            )

            fig.update_layout(
                title="Peak Heights Comparison",
                xaxis_title="Peak",
                yaxis_title="Height (dCp)",
                template="plotly_white",
            )
        else:
            fig.add_annotation(
                text=(
                    "No peak data available. \
                        Process thermograms with peak detection first."
                ),
                showarrow=False,
                font=dict(size=14),
            )

    return fig


@callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    State("processed-data", "data"),
    prevent_initial_call=True,
)
def download_processed_data(
    n_clicks: int, processed_data: Optional[Dict]
) -> Dict[str, str]:
    """Download processed data."""
    # In a real implementation, we would retrieve the actual processed data
    # For now, just return a placeholder
    if processed_data and processed_data.get("processed"):
        filename = processed_data.get("filename", "thermogram_data")
        if not filename.endswith(".csv"):
            filename = f"{filename.split('.')[0]}_processed.csv"

        return dict(
            content="Temperature,dCp\n45.0,0.0\n45.1,0.01\n...", filename=filename
        )

    return dict(content="No processed data available", filename="no_data.txt")


def preprocess_thermogram_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess thermogram data to ensure it's suitable for analysis.

    Args:
        df: DataFrame with thermogram data

    Returns:
        Preprocessed DataFrame
    """
    # Check required columns
    if "Temperature" not in df.columns or "dCp" not in df.columns:
        # Try to guess columns - first column might be temp, second might be dCp
        if len(df.columns) >= 2:
            df = df.rename({df.columns[0]: "Temperature", df.columns[1]: "dCp"})
        else:
            raise ValueError("Data must have Temperature and dCp columns")

    # Ensure Temperature and dCp are numeric
    df = df.with_columns(
        [pl.col("Temperature").cast(pl.Float64), pl.col("dCp").cast(pl.Float64)]
    )

    # Sort by temperature
    df = df.sort("Temperature")

    # Drop any duplicate temperature values (keep first occurrence)
    df = df.unique(subset=["Temperature"], keep="first")

    # Drop any NaN or infinite values
    df = df.filter(pl.col("Temperature").is_finite() & pl.col("dCp").is_finite())

    return df


if __name__ == "__main__":
    main()
