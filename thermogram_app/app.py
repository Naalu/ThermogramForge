"""
Web application for thermogram analysis.

This module provides a Dash-based web application for uploading, processing,
and visualizing thermogram data.
"""

import base64
import io
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, State, callback, dcc, html
from dash.development.base_component import Component

import thermogram_baseline
from thermogram_baseline.baseline import subtract_baseline
from thermogram_baseline.endpoint_detection import detect_endpoints
from thermogram_baseline.interpolation import interpolate_thermogram
from tlbparam.peak_detection import PeakDetector

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ThermogramForge",
    suppress_callback_exceptions=True,
)

# App layout
app.layout = dbc.Container(
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
                                        dbc.Button(
                                            "Process Thermograms",
                                            id="process-button",
                                            color="primary",
                                            className="mt-2",
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
                                        dcc.Graph(
                                            id="thermogram-plot",
                                            style={"height": "600px"},
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
                    ],
                    width=12,
                )
            ]
        ),
    ],
    fluid=True,
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
    if not contents:
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
]:
    """Process uploaded thermograms and update results."""
    if not contents:
        return (
            go.Figure(),
            "No data to process.",
            "No metrics available.",
            "No data available.",
        )

    # Process the first file for now
    content_type, content_string = contents[0].split(",")
    decoded = base64.b64decode(content_string)

    # Determine file type and read data
    try:
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
                go.Figure(),
                f"Unsupported file type: {filenames[0]}",
                "No metrics available.",
                "No data available.",
            )

        # Check if the data has the right columns
        if "Temperature" not in df.columns or "dCp" not in df.columns:
            # Try to guess columns - first column might be temp, second might be dCp
            if len(df.columns) >= 2:
                df = df.rename({df.columns[0]: "Temperature", df.columns[1]: "dCp"})
            else:
                return (
                    go.Figure(),
                    "Data must have Temperature and dCp columns.",
                    "No metrics available.",
                    "No data available.",
                )

        # Create a figure for visualization
        fig = go.Figure()

        # Add original data
        fig.add_trace(
            go.Scatter(
                x=df.select("Temperature").to_numpy().flatten(),
                y=df.select("dCp").to_numpy().flatten(),
                mode="lines",
                name="Original Data",
                line=dict(color="blue"),
            )
        )

        # Process data based on selected options
        processed_df = df
        peaks = {}

        # Baseline subtraction
        if "auto-baseline" in processing_options:
            # Auto-detect endpoints if requested
            endpoints = detect_endpoints(df)
            lower_temp, upper_temp = endpoints.lower, endpoints.upper
        else:
            # Use manual endpoints
            lower_temp, upper_temp = baseline_range

        # Perform baseline subtraction
        baseline_subtracted = subtract_baseline(df, lower_temp, upper_temp)
        processed_df = baseline_subtracted

        # Add baseline-subtracted data
        fig.add_trace(
            go.Scatter(
                x=baseline_subtracted.select("Temperature").to_numpy().flatten(),
                y=baseline_subtracted.select("dCp").to_numpy().flatten(),
                mode="lines",
                name="Baseline Subtracted",
                line=dict(color="green"),
            )
        )

        # Interpolation
        if "interpolate" in processing_options:
            interpolated = interpolate_thermogram(baseline_subtracted)
            processed_df = interpolated

            # Add interpolated data
            fig.add_trace(
                go.Scatter(
                    x=interpolated.select("Temperature").to_numpy().flatten(),
                    y=interpolated.select("dCp").to_numpy().flatten(),
                    mode="lines",
                    name="Interpolated",
                    line=dict(color="red"),
                )
            )

        # Peak detection
        if "detect-peaks" in processing_options:
            detector = PeakDetector()
            peaks = detector.detect_peaks(processed_df)

            # Add peak markers
            for peak_name, peak_info in peaks.items():
                if peak_name != "FWHM" and peak_info["peak_height"] > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[peak_info["peak_temp"]],
                            y=[peak_info["peak_height"]],
                            mode="markers",
                            name=f"{peak_name} ({peak_info['peak_temp']:.1f}°C)",
                            marker=dict(size=10, color="red"),
                        )
                    )

        # Update plot layout
        fig.update_layout(
            title="Thermogram Analysis",
            xaxis_title="Temperature (°C)",
            yaxis_title="dCp (kJ/mol·K)",
            legend_title="Data",
            template="plotly_white",
        )

        # Create peak info display
        peak_info_children = []
        if peaks:
            peak_info_table = html.Table(
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
                                else f"{peak_info['value']:.2f}°C"
                            ),
                        ]
                    )
                    for peak_name, peak_info in peaks.items()
                ],
                className="table table-striped table-sm",
            )
            peak_info_children = [peak_info_table]
        else:
            peak_info_children = ["No peak information available."]

        # Create metrics info
        metrics_info_children = ["Advanced metrics calculation not implemented yet."]

        # Create data preview
        data_preview_children = [
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

        return (
            fig,
            peak_info_children,
            metrics_info_children,
            data_preview_children,
        )

    except Exception as e:
        return (
            go.Figure(),
            f"Error processing file: {str(e)}",
            "No metrics available.",
            "No data available.",
        )


@callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True,
)
def download_processed_data(n_clicks: int) -> Dict[str, str]:
    """Download processed data."""
    # In a real application, we would need to store the processed data in a dcc.Store
    # For this example, we'll just return a placeholder
    return dict(
        content="Temperature,dCp\n45.0,0.0\n...", filename="processed_thermogram.csv"
    )


if __name__ == "__main__":
    app.run_server(debug=True)
