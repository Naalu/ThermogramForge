"""
Main layout definition for the ThermogramForge app.
"""

import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import dcc, html

# Import the app instance
from app.app import app
from app.components import data_preview_card, thermogram_card

# Define the layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                [
                    html.H1("ThermogramForge", className="my-4"),
                    html.P("Thermogram Analysis Tool", className="lead"),
                    html.Hr(),
                ],
                width=12,
            )
        ),
        # Main content
        dbc.Row(
            [
                # Left column - Upload and controls
                dbc.Col(
                    [
                        # File Upload Card
                        dbc.Card(
                            [
                                dbc.CardHeader("Data Input"),
                                dbc.CardBody(
                                    [
                                        # dash-uploader component
                                        html.Div(
                                            [
                                                html.H5("Upload Thermogram File"),
                                                html.P("Supported formats: CSV, Excel"),
                                                du.Upload(
                                                    id="dash-uploader",
                                                    text="Drag and Drop or Click to Upload",
                                                    text_completed="Uploaded: ",
                                                    cancel_button=True,
                                                    max_file_size=1024 * 20,  # 20 MB
                                                    filetypes=["csv", "xlsx", "xls"],
                                                    upload_id=None,  # auto-generated
                                                    default_style={
                                                        "width": "100%",
                                                        "height": "60px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "1px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "5px",
                                                        "textAlign": "center",
                                                        "margin": "10px 0",
                                                    },
                                                ),
                                                html.Div(
                                                    id="upload-status", className="mt-2"
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                        # Baseline Controls Card
                        dbc.Card(
                            [
                                dbc.CardHeader("Baseline Subtraction"),
                                dbc.CardBody(
                                    [
                                        html.P("Select baseline endpoints:"),
                                        dcc.RangeSlider(
                                            id="baseline-range",
                                            min=45,
                                            max=90,
                                            step=0.5,
                                            marks={
                                                i: f"{i}°C" for i in range(45, 91, 5)
                                            },
                                            value=[55, 85],
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                            className="mb-4",
                                        ),
                                        dbc.Button(
                                            "Apply Baseline Subtraction",
                                            id="apply-baseline",
                                            color="primary",
                                            className="mb-3",
                                        ),
                                        html.Div(id="baseline-info", className="mt-3"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=4,
                ),
                # Right column - Results and visualization
                dbc.Col(
                    [
                        # Loading component wrapping all visualization elements
                        dcc.Loading(
                            id="loading-output-id",
                            type="circle",
                            children=[
                                # Thermogram plot
                                thermogram_card(),
                                # Data preview
                                data_preview_card(),
                            ],
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
        # Data stores
        dcc.Store(id="thermogram-data"),
        # Download component
        dcc.Download(id="download-data"),
    ],
    fluid=True,
)
