"""
Main layout definition for the ThermogramForge app.
"""

import dash_bootstrap_components as dbc
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
                                        # Simple Upload component
                                        html.Div(
                                            [
                                                html.H5("Upload Thermogram File"),
                                                html.P("Supported formats: CSV, Excel"),
                                                dcc.Upload(
                                                    id="upload-data",
                                                    children=html.Div(
                                                        [
                                                            "Drag and Drop or ",
                                                            html.A("Click to Upload"),
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
                                                    multiple=False,
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
                        # Thermogram plot
                        thermogram_card(),
                        # Data preview
                        data_preview_card(),
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
