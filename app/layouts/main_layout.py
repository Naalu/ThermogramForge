"""
Main layout definition for the ThermogramForge application.

Defines the structure and components of the user interface using Dash Bootstrap Components
and Dash Core Components. Includes the main tab structure, data overview section,
review endpoints section (with grid, plots, control panel), upload modal, and data stores.
"""

# --- Add logger --- Start
import logging

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import Dash, dcc, html  # Import Dash for type hinting

# Import the new modal component
from app.components.upload_processed_modal import create_upload_processed_modal

logger = logging.getLogger(__name__)  # Add logger for consistency
# --- Add logger --- End

# """


def create_layout() -> dbc.Container:
    """Creates the main layout container with all UI elements.

    Defines the overall structure including header, tabs (Overview, Review),
    modals, and necessary dcc.Store components for holding application state.

    Returns:
        dbc.Container: The root Dash Bootstrap Container for the app layout.
    """
    layout = dbc.Container(
        [
            # --- Main Content Area (Full Width) --- Start
            dbc.Row(
                dbc.Col(
                    [
                        # Header Section
                        html.Div(
                            [
                                html.H2("ThermogramForge", className="mb-0"),
                                # Future: Add theme switch? User info?
                            ],
                            className="d-flex justify-content-between align-items-center p-4 bg-white border-bottom",
                            style={"border-bottom": "2px solid #dee2e6"},
                        ),
                        # Main Tabs Section
                        dbc.Tabs(
                            id="main-tabs",
                            active_tab="tab-overview",
                            children=[
                                # 1. Data Overview Tab
                                dbc.Tab(
                                    label="Data Overview",
                                    tab_id="tab-overview",
                                    children=[
                                        dbc.Container(
                                            [
                                                html.H3(
                                                    "Data Overview",
                                                    className="mt-4 mb-3",
                                                ),
                                                # --- Section: Raw Data --- Redesigned ---
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            # Use Row to place Title and Button side-by-side
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        html.H4(
                                                                            "Raw Thermogram Files",
                                                                            className="mb-0 align-middle",  # Keep vertical align
                                                                        ),
                                                                        width=True,  # Allow title col to take available space
                                                                    ),
                                                                    dbc.Col(
                                                                        dbc.Button(
                                                                            "Upload New Raw Thermogram Data",
                                                                            id="open-upload-modal-btn-overview",
                                                                            color="primary",
                                                                            size="sm",  # Smaller button in header
                                                                        ),
                                                                        width="auto",  # Fit button width
                                                                    ),
                                                                ],
                                                                align="center",  # Vertically align items
                                                                justify="between",  # Push items apart
                                                            )
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                dbc.Alert(
                                                                    id="upload-status",
                                                                    className="mb-3",
                                                                    is_open=False,
                                                                    dismissable=True,
                                                                ),
                                                                # Container for the styled list of raw data files
                                                                dbc.Spinner(
                                                                    dbc.ListGroup(
                                                                        id="raw-data-overview-display",
                                                                        flush=True,
                                                                        children=[
                                                                            dbc.ListGroupItem(
                                                                                "No raw files uploaded yet."
                                                                            )
                                                                        ],  # Placeholder
                                                                    ),
                                                                    color="primary",
                                                                    size="sm",
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-4",
                                                ),
                                                # --- Section: Processed Data --- Redesigned ---
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            # Use Row for Title and Button
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        html.H4(
                                                                            "Processed Thermograms",  # RENAMED Title
                                                                            className="mb-0 align-middle",
                                                                        ),
                                                                        width=True,
                                                                    ),
                                                                    dbc.Col(
                                                                        dbc.Button(
                                                                            "Upload New Processed Thermogram Data",  # RENAMED Button Text
                                                                            id="open-upload-processed-modal-btn",
                                                                            color="primary",
                                                                            size="sm",  # Smaller button
                                                                            # REMOVED outline=True
                                                                            # REMOVED disabled=True
                                                                        ),
                                                                        width="auto",
                                                                    ),
                                                                ],
                                                                align="center",
                                                                justify="between",
                                                            )
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                # REMOVED Button Row from Body
                                                                # Container for the styled list of processed datasets
                                                                dbc.Spinner(
                                                                    dbc.ListGroup(
                                                                        id="processed-data-overview-display",
                                                                        flush=True,
                                                                        children=[
                                                                            dbc.ListGroupItem(
                                                                                "No data processed yet."
                                                                            )
                                                                        ],  # Placeholder
                                                                    ),
                                                                    color="primary",
                                                                    size="sm",
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-4",
                                                ),
                                                # --- Section: Reports --- Redesigned ---
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            html.H4(
                                                                "Reports Generated",
                                                                className="mb-0",
                                                            )
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                # Container for the styled list of reports
                                                                dbc.Spinner(
                                                                    dbc.ListGroup(
                                                                        id="reports-overview-display",
                                                                        flush=True,
                                                                        children=[
                                                                            dbc.ListGroupItem(
                                                                                "No reports generated yet."
                                                                            )
                                                                        ],  # Placeholder
                                                                    ),
                                                                    color="primary",
                                                                    size="sm",
                                                                )
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            fluid=True,
                                            className="p-4 px-md-5",
                                        )
                                    ],  # End Data Overview Tab children
                                ),  # End Data Overview Tab
                                # 2. Review Endpoints Tab
                                dbc.Tab(
                                    label="Review Endpoints",
                                    tab_id="tab-review",
                                    children=[
                                        dbc.Container(
                                            [
                                                html.H3(
                                                    "Review Sample Endpoints",
                                                    className="mt-4 mb-3",
                                                ),
                                                # Dataset Selector Card
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            "Select Dataset for Review"
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(
                                                                            [
                                                                                dcc.Dropdown(
                                                                                    id="review-dataset-selector",
                                                                                    placeholder="Select an uploaded raw dataset...",
                                                                                    options=[],
                                                                                )
                                                                            ],
                                                                            width=True,
                                                                        ),
                                                                        dbc.Col(
                                                                            [
                                                                                dbc.Button(
                                                                                    "Upload New Data",
                                                                                    id="open-upload-modal-btn-review",
                                                                                    color="primary",
                                                                                    outline=False,
                                                                                    size=None,
                                                                                )
                                                                            ],
                                                                            width="auto",
                                                                        ),
                                                                    ],
                                                                    className="align-items-center mb-2",
                                                                ),  # Add margin below row
                                                                dbc.Alert(
                                                                    id="review-selector-message",
                                                                    is_open=False,
                                                                    dismissable=True,
                                                                    color="info",
                                                                ),  # Make dismissable
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-4",
                                                ),
                                                # Main Review Area (Grid, Plots, Controls)
                                                # Initially hidden, shown by callback when dataset selected
                                                html.Div(
                                                    id="review-content-area",
                                                    style={"display": "none"},
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                # Left Column: Control Panel & Grid
                                                                dbc.Col(
                                                                    [
                                                                        # Control Panel Card
                                                                        dbc.Card(
                                                                            [
                                                                                dbc.CardHeader(
                                                                                    "Review Samples"
                                                                                ),
                                                                                dbc.CardBody(
                                                                                    [
                                                                                        html.H5(
                                                                                            "Select a sample from the grid below",
                                                                                            id="selected-sample-display",
                                                                                        ),
                                                                                        # Collapsible controls section
                                                                                        html.Div(
                                                                                            [
                                                                                                dbc.Row(
                                                                                                    [
                                                                                                        # Lower Endpoint Display
                                                                                                        dbc.Col(
                                                                                                            [
                                                                                                                dbc.Label(
                                                                                                                    "Lower Endpoint (°C):"
                                                                                                                ),
                                                                                                                html.Div(
                                                                                                                    id="display-lower-endpoint",
                                                                                                                    children="N/A",
                                                                                                                ),
                                                                                                                dbc.FormText(
                                                                                                                    id="lower-endpoint-source",
                                                                                                                    children="Source: auto",
                                                                                                                ),
                                                                                                            ],
                                                                                                            width=6,
                                                                                                        ),
                                                                                                        # Upper Endpoint Display
                                                                                                        dbc.Col(
                                                                                                            [
                                                                                                                dbc.Label(
                                                                                                                    "Upper Endpoint (°C):"
                                                                                                                ),
                                                                                                                html.Div(
                                                                                                                    id="display-upper-endpoint",
                                                                                                                    children="N/A",
                                                                                                                ),
                                                                                                                dbc.FormText(
                                                                                                                    id="upper-endpoint-source",
                                                                                                                    children="Source: auto",
                                                                                                                ),
                                                                                                            ],
                                                                                                            width=6,
                                                                                                        ),
                                                                                                    ],
                                                                                                    className="mb-3",
                                                                                                ),
                                                                                                # Endpoint Selection Buttons
                                                                                                dbc.Row(
                                                                                                    [
                                                                                                        dbc.Col(
                                                                                                            dbc.Button(
                                                                                                                "Manually Adjust Lower Endpoint",
                                                                                                                id="select-lower-btn",
                                                                                                                outline=True,
                                                                                                                color="primary",
                                                                                                                size="sm",
                                                                                                                className="me-1",
                                                                                                                disabled=True,
                                                                                                            ),
                                                                                                            width="auto",
                                                                                                        ),
                                                                                                        dbc.Col(
                                                                                                            dbc.Button(
                                                                                                                "Manually Adjust Upper Endpoint",
                                                                                                                id="select-upper-btn",
                                                                                                                outline=True,
                                                                                                                color="primary",
                                                                                                                size="sm",
                                                                                                                className="me-1",
                                                                                                                disabled=True,
                                                                                                            ),
                                                                                                            width="auto",
                                                                                                        ),
                                                                                                    ],
                                                                                                    className="mb-3",
                                                                                                ),
                                                                                                # Exclude Checkbox
                                                                                                dbc.Row(
                                                                                                    [
                                                                                                        dbc.Col(
                                                                                                            dbc.Checkbox(
                                                                                                                id="edit-exclude-checkbox",
                                                                                                                label="Exclude this Sample",
                                                                                                                value=False,
                                                                                                                disabled=True,
                                                                                                            ),
                                                                                                            width="auto",
                                                                                                        )
                                                                                                    ],
                                                                                                    className="mb-3",
                                                                                                ),
                                                                                                # Action Buttons
                                                                                                dbc.Row(
                                                                                                    [
                                                                                                        dbc.Col(
                                                                                                            dbc.Button(
                                                                                                                "Discard Changes",
                                                                                                                id="cancel-sample-changes-btn",
                                                                                                                color="secondary",
                                                                                                                outline=True,
                                                                                                                disabled=True,
                                                                                                            ),
                                                                                                            width="auto",
                                                                                                        ),
                                                                                                        dbc.Col(
                                                                                                            [
                                                                                                                dbc.Button(
                                                                                                                    "Previous Sample",
                                                                                                                    id="previous-sample-btn",
                                                                                                                    color="secondary",
                                                                                                                    outline=True,
                                                                                                                    disabled=True,
                                                                                                                    className="me-2",
                                                                                                                ),
                                                                                                                dbc.Button(
                                                                                                                    "Mark Reviewed & Next",
                                                                                                                    id="mark-reviewed-next-btn",
                                                                                                                    color="info",
                                                                                                                    disabled=True,
                                                                                                                ),
                                                                                                            ],
                                                                                                            className="ms-auto d-flex",
                                                                                                        ),
                                                                                                    ],
                                                                                                    className="mb-3",
                                                                                                ),
                                                                                                # Alert Area
                                                                                                dbc.Alert(
                                                                                                    id="control-panel-alert",
                                                                                                    children="",
                                                                                                    color="info",
                                                                                                    is_open=False,
                                                                                                    duration=4000,
                                                                                                    className="mt-3",
                                                                                                    dismissable=True,
                                                                                                ),
                                                                                            ],
                                                                                            id="sample-control-panel-content",
                                                                                            style={
                                                                                                "display": "none"
                                                                                            },  # Initially hidden until a sample is selected
                                                                                        ),
                                                                                    ]
                                                                                ),  # End Control Panel CardBody
                                                                            ],
                                                                            className="mb-4",
                                                                        ),  # Add margin below control panel
                                                                        # Sample Overview Grid Card
                                                                        dbc.Card(
                                                                            [
                                                                                dbc.CardHeader(
                                                                                    dbc.Row(
                                                                                        [
                                                                                            dbc.Col(
                                                                                                html.H5(
                                                                                                    "Sample Overview",
                                                                                                    className="mb-0",
                                                                                                ),
                                                                                                width="auto",
                                                                                            ),
                                                                                            dbc.Col(
                                                                                                dbc.Button(
                                                                                                    "Save Processed Data",
                                                                                                    id="save-processed-data-btn",
                                                                                                    color="success",
                                                                                                    size="sm",  # Smaller button
                                                                                                    outline=True,
                                                                                                ),
                                                                                                width="auto",
                                                                                            ),
                                                                                        ],
                                                                                        justify="between",
                                                                                        align="center",
                                                                                    )
                                                                                ),
                                                                                dbc.CardBody(
                                                                                    [
                                                                                        # AG Grid Component
                                                                                        dag.AgGrid(
                                                                                            id="sample-grid",
                                                                                            rowData=[],
                                                                                            columnDefs=[
                                                                                                {
                                                                                                    "headerName": "Sample ID",
                                                                                                    "field": "sample_id",
                                                                                                    "filter": True,
                                                                                                    "sortable": True,
                                                                                                    "width": 150,
                                                                                                },
                                                                                                {
                                                                                                    "headerName": "Lower (°C)",
                                                                                                    "field": "lower",
                                                                                                    "type": "numericColumn",
                                                                                                    "valueFormatter": {
                                                                                                        "function": "params.value == null ? '' : d3.format('.1f')(params.value) + ' (' + params.data.lower_source + ')'"
                                                                                                    },
                                                                                                    "width": 120,
                                                                                                },
                                                                                                {
                                                                                                    "headerName": "Upper (°C)",
                                                                                                    "field": "upper",
                                                                                                    "type": "numericColumn",
                                                                                                    "valueFormatter": {
                                                                                                        "function": "params.value == null ? '' : d3.format('.1f')(params.value) + ' (' + params.data.upper_source + ')'"
                                                                                                    },
                                                                                                    "width": 120,
                                                                                                },
                                                                                                {
                                                                                                    "headerName": "Reviewed",
                                                                                                    "field": "reviewed",
                                                                                                    "cellRenderer": "CheckboxRenderer",
                                                                                                },  # Assuming CheckboxRenderer defined in assets
                                                                                                {
                                                                                                    "headerName": "Exclude",
                                                                                                    "field": "exclude",
                                                                                                    "cellRenderer": "CheckboxRenderer",
                                                                                                },
                                                                                            ],
                                                                                            dashGridOptions={
                                                                                                "rowSelection": "single",
                                                                                                "pagination": True,
                                                                                                "paginationPageSize": 10,
                                                                                                "paginationPageSizeSelector": False,
                                                                                                "domLayout": "autoHeight",
                                                                                                "suppressRowClickSelection": False,
                                                                                            },
                                                                                            columnSize="sizeToFit",
                                                                                            style={
                                                                                                "width": "100%",
                                                                                                "height": "auto",
                                                                                                "minHeight": "350px",
                                                                                            },
                                                                                        ),
                                                                                    ]
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                    width=5,
                                                                    className="d-flex flex-column",
                                                                ),
                                                                # Right Column: Plots
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Card(
                                                                            [
                                                                                dbc.CardHeader(
                                                                                    dbc.Tabs(
                                                                                        [
                                                                                            dbc.Tab(
                                                                                                label="Baseline Subtracted",
                                                                                                tab_id="tab-processed",
                                                                                            ),
                                                                                            dbc.Tab(
                                                                                                label="Raw Thermogram",
                                                                                                tab_id="tab-raw",
                                                                                            ),
                                                                                        ],
                                                                                        id="plot-tabs",
                                                                                        active_tab="tab-processed",
                                                                                    )
                                                                                ),
                                                                                dbc.CardBody(
                                                                                    [
                                                                                        # Raw Plot Area (Initially hidden)
                                                                                        html.Div(
                                                                                            dcc.Loading(
                                                                                                dcc.Graph(
                                                                                                    id="raw-plot-graph",
                                                                                                    style={
                                                                                                        "height": "45vh"
                                                                                                    },
                                                                                                )
                                                                                            ),
                                                                                            id="raw-plot-content",
                                                                                            style={
                                                                                                "display": "none"
                                                                                            },
                                                                                            className="flex-grow-1",
                                                                                        ),
                                                                                        # Processed Plot Area (Initially visible)
                                                                                        html.Div(
                                                                                            dcc.Loading(
                                                                                                dcc.Graph(
                                                                                                    id="processed-plot-graph",
                                                                                                    style={
                                                                                                        "height": "45vh"
                                                                                                    },
                                                                                                )
                                                                                            ),
                                                                                            id="processed-plot-content",
                                                                                            style={
                                                                                                "display": "block"
                                                                                            },
                                                                                            className="flex-grow-1",
                                                                                        ),
                                                                                    ],
                                                                                    className="d-flex flex-column",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                    width=7,
                                                                    className="d-flex flex-column",
                                                                ),
                                                            ]
                                                        )  # End Row for review grid/plots
                                                    ],
                                                ),  # End review-content-area div
                                            ],
                                            fluid=True,
                                            className="p-4 px-md-5",
                                        )  # End Review Tab Container
                                    ],  # End Review Tab children
                                ),  # End Review Tab
                                # 3. Report Builder Tab --- NEW ---
                                dbc.Tab(
                                    label="Report Builder",
                                    tab_id="tab-report-builder",  # New ID
                                    children=[
                                        dbc.Container(
                                            [
                                                html.H3(
                                                    "Report Builder",
                                                    className="mt-4 mb-3",
                                                ),
                                                # --- Dataset Selector --- Start
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            "Select Dataset for Report"
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Dropdown(
                                                                    id="report-dataset-selector",
                                                                    placeholder="Select a processed dataset...",
                                                                    options=[],  # Populated by callback
                                                                ),
                                                                dbc.Alert(
                                                                    id="report-selector-message",
                                                                    color="info",
                                                                    is_open=False,
                                                                    className="mt-3",
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-4",
                                                ),
                                                # --- Report Builder Content (Initially Hidden) --- Start
                                                html.Div(
                                                    id="report-builder-content",
                                                    style={"display": "none"},
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                # --- Left Column: Config & Metrics --- Start
                                                                dbc.Col(
                                                                    [
                                                                        # Report Configuration Card
                                                                        dbc.Card(
                                                                            [
                                                                                dbc.CardHeader(
                                                                                    "Report Configuration"
                                                                                ),
                                                                                dbc.CardBody(
                                                                                    [
                                                                                        dbc.Label(
                                                                                            "Report Name:"
                                                                                        ),
                                                                                        dbc.Input(
                                                                                            id="report-name-input",
                                                                                            value="Thermogram_Analysis_Report",
                                                                                            type="text",
                                                                                            className="mb-3",
                                                                                        ),
                                                                                        dbc.Label(
                                                                                            "Report Format:"
                                                                                        ),
                                                                                        dcc.Dropdown(
                                                                                            id="report-format-dropdown",
                                                                                            options=[
                                                                                                {
                                                                                                    "label": "CSV (Comma Separated Values)",
                                                                                                    "value": "csv",
                                                                                                },
                                                                                                {
                                                                                                    "label": "Excel Workbook (.xlsx)",
                                                                                                    "value": "xlsx",
                                                                                                },
                                                                                            ],
                                                                                            value="csv",
                                                                                            clearable=False,
                                                                                        ),
                                                                                    ]
                                                                                ),
                                                                            ],
                                                                            className="mb-4",
                                                                        ),
                                                                        # Select Metrics Card
                                                                        dbc.Card(
                                                                            [
                                                                                dbc.CardHeader(
                                                                                    html.H5(
                                                                                        "Select Metrics"
                                                                                    ),
                                                                                ),
                                                                                dbc.CardBody(
                                                                                    [
                                                                                        # Add Control Buttons
                                                                                        dbc.ButtonGroup(
                                                                                            [
                                                                                                dbc.Button(
                                                                                                    "Select All",
                                                                                                    id="report-metric-select-all",
                                                                                                    color="light",
                                                                                                    size="sm",
                                                                                                    className="me-1",
                                                                                                ),
                                                                                                dbc.Button(
                                                                                                    "Clear All",
                                                                                                    id="report-metric-clear-all",
                                                                                                    color="light",
                                                                                                    size="sm",
                                                                                                    className="me-1",
                                                                                                ),
                                                                                                dbc.Button(
                                                                                                    "Reset Selection",
                                                                                                    id="report-metric-reset",
                                                                                                    color="light",
                                                                                                    size="sm",
                                                                                                ),
                                                                                            ],
                                                                                            className="mb-2",  # Add margin below buttons
                                                                                        ),
                                                                                        # Metric Checklist
                                                                                        dbc.Checklist(
                                                                                            id="report-metric-selector",
                                                                                            options=[],  # Populated by callback
                                                                                            value=[],  # Default selected metrics set by callback
                                                                                            labelStyle={
                                                                                                "display": "block"
                                                                                            },  # Display options vertically
                                                                                        ),
                                                                                        # Add container for tooltips
                                                                                        html.Div(
                                                                                            id="report-metric-tooltips-div"
                                                                                        ),
                                                                                    ]
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                # --- Left Column: Config & Metrics --- End
                                                                # --- Right Column: Preview & Generate --- Start
                                                                dbc.Col(
                                                                    [
                                                                        # --- Report Preview Section --- Dynamically Updated
                                                                        dbc.Card(
                                                                            [
                                                                                dbc.CardHeader(
                                                                                    # Row for Title and Button
                                                                                    dbc.Row(
                                                                                        [
                                                                                            dbc.Col(
                                                                                                html.H4(
                                                                                                    "Report Preview",
                                                                                                    className="mb-0",  # Remove bottom margin if needed
                                                                                                ),
                                                                                                width=True,  # Allow title to take space
                                                                                            ),
                                                                                            dbc.Col(
                                                                                                dbc.Button(
                                                                                                    "Generate Report",
                                                                                                    id="generate-report-button",
                                                                                                    color="success",
                                                                                                    disabled=True,  # Initially disabled
                                                                                                    # n_clicks=0, # REMOVE - n_clicks is input only
                                                                                                    size="sm",  # Optional: smaller button
                                                                                                ),
                                                                                                width="auto",  # Fit button width
                                                                                            ),
                                                                                        ],
                                                                                        justify="between",  # Push items apart
                                                                                        align="center",  # Vertically align items
                                                                                    )
                                                                                ),
                                                                                dbc.CardBody(
                                                                                    [
                                                                                        dbc.Spinner(
                                                                                            html.Div(  # Wrap table preview for spinner
                                                                                                id="report-preview-table-div",
                                                                                                children=html.Em(
                                                                                                    "Select a dataset and metrics to see a preview."
                                                                                                ),
                                                                                            ),
                                                                                            color="primary",
                                                                                            size="sm",
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ],
                                                                            # className="mt-4", # REMOVED margin top
                                                                        ),
                                                                    ],
                                                                    width=8,  # Set width 8 for right column
                                                                ),
                                                                # --- Right Column: Preview & Generate --- End
                                                            ]
                                                        )  # End Row
                                                    ],
                                                ),
                                                # --- Report Builder Content (Initially Hidden) --- End
                                            ],
                                            fluid=True,
                                            className="p-4 px-md-5",
                                        )
                                    ],  # End Report Builder Tab children
                                ),  # End Report Builder Tab
                            ],  # End Main Tabs Children
                            className="main-tabs-container mt-3",
                        ),  # End Main Tabs
                    ],  # End Main Column Children
                    width=12,
                    className="p-0 d-flex flex-column",
                    style={"height": "100vh", "overflowY": "auto"},
                )  # End Main Column
            ),  # End Main Content Row
            # --- Main Content Area --- End
            # --- Upload Modal ---
            # (Keep as is, already reviewed)
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle("Data Input Settings"), close_button=True
                    ),
                    dbc.ModalBody(
                        [
                            # Upload Component
                            html.Div(
                                [
                                    html.H5("Upload Thermogram File"),
                                    html.P("Supported formats: CSV, Excel"),
                                    du.Upload(
                                        id="dash-uploader",
                                        text="Drag and Drop or Click to Upload",
                                        text_completed="Uploaded: ",
                                        cancel_button=True,
                                        max_file_size=1024 * 20,
                                        filetypes=["csv", "xlsx", "xls"],
                                        upload_id=None,
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
                                ],
                                className="mb-4",
                            ),
                            # Temperature Range
                            html.Div(
                                [
                                    html.H5("Temperature Range (°C)"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Min Temp:"),
                                                    dbc.Input(
                                                        id="min-temp-input",
                                                        type="number",
                                                        value=45,
                                                        step=0.1,
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Max Temp:"),
                                                    dbc.Input(
                                                        id="max-temp-input",
                                                        type="number",
                                                        value=90,
                                                        step=0.1,
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Advanced Options
                            html.Div(
                                [
                                    dbc.Checklist(
                                        options=[
                                            {
                                                "label": "Show Advanced Options",
                                                "value": 1,
                                            }
                                        ],
                                        value=[],
                                        id="advanced-options-toggle",
                                        switch=True,
                                    ),
                                    dbc.Collapse(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label(
                                                                        "Lower Exclusion Temp (°C):"
                                                                    ),
                                                                    dbc.Input(
                                                                        id="lower-exclusion-input",
                                                                        type="number",
                                                                        value=60,
                                                                        step=0.1,
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label(
                                                                        "Upper Exclusion Temp (°C):"
                                                                    ),
                                                                    dbc.Input(
                                                                        id="upper-exclusion-input",
                                                                        type="number",
                                                                        value=80,
                                                                        step=0.1,
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                    dbc.Label(
                                                        "Automatic Point Selection Method:"
                                                    ),
                                                    dbc.RadioItems(
                                                        options=[
                                                            {
                                                                "label": "Innermost",
                                                                "value": "Innermost",
                                                            },
                                                            {
                                                                "label": "Outermost",
                                                                "value": "Outermost",
                                                            },
                                                            {
                                                                "label": "Middle",
                                                                "value": "Middle",
                                                            },
                                                        ],
                                                        value="Innermost",
                                                        id="point-selection-method",
                                                        inline=True,
                                                    ),
                                                ]
                                            )
                                        ),
                                        id="advanced-options-collapse",
                                        is_open=False,
                                        className="mt-3",
                                    ),
                                ],
                                className="mt-4",
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close",
                            id="close-upload-modal-btn",
                            className="ms-auto",
                            n_clicks=0,
                        )
                    ),
                ],
                id="upload-modal",
                is_open=False,
                size="lg",
                backdrop="static",
            ),
            # --- Upload Modal --- End
            # Add the new processed data upload modal
            create_upload_processed_modal(),
            # --- Data Stores ---
            # (Keep as is, already reviewed)
            dcc.Store(id="all-samples-data", data={}),
            dcc.Store(id="baseline-params", data={}),
            dcc.Store(id="processed-data"),
            dcc.Store(id="temporary-baseline-params"),
            dcc.Store(id="endpoint-selection-mode", data=None),
            dcc.Store(id="grid-action-trigger", data=None),
            dcc.Store(id="select-first-row-trigger", data=0),
            dcc.Download(id="download-data"),
            dcc.Store(id="processed-datasets-store", data={}),
            # Add store for temporary processed upload data
            dcc.Store(id="upload-processed-temp-store"),
            # Add store for report builder intermediate results
            dcc.Store(id="report-calculated-metrics-temp"),
            # Add download component for reports
            dcc.Download(id="download-report"),
            # Add store for generated report metadata
            dcc.Store(id="generated-reports-store", data={}),
            # --- Data Stores --- End
        ],  # End Root Container Children
        fluid=True,
        className="vh-100 d-flex flex-column p-0",
        style={"overflow": "hidden"},  # Prevent scrolling on the main container itself
    )
    return layout


def register_layout(app: Dash):  # Add type hint for app
    """Registers the layout with the Dash app instance.

    Assigns the layout created by `create_layout()` to the `app.layout` attribute.

    Args:
        app (Dash): The Dash application instance.
    """
    app.layout = create_layout()
    logger.info("Main layout registered with app")  # Use logger


# Note: Removed the print statement, assuming logger is configured in main.py
