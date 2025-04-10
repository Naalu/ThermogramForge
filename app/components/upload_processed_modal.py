import dash_bootstrap_components as dbc
from dash import dcc, html


def create_upload_processed_modal() -> dbc.Modal:
    """Creates the modal for uploading processed thermogram data."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Upload Processed Thermogram Data")),
            dbc.ModalBody(
                [
                    html.P(
                        "Select a processed thermogram file (.csv or .xlsx) to upload."
                    ),
                    dcc.Upload(
                        id="upload-processed-data-component",
                        children=html.Div(
                            ["Drag and Drop or ", html.A("Select Files")]
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
                        # Accept CSV and Excel files
                        accept=".csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel",
                        multiple=False,  # Allow only single file upload
                    ),
                    # Input for Excel sheet name (conditionally shown)
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Select Sheet (Excel)"),
                            dcc.Dropdown(
                                id="upload-processed-sheet-dropdown",
                                placeholder="Select sheet...",
                                options=[],
                                value=None,
                                disabled=True,
                                clearable=False,
                            ),
                        ],
                        className="mb-3",
                        style={"display": "none"},
                        id="upload-processed-sheet-name-group",
                    ),
                    html.Div(id="upload-processed-validation-message"),
                    html.Hr(),
                    html.H5("Data Preview"),
                    # Use DataTable for preview
                    html.Div(
                        id="upload-processed-preview-div",  # Wrapper div for table/message
                        children=[
                            html.Em("Select a file to see a preview (first 5 rows).")
                        ],
                        style={
                            "maxHeight": "300px",
                            "overflowY": "auto",
                            "marginBottom": "10px",
                        },
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="upload-processed-cancel-button",
                        color="secondary",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Submit Data",
                        id="upload-processed-submit-button",
                        color="primary",
                        n_clicks=0,
                        disabled=True,  # Initially disabled
                    ),
                ]
            ),
        ],
        id="upload-processed-modal",
        is_open=False,
        size="lg",  # Larger modal for preview table
    )
