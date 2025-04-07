"""
Callbacks for handling file uploads with dash-uploader.
"""

import os
import time

import dash_bootstrap_components as dbc
import dash_uploader as du
import pandas as pd
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from app.app import UPLOAD_FOLDER_ROOT, app
from app.components import create_data_preview, create_thermogram_figure
from app.utils import preprocess_thermogram_data

# Configure dash-uploader
du.configure_upload(app, UPLOAD_FOLDER_ROOT)


@callback(
    Output("upload-status", "children"),
    Output("loading-output-id", "children", allow_duplicate=True),
    Input("dash-uploader", "isCompleted"),
    State("dash-uploader", "fileNames"),
    State("dash-uploader", "upload_id"),
    prevent_initial_call=True,
)
def upload_complete(is_completed, filenames, upload_id):
    """Handle the uploaded file once it's complete."""
    if not is_completed or not filenames:
        return html.Div("Upload a thermogram file to begin"), None

    # Get the most recent filename
    filename = filenames[-1]
    file_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id, filename)

    # Display a message during processing
    processing_msg = html.Div(
        [dbc.Spinner(size="sm", color="primary", type="grow"), " Processing file..."]
    )

    return dbc.Alert(f"Uploaded: {filename}", color="success"), processing_msg


@callback(
    Output("thermogram-data", "data"),
    Output("thermogram-plot", "figure"),
    Output("data-preview", "children"),
    Input("upload-status", "children"),
    State("dash-uploader", "fileNames"),
    State("dash-uploader", "upload_id"),
    prevent_initial_call=True,
)
def process_upload(upload_status, filenames, upload_id):
    """Process the uploaded thermogram file and update the display."""
    # Skip if no upload has happened
    if not filenames or not upload_id:
        raise PreventUpdate

    try:
        # Get the most recent filename
        filename = filenames[-1]
        file_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id, filename)

        # Add a small delay to ensure file is fully written
        time.sleep(0.5)

        # Parse file content based on extension
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return None, {}, html.Div("Unsupported file type")

        # Preprocess data
        df = preprocess_thermogram_data(df)

        # Create visualization
        fig = create_thermogram_figure(df, title=f"Thermogram: {filename}")

        # Create data preview
        preview = create_data_preview(df)

        # Store the data
        data = df.to_json(date_format="iso", orient="split")

        return data, fig, preview

    except Exception as e:
        # If there's an error in processing, display it
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)
        return None, {}, html.Div(error_msg)
