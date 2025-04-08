"""
Simple and robust file upload callbacks for multi-sample data.
"""

import base64
import io
import json
import traceback
from collections import OrderedDict

import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from app.components import create_data_preview, create_thermogram_figure


def parse_contents(contents, filename):
    """
    Parse uploaded file contents into a DataFrame.

    Args:
        contents: Content string from dcc.Upload
        filename: Name of the uploaded file

    Returns:
        pandas.DataFrame or error message
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if filename.lower().endswith(".csv"):
            # Parse CSV
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.lower().endswith((".xls", ".xlsx")):
            # Parse Excel
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div([f"Unsupported file type: {filename}"])

        return df
    except Exception as e:
        print(f"Error parsing {filename}: {str(e)}")
        return html.Div(
            [
                "Error processing this file:",
                html.Pre(str(e) + "\n" + traceback.format_exc()),
            ]
        )


def extract_samples(df):
    """
    Extract individual samples from a DataFrame with paired columns.

    The function looks for column pairs like:
    - T[SampleID] (Temperature column)
    - [SampleID] (dCp column)

    Args:
        df: DataFrame with raw data

    Returns:
        Dictionary mapping sample IDs to DataFrames with Temperature and dCp columns
    """
    try:
        # Find all columns that start with 'T' (Temperature columns)
        temp_cols = [col for col in df.columns if col.startswith("T")]

        samples = OrderedDict()

        # For each temperature column, try to find the corresponding value column
        for temp_col in temp_cols:
            # The value column should be the same as the temp column without the 'T'
            value_col = temp_col[1:]

            # Check if the value column exists
            if value_col in df.columns:
                # Create a new DataFrame for this sample with Temperature and dCp columns
                sample_df = pd.DataFrame(
                    {"Temperature": df[temp_col], "dCp": df[value_col]}
                )

                # Clean data: remove NaNs, sort by temperature
                sample_df = sample_df.dropna()
                sample_df = sample_df.sort_values("Temperature")
                sample_df = sample_df.reset_index(drop=True)

                # Add to the samples dictionary
                samples[value_col] = sample_df

        if not samples:
            # No sample pairs found, check if the DataFrame already has Temperature and dCp columns
            if "Temperature" in df.columns and "dCp" in df.columns:
                # Clean data
                df = df.dropna(subset=["Temperature", "dCp"])
                df = df.sort_values("Temperature")
                df = df.reset_index(drop=True)

                # Create a generic sample ID if none exists
                sample_id = "Sample1"
                if "SampleID" in df.columns:
                    # Use the first sample ID found
                    sample_id = str(df["SampleID"].iloc[0])

                samples[sample_id] = df[["Temperature", "dCp"]]

        return samples

    except Exception as e:
        print(f"Error extracting samples: {str(e)}")
        return None


@callback(
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def update_upload_status(contents, filename):
    """Update upload status after file is uploaded."""
    if not contents or not filename:
        return html.Div("No file uploaded yet.")

    return html.Div(
        [
            html.P(f"Uploaded: {filename}", className="mb-1"),
            html.Div(id="processing-status"),
        ]
    )


@callback(
    [
        Output("thermogram-data", "data"),
        Output("sample-dropdown-container", "children"),
        Output("thermogram-plot", "figure"),
        Output("data-preview", "children"),
        Output("processing-status", "children"),
    ],
    Input("upload-status", "children"),
    [State("upload-data", "contents"), State("upload-data", "filename")],
    prevent_initial_call=True,
)
def process_upload(upload_status, contents, filename):
    """Process the uploaded thermogram file and update the display."""
    if not contents or not filename:
        raise PreventUpdate

    # Parse file contents
    df = parse_contents(contents, filename)

    # Check if parsing returned an error message
    if isinstance(df, html.Div):
        return (
            None,
            html.Div(),
            {},
            df,
            html.Div("Error processing file", style={"color": "red"}),
        )

    # Extract samples
    samples = extract_samples(df)

    if not samples:
        return (
            None,
            html.Div(),
            {},
            html.Div("No valid samples found in the file"),
            html.Div("No valid samples found", style={"color": "red"}),
        )

    try:
        # Create sample dropdown
        dropdown = dcc.Dropdown(
            id="sample-dropdown",
            options=[
                {"label": f"Sample: {sample_id}", "value": sample_id}
                for sample_id in samples.keys()
            ],
            value=list(samples.keys())[0],  # Select first sample by default
            clearable=False,
            className="mb-2",
        )

        # Get the first sample for initial display
        first_sample_id = list(samples.keys())[0]
        first_sample = samples[first_sample_id]

        # Create visualization
        fig = create_thermogram_figure(
            first_sample, title=f"Thermogram: {first_sample_id} (from {filename})"
        )

        # Create data preview
        preview = create_data_preview(first_sample)

        # Store the data - ensure it's JSON serializable
        # Store all samples in a structured format
        samples_json = {}
        for sample_id, sample_df in samples.items():
            samples_json[sample_id] = sample_df.to_json(
                date_format="iso", orient="split"
            )

        data_json = json.dumps(
            {
                "filename": filename,
                "samples": samples_json,
                "active_sample": first_sample_id,
            }
        )

        dropdown_container = html.Div(
            [
                html.Label("Select Sample:"),
                dropdown,
                html.Div(
                    f"Found {len(samples)} samples in the uploaded file",
                    className="text-muted small",
                ),
            ]
        )

        return (
            data_json,
            dropdown_container,
            fig,
            preview,
            html.Div(f"Processed {len(samples)} samples", style={"color": "green"}),
        )

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        error_msg = html.Div(
            [
                "Error creating visualization:",
                html.Pre(str(e) + "\n" + traceback.format_exc()),
            ]
        )
        return (
            None,
            html.Div(),
            {},
            error_msg,
            html.Div("Error in visualization", style={"color": "red"}),
        )


@callback(
    [
        Output("thermogram-plot", "figure", allow_duplicate=True),
        Output("data-preview", "children", allow_duplicate=True),
        Output("thermogram-data", "data", allow_duplicate=True),
    ],
    Input("sample-dropdown", "value"),
    State("thermogram-data", "data"),
    prevent_initial_call=True,
)
def update_selected_sample(sample_id, data_json):
    """Update the display when a different sample is selected."""
    if not sample_id or not data_json:
        raise PreventUpdate

    try:
        # Parse the data JSON
        from io import StringIO

        data = json.loads(data_json)

        # Get the selected sample data
        sample_json = data["samples"][sample_id]
        sample_df = pd.read_json(StringIO(sample_json), orient="split")

        # Create visualization
        fig = create_thermogram_figure(
            sample_df, title=f"Thermogram: {sample_id} (from {data['filename']})"
        )

        # Create data preview
        preview = create_data_preview(sample_df)

        # Update the active sample in the data store
        data["active_sample"] = sample_id
        updated_data_json = json.dumps(data)

        return fig, preview, updated_data_json

    except Exception as e:
        print(f"Error updating sample: {str(e)}")
        # If there's an error, don't update anything
        raise PreventUpdate
