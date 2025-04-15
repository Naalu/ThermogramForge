"""
Callbacks related to file uploads and initial data processing.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple  # Added types

import dash_bootstrap_components as dbc
import pandas as pd
from dash import (  # Added ctx, ALL, MATCH just in case
    Input,
    Output,
    State,
    callback,
    ctx,
    html,
    no_update,
)

from app import UPLOAD_FOLDER_ROOT
from app.utils.data_processing import extract_samples
from core.baseline import EndpointSelectionMethod, find_spline_endpoints

# Ensure core baseline functions are imported

# Set up logger
logger = logging.getLogger(__name__)


@callback(
    # Store Outputs
    Output("baseline-params", "data", allow_duplicate=True),
    Output("all-samples-data", "data", allow_duplicate=True),
    # UI Outputs
    Output("upload-status", "children", allow_duplicate=True),
    # Inputs
    Input("dash-uploader", "isCompleted"),
    # States (Fetch stores to update them)
    State("baseline-params", "data"),
    State("all-samples-data", "data"),
    State("dash-uploader", "fileNames"),
    State("dash-uploader", "upload_id"),
    State("min-temp-input", "value"),
    State("max-temp-input", "value"),
    State("lower-exclusion-input", "value"),
    State("upper-exclusion-input", "value"),
    State("point-selection-method", "value"),
    prevent_initial_call=True,
)
def process_upload_multi_sample(
    # Input Args
    is_completed: bool,
    # State Args
    store_baseline_params: Optional[Dict[str, Any]],
    store_all_samples_data: Optional[Dict[str, Any]],
    filenames: Optional[List[str]],
    upload_id: Optional[str],
    min_temp: Optional[float],
    max_temp: Optional[float],
    lower_exclusion: Optional[float],
    upper_exclusion: Optional[float],
    selection_method: Optional[str],
) -> Tuple[  # Define return tuple types
    Optional[Dict[str, Any]],  # baseline-params store
    Optional[Dict[str, Any]],  # all-samples-data store
    Any,  # upload-status children
]:
    """Processes an uploaded thermogram file (CSV or Excel).

    Extracts samples, filters by temperature, calculates initial baseline parameters,
    updates the main data stores (`all-samples-data`, `baseline-params`).
    Only performs processing when triggered by the dash-uploader component completion.

    Args:
        is_completed: Flag from dash-uploader indicating if upload finished.
        store_baseline_params: Current state of the baseline parameters store.
        store_all_samples_data: Current state of the raw sample data store.
        filenames: List of uploaded filenames from dash-uploader.
        upload_id: Upload ID from dash-uploader.
        min_temp: Minimum temperature selected in the modal.
        max_temp: Maximum temperature selected in the modal.
        lower_exclusion: Lower exclusion temperature from modal (unused currently).
        upper_exclusion: Upper exclusion temperature from modal (unused currently).
        selection_method: Point selection method from modal (unused currently).

    Returns:
        A tuple containing:
            - Updated baseline parameters dictionary for the store.
            - Updated raw sample data dictionary for the store.
            - Content for the upload status alert.

        Returns `no_update` for store outputs if not triggered by uploader or on error.

    """
    triggered_input_id = ctx.triggered_id
    logger.info(f"process_upload_multi_sample triggered by: {triggered_input_id}")

    # --- Trigger Check --- Start
    if triggered_input_id != "dash-uploader":
        logger.debug(
            f"Callback triggered by '{triggered_input_id}', not uploader. No processing."
        )
        return no_update, no_update, no_update
    # --- Trigger Check --- End

    # Initialize return values
    updated_baseline_store = store_baseline_params or {}
    updated_all_samples_store = store_all_samples_data or {}
    status_message = no_update

    # --- Uploader Completion Check & Input Validation --- Start
    if not is_completed or not filenames or not upload_id:
        logger.warning("Uploader triggered but not completed or missing filenames/ID.")
        return (
            no_update,
            no_update,
            dbc.Alert("Upload incomplete or file info missing.", color="warning"),
        )

    try:
        min_temp_val = float(min_temp) if min_temp is not None else -float("inf")
        max_temp_val = float(max_temp) if max_temp is not None else float("inf")
        if min_temp_val >= max_temp_val:
            raise ValueError("Min temperature must be less than Max temperature.")
    except (TypeError, ValueError) as e:
        error_msg = f"Invalid temperature range input: {e}"
        logger.error(error_msg)
        return no_update, no_update, dbc.Alert(error_msg, color="danger")
    # --- Uploader Completion Check & Input Validation --- End

    try:
        # Process the most recently uploaded file
        filename = filenames[-1]
        file_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id, filename)
        logger.info(f"Processing file: {file_path}")

        if not os.path.exists(file_path):
            error_msg = f"Error: Uploaded file not found at {file_path}"
            logger.error(error_msg)
            return no_update, no_update, dbc.Alert(error_msg, color="danger")

        # Read file based on extension
        if filename.lower().endswith(".csv"):
            df_full = pd.read_csv(file_path)
        elif filename.lower().endswith((".xls", ".xlsx")):
            df_full = pd.read_excel(file_path)
        else:
            error_msg = f"Unsupported file type: {filename}"
            logger.warning(error_msg)
            return no_update, no_update, dbc.Alert(error_msg, color="warning")

        # Extract and Filter Samples
        logger.info(f"Extracting samples from {filename} (shape {df_full.shape})")
        samples_extracted = extract_samples(df_full)
        if not samples_extracted:
            error_msg = f"Could not identify any valid sample data columns (e.g., T[ID]/ID pairs) in file: {filename}. Please check the file format."
            logger.error(error_msg)
            return no_update, no_update, dbc.Alert(error_msg, color="danger")

        samples_filtered: Dict[str, List[Dict[str, Any]]] = {}
        initial_baseline_params_for_file: Dict[str, Dict[str, Any]] = {}
        num_samples_in_file: int = 0

        for sample_id, df_sample in samples_extracted.items():
            # df_sample is already preprocessed (sorted, numeric) by extract_samples
            if "Temperature" in df_sample.columns:
                df_filtered = df_sample[
                    (df_sample["Temperature"] >= min_temp_val)
                    & (df_sample["Temperature"] <= max_temp_val)
                ].copy()

                if not df_filtered.empty:
                    # Convert df to list of dicts for storage
                    samples_filtered[sample_id] = df_filtered.to_dict("records")
                    num_samples_in_file += 1
                    logger.info(
                        f"  Sample '{sample_id}' filtered: {len(df_filtered)} points [{min_temp_val:.1f}-{max_temp_val:.1f}°C] stored."
                    )

                    # Calculate default baseline params using the new core function
                    try:
                        # Get parameters from state for endpoint detection
                        # Ensure defaults if states are None
                        lower_excl_val = (
                            float(lower_exclusion)
                            if lower_exclusion is not None
                            else 60.0
                        )
                        upper_excl_val = (
                            float(upper_exclusion)
                            if upper_exclusion is not None
                            else 80.0
                        )
                        try:
                            point_sel_method_enum = EndpointSelectionMethod(
                                selection_method
                                or EndpointSelectionMethod.INNERMOST.value
                            )
                        except ValueError:
                            logger.warning(
                                f"Invalid point selection method '{selection_method}', defaulting to Innermost."
                            )
                            point_sel_method_enum = EndpointSelectionMethod.INNERMOST

                        logger.debug(
                            f"Finding endpoints for '{sample_id}' with Excl: {lower_excl_val}-{upper_excl_val}, Method: {point_sel_method_enum.value}"
                        )
                        auto_endpoints = find_spline_endpoints(
                            df=df_filtered,  # Pass the filtered dataframe
                            lower_exclusion_temp=lower_excl_val,
                            upper_exclusion_temp=upper_excl_val,
                            # window_size=10, # Use default or make configurable?
                            # spline_smooth_factor=None, # Use default
                            point_selection_method=point_sel_method_enum,
                        )

                        initial_baseline_params_for_file[sample_id] = {
                            "lower": auto_endpoints.get("lower"),
                            "upper": auto_endpoints.get("upper"),
                            "lower_source": "auto",
                            "upper_source": "auto",
                            "reviewed": False,
                            "exclude": False,
                        }
                        logger.debug(
                            f"  Found auto params for '{sample_id}': L={auto_endpoints.get('lower'):.2f}, U={auto_endpoints.get('upper'):.2f}"
                        )
                    except Exception as e_param:
                        logger.error(
                            f"Error finding auto endpoints for '{sample_id}': {e_param}",
                            exc_info=True,
                        )
                        # Store None if detection fails
                        initial_baseline_params_for_file[sample_id] = {
                            "lower": None,
                            "upper": None,
                            "lower_source": "error",
                            "upper_source": "error",
                            "reviewed": False,
                            "exclude": False,
                        }
                else:
                    logger.warning(
                        f"  Sample '{sample_id}' was empty after temperature filtering. Skipping."
                    )
            else:
                logger.warning(
                    f"Sample '{sample_id}' missing 'Temperature' column after extraction. Skipping."
                )

        if not samples_filtered:
            error_msg = f"No samples remained after temperature filtering ({min_temp_val:.1f}-{max_temp_val:.1f}°C) in {filename}."
            logger.warning(error_msg)
            return no_update, no_update, dbc.Alert(error_msg, color="warning")

        # --- Update Stores --- Start
        # Make copies to avoid modifying state directly before returning
        current_baseline_params = (store_baseline_params or {}).copy()
        current_all_samples_data = (store_all_samples_data or {}).copy()

        # Store samples AND metadata under the filename key
        upload_metadata = {
            "min_temp": min_temp,
            "max_temp": max_temp,
            "lower_exclusion": lower_exclusion,
            "upper_exclusion": upper_exclusion,
            "selection_method": selection_method,
            "original_filename": filename,  # Store original name too
            "processed_at": datetime.now().isoformat(),  # Add processing timestamp
        }
        current_baseline_params[filename] = initial_baseline_params_for_file
        current_all_samples_data[filename] = {
            "samples": samples_filtered,
            "metadata": upload_metadata,
        }
        logger.debug(f"Storing data and metadata for {filename}: {upload_metadata}")

        updated_baseline_store = current_baseline_params
        updated_all_samples_store = current_all_samples_data
        # --- Update Stores --- End

        logger.info(
            f"Successfully processed {filename}: {num_samples_in_file} samples filtered."
        )
        status_message = dbc.Alert(
            f"Successfully processed {filename}: {num_samples_in_file} samples found and filtered.",
            color="success",
            dismissable=True,
            duration=5000,
        )

    except pd.errors.EmptyDataError:
        error_msg = f"Error: File {filename} is empty."
        logger.error(error_msg)
        status_message = dbc.Alert(error_msg, color="danger")
        updated_baseline_store = no_update
        updated_all_samples_store = no_update
    except Exception as e:
        error_msg = f"An error occurred during processing of {filename}: {str(e)}"
        logger.error(error_msg, exc_info=True)  # Log full traceback
        status_message = dbc.Alert(error_msg, color="danger")
        updated_baseline_store = no_update
        updated_all_samples_store = no_update
    finally:
        # Clean up the uploaded file and folder if they exist
        # Ensure cleanup happens even if processing fails mid-way
        if (
            "upload_id" in locals()
            and "filename" in locals()
            and upload_id
            and filename
        ):
            cleanup_upload_folder(upload_id, filename)
        elif upload_id:  # If filename wasn't assigned due to early exit
            folder_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)
            if os.path.exists(folder_path) and not os.listdir(folder_path):
                try:
                    os.rmdir(folder_path)
                    logger.info(
                        f"Cleaned up empty upload folder (early exit case): {folder_path}"
                    )
                except OSError as e_rmdir:
                    logger.error(f"Error removing directory {folder_path}: {e_rmdir}")

    return (
        updated_baseline_store,
        updated_all_samples_store,
        status_message,
    )


def build_raw_data_overview(
    all_samples_data: Optional[Dict[str, Any]],
    _baseline_params: Optional[Dict[str, Any]],
    processed_datasets_store: Optional[Dict[str, Any]],
) -> List[dbc.ListGroupItem]:
    """Builds the list group item display for the raw data overview.

    Args:
        all_samples_data: The store containing raw data and metadata keyed by filename.
        _baseline_params: The store containing initial baseline params (unused for status now).
        processed_datasets_store: Store containing final processed data keyed by generated name.

    Returns:
        A list of dbc.ListGroupItem components.
    """
    if not all_samples_data:
        return [dbc.ListGroupItem("No raw files uploaded yet.")]

    items = []
    processed_raw_files = set()
    if processed_datasets_store:
        for data in processed_datasets_store.values():
            if isinstance(data, dict) and "source_raw_file" in data:
                processed_raw_files.add(data["source_raw_file"])

    for filename, file_data in sorted(all_samples_data.items()):
        if not isinstance(file_data, dict):
            logger.warning(f"Skipping invalid entry in all_samples_data for {filename}")
            continue

        metadata = file_data.get("metadata", {})
        min_temp = metadata.get("min_temp", "N/A")
        max_temp = metadata.get("max_temp", "N/A")
        lower_excl = metadata.get("lower_exclusion", "N/A")
        upper_excl = metadata.get("upper_exclusion", "N/A")
        sel_method = metadata.get("selection_method", "N/A")

        is_processed = filename in processed_raw_files
        status_badge = dbc.Badge(
            "Processed" if is_processed else "Not Processed",
            color="success" if is_processed else "secondary",
            className="ms-2 align-self-center fs-6",
        )

        review_button = dbc.Button(
            "Review Endpoints",
            id={"type": "review-btn", "index": filename},
            color="info",
            outline=False,
            size="sm",
            className="me-2",
        )

        metadata_spans = [
            html.Small(f"Range: {min_temp}-{max_temp}°C", className="text-muted me-3"),
            html.Small(
                f"Excl: {lower_excl}-{upper_excl}°C", className="text-muted me-3"
            ),
            html.Small(f"Method: {sel_method}", className="text-muted"),
        ]
        metadata_display = html.Div(metadata_spans, className="mt-1")

        item_content = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Strong(filename), width=True, className="text-truncate"
                        ),
                        dbc.Col(status_badge, width="auto"),
                        dbc.Col(
                            [review_button], width="auto", className="flex-shrink-0"
                        ),
                    ],
                    className="d-flex align-items-center justify-content-between w-100",
                ),
                metadata_display,
            ]
        )

        items.append(dbc.ListGroupItem(item_content))

    return items if items else [dbc.ListGroupItem("No valid raw files found.")]


def cleanup_upload_folder(upload_id: str, filename: str) -> None:
    """Removes the specific uploaded file and its parent folder."""
    if not upload_id or not filename:  # Basic check
        return
    try:
        file_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id, filename)
        folder_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)

        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up uploaded file: {file_path}")

        # Check if folder is empty before removing
        if os.path.exists(folder_path) and not os.listdir(folder_path):
            os.rmdir(folder_path)
            logger.info(f"Cleaned up empty upload folder: {folder_path}")
        elif os.path.exists(folder_path):
            # Folder might contain other files if multiple uploads happened quickly
            # Or potentially debug files. Only log if not removing.
            logger.debug(f"Upload folder not empty, not removing: {folder_path}")

    except OSError as e:
        logger.error(f"Error cleaning up upload folder {upload_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during upload cleanup: {e}", exc_info=True)


# --- NEW Callback to Update Raw Data Overview UI ---
@callback(
    Output("raw-data-overview-display", "children"),
    Input("all-samples-data", "data"),
    Input("baseline-params", "data"),
    Input("processed-datasets-store", "data"),
    prevent_initial_call=True,
)
def update_raw_data_overview_ui(
    all_samples_data: Optional[Dict[str, Any]],
    baseline_params: Optional[Dict[str, Any]],
    processed_datasets_store: Optional[Dict[str, Any]],
) -> List[dbc.ListGroupItem]:
    """Updates the list group display for raw files when data stores change."""
    logger.info("Updating raw data overview UI due to store change.")
    # Call the existing helper function, now passing the processed data store
    return build_raw_data_overview(
        all_samples_data, baseline_params, processed_datasets_store
    )
