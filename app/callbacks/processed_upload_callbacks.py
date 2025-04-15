import base64
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import pandas as pd
from dash import (
    Input,
    Output,
    State,
    callback,
    callback_context,
    dash_table,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)


# --- FORMAT DETECTION HELPER --- Start
def detect_and_validate_format(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """Detects if DataFrame is wide or long format and validates accordingly.

    Args:
        df: The input DataFrame read from the file/sheet.

    Returns:
        Tuple containing:
            - Validated DataFrame (or None if invalid).
            - Detected format string ('wide' or 'long') or None.
            - Error message string or None.
    """
    logger.info("Attempting to detect and validate DataFrame format.")
    df_original = df.copy()  # Keep original for long format check if wide fails

    # --- Try Standard Wide Format Validation --- Start
    e_wide_std = "Standard Wide validation skipped or passed."
    try:
        logger.debug("Attempting WIDE format validation.")
        temp_df_wide = df.copy()
        logger.debug(f"WIDE Check: Initial df shape: {temp_df_wide.shape}")
        logger.debug(f"WIDE Check: Initial columns: {temp_df_wide.columns.tolist()}")
        logger.debug(f"WIDE Check: Initial index: {temp_df_wide.index}")

        # 1. Identify SampleID/SampleCode (Index or First Column)
        if (
            temp_df_wide.index.name == "SampleID"
            or temp_df_wide.index.name == "SampleCode"
        ):
            logger.debug(f"WIDE Check: Identifier is index '{temp_df_wide.index.name}'")
            temp_df_wide.index.name = "SampleID"
        elif temp_df_wide.shape[1] > 0 and (  # Check column exists before accessing
            temp_df_wide.columns[0].lower() in ["sampleid", "samplecode"]
        ):
            original_col_name = temp_df_wide.columns[0]
            logger.debug(
                f"WIDE Check: Identifier is first column '{original_col_name}'"
            )
            try:
                temp_df_wide = temp_df_wide.set_index(original_col_name)
                temp_df_wide.index.name = "SampleID"
                logger.debug(
                    f"WIDE Check: df shape after set_index: {temp_df_wide.shape}"
                )
                logger.debug(
                    f"WIDE Check: columns after set_index: {temp_df_wide.columns.tolist()}"
                )
            except Exception as e_set_index:
                logger.error(
                    f"WIDE Check: Error during set_index('{original_col_name}'): {e_set_index}",
                    exc_info=True,
                )
                raise ValueError(
                    f"Failed to set index using column '{original_col_name}'."
                )
        elif temp_df_wide.shape[1] > 0:  # Check column exists before assuming
            # Assume first column if unnamed
            original_col_name = temp_df_wide.columns[0]
            logger.warning(
                f"Assuming first column ('{original_col_name}') is sample identifier for wide format."
            )
            try:
                temp_df_wide = temp_df_wide.set_index(original_col_name)
                temp_df_wide.index.name = "SampleID"
                logger.debug(
                    f"WIDE Check: df shape after assuming and set_index: {temp_df_wide.shape}"
                )
                logger.debug(
                    f"WIDE Check: columns after assuming and set_index: {temp_df_wide.columns.tolist()}"
                )
            except Exception as e_set_index:
                logger.error(
                    f"WIDE Check: Error during set_index on assumed col '{original_col_name}': {e_set_index}",
                    exc_info=True,
                )
                raise ValueError(
                    f"Failed to set index using assumed column '{original_col_name}'."
                )
        else:
            # No columns to treat as SampleID or Data
            raise ValueError(
                "DataFrame has no columns to identify as SampleID or Data."
            )

        if temp_df_wide.index.isnull().any():
            raise ValueError(
                "SampleID/SampleCode column/index contains missing values."
            )

        # ---> MOVE Temperature Column Check AFTER Index is Set <---
        # 2. Check Temperature Columns (must be numeric headers)
        logger.debug(
            f"WIDE Check: About to check columns: {temp_df_wide.columns.tolist()}"
        )
        if len(temp_df_wide.columns) == 0:
            logger.error(
                "WIDE Check: Validation failed - length of columns is 0 after index set."
            )  # Specific log
            raise ValueError(
                "No data columns found after identifying SampleID/SampleCode."
            )

        num_data_columns = len(temp_df_wide.columns)
        invalid_temp_cols = []
        for col in temp_df_wide.columns:
            try:
                pd.to_numeric(col)  # Attempt conversion
            except (ValueError, TypeError):
                invalid_temp_cols.append(str(col))

        # If more than half the columns are non-numeric, assume it's a metrics report
        if len(invalid_temp_cols) > num_data_columns / 2:
            logger.error(
                f"WIDE Check: Validation failed - Headers resemble metrics, not temperatures: {invalid_temp_cols}"
            )
            raise ValueError(
                "Invalid column headers. File appears to be a metrics report, not wide-format thermogram data. Expected numeric temperature headers."
            )
        elif invalid_temp_cols:  # If *any* column is invalid (stricter, optional)
            # This case handles files that are mixed or have a few bad headers
            logger.warning(
                f"Found some non-numeric column headers: {invalid_temp_cols}"
            )
            raise ValueError(
                f"Invalid column headers. Expected numeric temperatures, but got non-numeric: {', '.join(invalid_temp_cols)}"
            )
        # ---> END MOVED BLOCK <---

        # 3. Check dCp Values (must be numeric data)
        try:
            temp_df_wide = temp_df_wide.astype(float)
            # NaNs might be acceptable, depending on requirements. Pass for now.
        except (ValueError, TypeError) as e_val:
            raise ValueError(
                f"Non-numeric dCp data found. Ensure all values are numbers. Hint: {e_val}"
            )

        logger.info("Standard WIDE format validation successful.")
        return temp_df_wide, "wide", None  # Return validated wide df

    except Exception as wide_exception:
        e_wide_std = wide_exception  # Capture the actual exception
        logger.debug(f"Standard WIDE format validation failed: {e_wide_std}")
        pass
    # --- Try Standard Wide Format Validation --- End

    # --- Try TRANSPOSED Wide Format Validation --- Start
    e_wide_transposed = "Transposed Wide validation skipped or passed."
    try:
        logger.debug("Attempting TRANSPOSED WIDE format validation.")
        temp_df_transposed = df_original.copy()

        # 1. Identify Temperature Column (Index or First Column)
        temp_col_header = None
        if "temp" in str(temp_df_transposed.index.name).lower():
            logger.debug("TRANSPOSED Check: Temperature found as index.")
            temp_df_transposed.index.name = "Temperature"  # Standardize
        elif temp_df_transposed.shape[1] > 0 and (
            "temp" in str(temp_df_transposed.columns[0]).lower()
        ):
            temp_col_header = temp_df_transposed.columns[0]
            logger.debug(
                f"TRANSPOSED Check: Temperature likely first column '{temp_col_header}'"
            )
            try:
                temp_df_transposed = temp_df_transposed.set_index(temp_col_header)
                temp_df_transposed.index.name = "Temperature"
                logger.debug(
                    f"TRANSPOSED Check: df shape after setting Temp index: {temp_df_transposed.shape}"
                )
            except Exception as e_set_idx:
                raise ValueError(
                    f"Failed setting Temp index '{temp_col_header}': {e_set_idx}"
                )
        else:
            raise ValueError(
                "Cannot identify Temperature column/index for transposed format."
            )

        # 2. Check Temperature Index is Numeric
        if not pd.api.types.is_numeric_dtype(temp_df_transposed.index):
            try:
                temp_df_transposed.index = pd.to_numeric(temp_df_transposed.index)
            except (ValueError, TypeError):
                raise ValueError(
                    "Temperature index/column contains non-numeric values."
                )
        if temp_df_transposed.index.isnull().any():
            raise ValueError("Temperature index/column contains missing values.")

        # 3. Check SampleID Columns Headers (remaining columns)
        if len(temp_df_transposed.columns) == 0:
            raise ValueError("No SampleID columns found after identifying Temperature.")
        # Check if column headers look like Sample IDs (simplistic check: not purely numeric)
        non_string_headers = [
            h for h in temp_df_transposed.columns if pd.api.types.is_numeric_dtype(h)
        ]
        if non_string_headers:
            logger.warning(
                f"TRANSPOSED Check: Found numeric-like column headers treated as SampleIDs: {non_string_headers}"
            )
            # Allow numeric sample IDs for now, but log a warning.

        # 4. Check dCp values are numeric
        try:
            # Apply astype to data columns only
            for col in temp_df_transposed.columns:
                temp_df_transposed[col] = pd.to_numeric(temp_df_transposed[col])
        except (ValueError, TypeError) as e_val:
            raise ValueError(f"Non-numeric dCp data found in cells. Hint: {e_val}")

        # 5. TRANSPOSE to standard wide format
        logger.info(
            "TRANSPOSED WIDE format validation successful. Transposing to standard format."
        )
        validated_wide_df = temp_df_transposed.transpose()
        validated_wide_df.index.name = "SampleID"  # Set correct index name
        logger.debug(f"Final shape after transpose: {validated_wide_df.shape}")

        return validated_wide_df, "wide", None  # Return standard wide df

    except Exception as transposed_exception:
        e_wide_transposed = transposed_exception
        logger.debug(f"TRANSPOSED WIDE format validation failed: {e_wide_transposed}")
        pass  # Continue to long format check
    # --- Try TRANSPOSED Wide Format Validation --- End

    # --- Try Long Format Validation --- Start
    e_long = "Long format validation skipped or passed."
    try:
        logger.debug("Attempting LONG format validation.")
        temp_df_long = df_original.copy()  # Use the original df
        required_long_cols = ["Temperature", "dCp_subtracted"]
        sample_id_col_options = ["SampleID", "SampleCode", "sampleid", "samplecode"]
        actual_sample_id_col = None

        # Find SampleID/SampleCode column
        for col in sample_id_col_options:
            if col in temp_df_long.columns:
                actual_sample_id_col = col
                break

        if not actual_sample_id_col:
            raise ValueError("Missing required column: 'SampleID' or 'SampleCode'.")

        # Check other required columns
        missing_cols = [
            col for col in required_long_cols if col not in temp_df_long.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for long format: {', '.join(missing_cols)}. Expected SampleID/Code, Temperature, dCp_subtracted."
            )

        # Standardize SampleID column name
        if actual_sample_id_col != "SampleID":
            temp_df_long = temp_df_long.rename(
                columns={actual_sample_id_col: "SampleID"}
            )

        # Check numeric types
        numeric_cols_to_check = ["Temperature", "dCp_subtracted"]
        numeric_errors = []
        for col in numeric_cols_to_check:
            if not pd.api.types.is_numeric_dtype(temp_df_long[col]):
                try:
                    temp_df_long[col] = pd.to_numeric(temp_df_long[col])
                except (ValueError, TypeError):
                    numeric_errors.append(col)
        if numeric_errors:
            raise ValueError(
                f"Data type error in long format. Columns must be numeric: {', '.join(numeric_errors)}."
            )

        # Check for missing values in essential columns
        if (
            temp_df_long["SampleID"].isnull().any()
            or temp_df_long["Temperature"].isnull().any()
        ):
            raise ValueError(
                "'SampleID' or 'Temperature' column contains missing values."
            )

        logger.info("LONG format validation successful.")
        # Select only the required columns in standard order for consistency
        final_long_df = temp_df_long[["SampleID", "Temperature", "dCp_subtracted"]]
        return final_long_df, "long", None  # Return validated long df

    except Exception as long_exception:
        e_long = long_exception
        logger.debug(f"LONG format validation failed: {e_long}")
        # All formats failed
        return (
            None,
            None,
            f"File does not match expected format. Std Wide Error: {e_wide_std} | Transposed Wide Error: {e_wide_transposed} | Long Error: {e_long}",
        )
    # --- Try Long Format Validation --- End


# --- FORMAT DETECTION HELPER --- End


@callback(
    Output("upload-processed-modal", "is_open"),
    Input("open-upload-processed-modal-btn", "n_clicks"),
    Input("upload-processed-cancel-button", "n_clicks"),
    Input(
        "upload-processed-submit-button", "n_clicks"
    ),  # Close on successful submit too
    State("upload-processed-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_upload_processed_modal(n_open, n_cancel, n_submit, is_open):
    """Opens and closes the processed data upload modal."""
    # Check which button triggered the callback
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "open-upload-processed-modal-btn":
        logger.info("Opening processed upload modal.")
        return True
    elif trigger_id in [
        "upload-processed-cancel-button",
        "upload-processed-submit-button",
    ]:
        # We'll handle the actual submit logic elsewhere, just close the modal here
        logger.info(f"Closing processed upload modal via {trigger_id}.")
        return False

    return is_open  # Keep current state if no relevant button clicked


# --- File Upload, Parse, Validate, Preview Callback --- REFACTORED for Dropdown
@callback(
    # Existing Outputs
    Output("upload-processed-preview-div", "children"),
    Output("upload-processed-validation-message", "children"),
    Output("upload-processed-submit-button", "disabled"),
    Output("upload-processed-sheet-name-group", "style"),  # Keep controlling visibility
    Output("upload-processed-temp-store", "data"),
    # New Outputs for Dropdown
    Output("upload-processed-sheet-dropdown", "options"),
    Output("upload-processed-sheet-dropdown", "value"),
    Output("upload-processed-sheet-dropdown", "disabled"),
    # --- INPUTS --- Trigger on file upload OR dropdown selection
    Input("upload-processed-data-component", "contents"),
    Input("upload-processed-sheet-dropdown", "value"),  # Trigger on dropdown change
    # --- STATES ---
    State("upload-processed-data-component", "filename"),
    State("upload-processed-sheet-dropdown", "options"),  # Get current options
    # Remove n_submit/n_blur inputs for text input
    prevent_initial_call=True,
)
def handle_processed_file_upload(
    contents: Optional[str],
    selected_sheet_value: Optional[str],  # Value from sheet dropdown
    filename: Optional[str],
    current_dropdown_options: List[Dict[str, str]],  # Current options state
) -> Tuple[
    Any,
    Any,
    bool,
    Dict[str, str],
    Optional[Dict[str, Any]],
    List[Dict[str, str]],
    Optional[str],
    bool,
]:
    """Handles file upload, populates Excel sheet dropdown, validates selected sheet,
    detects format, shows preview, and enables submit."""
    ctx = callback_context
    triggered_id = ctx.triggered_id
    if not triggered_id:
        raise PreventUpdate

    # --- Reset Outputs --- Start
    preview_content: Any = html.Em("Processing...")  # Default preview
    validation_message: Any = None
    submit_disabled: bool = True
    sheet_group_style: Dict[str, str] = {"display": "none"}
    temp_store_data: Optional[Dict[str, Any]] = None
    dropdown_options: List[Dict[str, str]] = []  # Default empty options
    dropdown_value: Optional[str] = None  # Default no selection
    dropdown_disabled: bool = True  # Default disabled
    df: Optional[pd.DataFrame] = None
    run_validation = False
    # --- Reset Outputs --- End

    # --- Determine Workflow based on Trigger --- Start

    # Scenario 1: File Uploaded
    if triggered_id == "upload-processed-data-component":
        logger.info(f"File uploaded: {filename}")
        # Reset dropdown value if a new file is uploaded
        dropdown_value = None
        # Reset temp store data
        temp_store_data = None

        if not contents or not filename:
            logger.debug("File upload triggered but no content/filename.")
            preview_content = html.Em("Select a file.")
            # Return default reset state
            return (
                preview_content,
                validation_message,
                submit_disabled,
                sheet_group_style,
                temp_store_data,
                dropdown_options,
                dropdown_value,
                dropdown_disabled,
            )

        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # CSV Upload
            if filename.lower().endswith(".csv"):
                logger.info("Processing CSV file.")
                sheet_group_style = {"display": "none"}
                dropdown_options = []
                dropdown_disabled = True
                try:
                    # --> REVERT to default CSV reading <-- # MODIFIED: Explicitly handle index
                    df = pd.read_csv(
                        io.StringIO(decoded.decode("utf-8")),
                        index_col=False,  # ADDED: Prevent first column from being treated as index
                        # header=0, # REMOVE explicit header
                        # index_col=0 # REMOVE explicit index -> Replaced above
                    )
                    # ---> REMOVE erroneous column conversion attempt <---
                    # try:
                    #     numeric_columns = [pd.to_numeric(col) for col in df.columns]
                    #     df.columns = numeric_columns  # Assign list of numeric columns
                    #     logger.info(
                    #         "Successfully converted CSV column headers to numeric."
                    #     )
                    # except (ValueError, TypeError) as e_conv:
                    #     logger.warning(
                    #         f"CSV column headers could not be fully converted to numeric: {e_conv}. "
                    #         "Proceeding, but this might indicate an issue."
                    #     )
                    #     # Continue, validation function might handle it or raise specific error
                    #     pass

                    run_validation = True
                    preview_content = html.Em("Validating CSV...")
                except Exception as e_csv:
                    logger.error(f"Error reading CSV: {e_csv}", exc_info=True)
                    validation_message = dbc.Alert(
                        f"Error reading CSV: {e_csv}", color="danger"
                    )
                    df = None

            # Excel Upload - Populate Dropdown
            elif filename.lower().endswith((".xls", ".xlsx")):
                logger.info("Processing Excel file - populating sheets.")
                sheet_group_style = {"display": "block"}
                try:
                    excel_file = pd.ExcelFile(io.BytesIO(decoded))
                    sheet_names = excel_file.sheet_names
                    dropdown_options = [
                        {"label": name, "value": name} for name in sheet_names
                    ]
                    dropdown_disabled = False
                    preview_content = html.Em("Select a sheet from the dropdown above.")
                    if not sheet_names:
                        validation_message = dbc.Alert(
                            "Excel file contains no sheets.", color="warning"
                        )
                        dropdown_disabled = True
                    else:
                        validation_message = dbc.Alert(
                            "Select the sheet containing the processed data.",
                            color="info",
                        )
                except Exception as e_excel_sheets:
                    logger.error(
                        f"Error reading Excel sheet names: {e_excel_sheets}",
                        exc_info=True,
                    )
                    validation_message = dbc.Alert(
                        f"Error reading Excel file structure: {e_excel_sheets}",
                        color="danger",
                    )
                    sheet_group_style = {"display": "none"}
                df = None  # Don't read data yet
                run_validation = False

            # Unsupported Type
            else:
                logger.warning(f"Unsupported file type: {filename}")
                validation_message = dbc.Alert(
                    "Unsupported file type.", color="warning"
                )
                sheet_group_style = {"display": "none"}

        except Exception as e_parse:
            logger.error(f"Error parsing file content: {e_parse}", exc_info=True)
            validation_message = dbc.Alert(
                f"Error processing file content: {e_parse}", color="danger"
            )
            # Reset everything
            return (
                html.Em("Error processing file."),
                validation_message,
                True,
                {"display": "none"},
                None,
                [],
                None,
                True,
            )

    # Scenario 2: Sheet Selected from Dropdown
    elif triggered_id == "upload-processed-sheet-dropdown":
        logger.info(f"Sheet selected: {selected_sheet_value}")
        dropdown_options = current_dropdown_options  # Preserve options
        dropdown_value = selected_sheet_value  # Keep selected value
        dropdown_disabled = False  # Keep dropdown enabled
        sheet_group_style = {"display": "block"}  # Keep dropdown visible

        # Need file content to read the selected sheet
        if not contents or not filename or not selected_sheet_value:
            logger.debug("Sheet selected but no file content or sheet value. Ignoring.")
            # This might happen if selection changes while file is cleared, prevent update
            raise PreventUpdate

        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            logger.info(
                f"Reading selected sheet '{selected_sheet_value}' from {filename}"
            )
            df = pd.read_excel(io.BytesIO(decoded), sheet_name=selected_sheet_value)
            run_validation = True  # Validate this specific sheet
            preview_content = html.Em(f"Validating sheet '{selected_sheet_value}'...")
        except Exception as e_read_sheet:
            logger.error(
                f"Error reading selected sheet '{selected_sheet_value}': {e_read_sheet}",
                exc_info=True,
            )
            validation_message = dbc.Alert(
                f"Error reading sheet '{selected_sheet_value}': {e_read_sheet}",
                color="danger",
            )
            df = None
            run_validation = False
            # Reset temp store if sheet read fails
            temp_store_data = None

    else:
        # Unexpected trigger
        logger.warning(f"Callback triggered by unexpected ID: {triggered_id}")
        raise PreventUpdate

    # --- Determine Workflow based on Trigger --- End

    # --- Run Validation (if applicable) --- Start
    if run_validation and df is not None:
        logger.info(f"Running format detection and validation for {filename}")
        validated_df, detected_format, error_msg = detect_and_validate_format(df)

        if error_msg:
            validation_message = dbc.Alert(
                f"Validation Error: {error_msg}", color="danger"
            )
            logger.warning(f"Validation failed for {filename}: {error_msg}")
            preview_content = html.Em("Validation failed.")  # Clearer message
            submit_disabled = True
            temp_store_data = None
        elif validated_df is not None and detected_format:
            logger.info(
                f"Validation successful for {filename}. Format: {detected_format}"
            )
            validation_message = dbc.Alert(
                f"Sheet '{selected_sheet_value or 'CSV'}' validated as {detected_format} format. Ready to submit.",
                color="success",
            )
            submit_disabled = False
            # Store validated data and format type
            df_json = validated_df.to_json(orient="split", date_format="iso")
            temp_store_data = {
                "filename": filename,
                "sheet_name": selected_sheet_value,
                "data_json": df_json,
                "format": detected_format,
            }
            preview_df = (
                validated_df.reset_index()
                if detected_format == "wide"
                else validated_df
            )
            preview_table = dash_table.DataTable(
                columns=[{"name": str(i), "id": str(i)} for i in preview_df.columns],
                data=preview_df.head(5).to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "5px",
                    "fontSize": "0.8rem",
                },
                style_header={"fontWeight": "bold"},
            )
            preview_content = preview_table
        else:
            validation_message = dbc.Alert("Internal validation error.", color="danger")
            preview_content = html.Em("Internal validation error.")
            submit_disabled = True
            temp_store_data = None

    elif df is None and validation_message is None:
        # This state now primarily occurs on initial Excel upload before sheet selection
        logger.debug("df is None, likely waiting for Excel sheet selection.")
        # preview_content/validation_message are already set in the Excel workflow block
        submit_disabled = True
        temp_store_data = None

    # --- Run Validation (if applicable) --- End

    # Final adjustments if needed (e.g., ensuring dropdown state is correct on error)
    if validation_message and "Error" in validation_message.children:
        submit_disabled = True  # Ensure submit is disabled on any error
        temp_store_data = None  # Clear store on any error

    logger.debug(
        f"Returning - Submit Disabled: {submit_disabled}, Sheet Style: {sheet_group_style}, "
        f"Dropdown Disabled: {dropdown_disabled}, Dropdown Value: {dropdown_value}"
    )
    return (
        preview_content,
        validation_message,
        submit_disabled,
        sheet_group_style,
        temp_store_data,
        dropdown_options,
        dropdown_value,
        dropdown_disabled,
    )


# Callback to handle final submission
@callback(
    Output("processed-datasets-store", "data", allow_duplicate=True),
    Output("upload-processed-validation-message", "children", allow_duplicate=True),
    Output(
        "upload-processed-temp-store", "data", allow_duplicate=True
    ),  # Clear temp store on submit
    Input("upload-processed-submit-button", "n_clicks"),
    State("upload-processed-temp-store", "data"),
    State("processed-datasets-store", "data"),
    prevent_initial_call=True,
)
def submit_processed_data(
    n_clicks,
    temp_data: Optional[Dict[str, Any]],
    existing_processed_data: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Any, None]:
    """Adds the validated processed data to the main store."""
    if not n_clicks or not temp_data:
        logger.debug("Submit callback triggered without click or temp data.")
        return no_update, no_update, no_update

    updated_processed_data = existing_processed_data or {}
    final_validation_message: Any = None

    try:
        filename = temp_data.get("filename")
        df_json = temp_data.get("data_json")
        detected_format = temp_data.get("format")  # Get the detected format

        if not filename or not df_json or not detected_format:
            raise ValueError(
                "Temporary data is missing filename, dataframe JSON, or format."
            )

        # Determine data_type based on format
        data_type = (
            f"uploaded_{detected_format}"  # e.g., 'uploaded_wide' or 'uploaded_long'
        )

        # Generate a unique ID (e.g., filename + timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"UPLOADED_{filename}_{timestamp}"

        # Check for ID collision
        while dataset_id in updated_processed_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dataset_id = f"UPLOADED_{filename}_{timestamp}"

        logger.info(f"Submitting uploaded {detected_format} data with ID: {dataset_id}")

        # Store the data, marking its type based on detected format
        updated_processed_data[dataset_id] = {
            "data_type": data_type,  # Use the dynamic type
            "original_filename": filename,
            "data_json": df_json,
            "upload_time": datetime.now().isoformat(),
        }

        logger.info(
            f"Successfully added uploaded dataset '{dataset_id}' to processed-datasets-store."
        )
        final_validation_message = dbc.Alert(
            f"Successfully uploaded and added dataset '{filename}' as '{dataset_id}'.",
            color="success",
            duration=4000,
        )

    except Exception as e:
        logger.error(f"Error submitting processed data: {e}", exc_info=True)
        final_validation_message = dbc.Alert(
            f"Failed to submit data. An error occurred: {e}", color="danger"
        )
        return no_update, final_validation_message, None

    return (
        updated_processed_data,
        final_validation_message,
        None,
    )
