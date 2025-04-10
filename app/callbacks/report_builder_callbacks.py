"""Callbacks for the Report Builder tab."""

import io
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    callback_context,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

# Import core processing functions
from app.utils.data_processing import interpolate_thermogram
from core.baseline.advanced import subtract_spline_baseline
from core.metrics.metric_calculation import calculate_thermogram_metrics
from core.peaks.peak_detection import detect_thermogram_peaks

# Assuming core functions are importable

logger = logging.getLogger(__name__)


# --- Callback to Switch to Report Tab & Select Dataset --- Start
@callback(
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Output("report-dataset-selector", "value"),
    Input({"type": "go-to-report-builder-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def switch_to_report_builder_tab(n_clicks: List[Optional[int]]) -> Tuple[str, str]:
    """Switches to the Report Builder tab and selects the dataset when
    the 'Generate Report' button is clicked on the Data Overview page."""
    ctx = callback_context
    if not ctx.triggered_id or not any(n for n in n_clicks if n):
        raise PreventUpdate

    dataset_id = ctx.triggered_id.get("index")
    if not dataset_id:
        logger.warning("Could not get dataset_id from report builder trigger button.")
        raise PreventUpdate

    logger.info(f"Switching to Report Builder tab for dataset: {dataset_id}")
    return "tab-report-builder", dataset_id


# --- Callback to Switch to Report Tab & Select Dataset --- End


# --- Callback to Populate Dataset Selector --- Start
@callback(
    Output("report-dataset-selector", "options"),
    Input("processed-datasets-store", "data"),
    prevent_initial_call=True,
)
def populate_report_dataset_selector(
    processed_datasets: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Populates the dataset selector dropdown for the Report Builder."""
    if not processed_datasets:
        return []

    options = [
        {"label": dataset_id, "value": dataset_id}
        for dataset_id in sorted(processed_datasets.keys())
    ]
    logger.debug(f"Populating report dataset selector with {len(options)} options.")
    return options


# --- Callback to Populate Dataset Selector --- End


# --- Callback to Show/Hide Content, Populate Metrics, Trigger Calculation --- Start
@callback(
    Output("report-builder-content", "style"),
    Output("report-selector-message", "is_open"),
    Output("report-selector-message", "children"),
    Output("report-metric-selector", "options"),
    Output("report-metric-selector", "value"),
    Output("report-metric-tooltips-div", "children"),
    Output("report-calculated-metrics-temp", "data", allow_duplicate=True),
    Output("report-preview-table-div", "children", allow_duplicate=True),
    Output("generate-report-button", "disabled", allow_duplicate=True),
    Output("report-name-input", "value"),
    Input("report-dataset-selector", "value"),
    State("generated-reports-store", "data"),
    prevent_initial_call=True,
)
def handle_report_dataset_selection(
    selected_dataset_id: Optional[str],
    existing_reports: Optional[Dict[str, Any]],
) -> Tuple[
    Dict,
    bool,
    str,
    List[Dict[str, Any]],
    List[str],
    List[dbc.Tooltip],
    Dict,
    Any,
    bool,
    str,
]:
    """Shows the report builder content when a dataset is selected,
    populates available metrics with tooltips, triggers calculation,
    and sets a unique default report name."""
    existing_reports = existing_reports or {}

    # --- Generate Set of Existing Report Basenames --- Start
    existing_report_basenames = set()
    for report_meta in existing_reports.values():
        if isinstance(report_meta, dict) and "report_filename" in report_meta:
            basename, _ = os.path.splitext(report_meta["report_filename"])
            existing_report_basenames.add(basename)
    # --- Generate Set of Existing Report Basenames --- End

    if not selected_dataset_id:
        logger.debug("No dataset selected for report. Hiding content.")
        # Hide content, clear message, reset metrics/preview/button/temp_store/tooltips/name
        return (
            {"display": "none"},
            False,
            "",  # Message closed and cleared
            [],
            [],  # Metrics checklist empty
            [],  # Clear tooltips
            {"clear": True},  # Clear the temp store
            None,  # Clear preview
            True,  # Disable generate button
            "",  # Clear report name input
        )

    logger.info(f"Dataset '{selected_dataset_id}' selected for report.")

    # --- Generate Unique Default Report Name --- Start
    base_name = "New_Report"
    unique_report_name = base_name
    counter = 1
    while unique_report_name in existing_report_basenames:
        unique_report_name = f"{base_name}_{counter}"
        counter += 1
    logger.debug(f"Setting default report name to: {unique_report_name}")
    # --- Generate Unique Default Report Name --- End

    # Define available metrics (Keys should match output of calculate_thermogram_metrics)
    # Moved definition here to be reused for tooltips
    available_metrics = {
        "Peak_F": "Peak F Height",
        "TPeak_F": "Peak F Temp (°C)",
        "Peak_1": "Peak 1 Height",
        "TPeak_1": "Peak 1 Temp (°C)",
        "Peak_2": "Peak 2 Height",
        "TPeak_2": "Peak 2 Temp (°C)",
        "Peak_3": "Peak 3 Height",
        "TPeak_3": "Peak 3 Temp (°C)",
        "Peak1_Peak2_Ratio": "Peak 1 / Peak 2 Ratio",
        "Peak1_Peak3_Ratio": "Peak 1 / Peak 3 Ratio",
        "Peak2_Peak3_Ratio": "Peak 2 / Peak 3 Ratio",
        "V1.2": "Valley 1-2 Height",
        "TV1.2": "Valley 1-2 Temp (°C)",
        "V1.2_Peak1_Ratio": "V1.2 / Peak 1 Ratio",
        "V1.2_Peak2_Ratio": "V1.2 / Peak 2 Ratio",
        "V1.2_Peak3_Ratio": "V1.2 / Peak 3 Ratio",
        "Max": "Max Height",
        "TMax": "Temp at Max Height (°C)",
        "Min": "Min Height",
        "TMin": "Temp at Min Height (°C)",
        "Median": "Median Height",
        "Area": "Area Under Curve",
        "TFM": "First Moment Temp (TFM) (°C)",
        "FWHM": "Full Width at Half Max (FWHM) (°C)",
    }

    # Define tooltips for each metric
    metric_tooltips_text = {
        "Peak_F": "Height of peak corresponding to Fibrinogen temperature region (47 - 60 C).",
        "TPeak_F": "Temperature of peak corresponding to Fibrinogen temperature region (47 - 60 C).",
        "Peak_1": "Height of peak corresponding to Peak 1 temperature region (60 - 66 C).",
        "TPeak_1": "Temperature of peak corresponding to Peak 1.",
        "Peak_2": "Height of peak corresponding to Peak 2 temperature region (67 - 73 C).",
        "TPeak_2": "Temperature of peak corresponding to Peak 2.",
        "Peak_3": "Height of peak corresponding to Peak 3 temperature region (73 - 81 C).",
        "TPeak_3": "Temperature of peak corresponding to Peak 3.",
        "Peak1_Peak2_Ratio": "Peak 1 to Peak 2 ratio.",
        "Peak1_Peak3_Ratio": "Peak 1 to Peak 3 ratio.",
        "Peak2_Peak3_Ratio": "Peak 2 to Peak 3 ratio.",
        "V1.2": "Valley (minimum amplitude) between Peak 1 and Peak 2 (inclusive).",
        "TV1.2": "Temperature of Valley between Peak 1 and Peak 2 (inclusive).",
        "V1.2_Peak1_Ratio": "Ratio of Valley between Peak 1 and Peak 2 to the amplitude of Peak 1.",
        "V1.2_Peak2_Ratio": "Ratio of Valley between Peak 1 and Peak 2 to the amplitude of Peak 2.",
        "V1.2_Peak3_Ratio": "Ratio of Valley between Peak 1 and Peak 2 to the amplitude of Peak 3.",
        "Max": "Maximum observed excess heat capacity.",
        "TMax": "Temperature corresponding to maximum height.",
        "Min": "Minimum observed excess heat capacity.",
        "TMin": "Temperature corresponding to minimum observed excess heat capacity.",
        "Median": "Median observed excess heat capacity.",
        "Area": "Total area under the thermogram signature.",
        "TFM": "Temperature corresponding to the first moment.",
        "FWHM": "Full width at half max.",
    }

    metric_options = []
    metric_tooltips = []
    for key, label in available_metrics.items():
        # label_id = f"metric-label-{key}"  # No longer target the main label
        icon_id = f"metric-tooltip-icon-{key}"  # ID for the icon

        # Create the label with text and an icon
        option_label = html.Span(
            [
                label,  # The metric name
                html.Sup(  # Superscript for small text
                    html.Abbr(
                        " (?) ",  # Icon text
                        title="",  # Tooltip will provide info
                        id=icon_id,
                        style={
                            "cursor": "pointer",
                            "textDecoration": "none",
                        },  # Style icon
                    ),
                    style={"marginLeft": "5px"},  # Space between label and icon
                ),
            ]
        )

        metric_options.append({"label": option_label, "value": key})

        tooltip_text = metric_tooltips_text.get(key, "No description available.")
        # Format tooltip content (already done)
        tooltip_content = html.Div(
            [
                html.Strong(label),  # Use the friendly label as title
                html.Br(),
                tooltip_text,
            ]
        )
        metric_tooltips.append(
            dbc.Tooltip(
                tooltip_content,  # Use formatted content
                target=icon_id,  # Target the ICON ID
                placement="right",
                delay={"show": 500, "hide": 50},  # Add delay
                style={"maxWidth": "300px"},  # Optional: constrain width
            )
        )

    # Sensible default selections
    default_selected_metrics = ["TPeak_1", "TPeak_2", "TPeak_3", "Area", "FWHM", "TFM"]
    # Filter defaults to ensure they exist in options (in case list changes)
    default_selected_metrics = [
        m for m in default_selected_metrics if m in available_metrics
    ]

    # Trigger data calculation
    trigger_data = {"trigger": time.time(), "dataset_id": selected_dataset_id}

    return (
        {"display": "block"},  # Show content
        True,
        f"Calculating metrics for: {selected_dataset_id}",  # Show message
        metric_options,
        default_selected_metrics,
        metric_tooltips,  # <-- ADD tooltips to return tuple
        trigger_data,  # Trigger calculation
        html.Em("Calculating metrics..."),  # Placeholder for preview
        True,  # Disable generate button until preview is ready
        unique_report_name,  # <-- ADDED default report name
    )


# --- Callback to Show/Hide Content, Populate Metrics, Trigger Calculation --- End


# --- Helper to Generate Processed Data for Saved Review Datasets --- Start
def _get_processed_data_for_saved_review(
    dataset_id: str,
    dataset_info: Dict[str, Any],
    all_samples_data: Optional[Dict[str, Any]],
) -> Optional[pd.DataFrame]:
    """Generates a processed (baseline-subtracted, interpolated) DataFrame
    for datasets saved via the 'Save Processed Data' button.

    Handles interpolation to a common grid like the download function.

    Args:
        dataset_id: The ID of the processed dataset.
        dataset_info: The metadata for this dataset from processed-datasets-store.
        all_samples_data: The main store containing raw data.

    Returns:
        A DataFrame with columns ['SampleID', 'Temperature', 'dCp_subtracted'],
        or None if data generation fails.
    """
    logger.info(f"Generating processed data for saved dataset: {dataset_id}")
    baseline_parameters = dataset_info.get("baseline_parameters")
    source_raw_file = dataset_info.get("source_raw_file")

    if not baseline_parameters or not source_raw_file:
        logger.error(f"Missing baseline params or source file for '{dataset_id}'.")
        return None
    if not all_samples_data or source_raw_file not in all_samples_data:
        logger.error(f"Source raw data '{source_raw_file}' not found.")
        return None

    raw_file_data = all_samples_data[source_raw_file]
    raw_samples_dict = raw_file_data.get("samples")
    if not raw_samples_dict:
        logger.error(f"'samples' key missing in raw data for '{source_raw_file}'.")
        return None

    # --- Determine Common Temp Grid (like download) --- Start
    min_temp_overall = np.inf
    max_temp_overall = -np.inf
    valid_sample_count = 0
    for sample_id, params in baseline_parameters.items():
        raw_data_list = raw_samples_dict.get(sample_id)
        if not raw_data_list:
            continue
        try:
            df_raw = pd.DataFrame(raw_data_list)
            if df_raw.empty or "Temperature" not in df_raw:
                continue
            current_min = df_raw["Temperature"].min()
            current_max = df_raw["Temperature"].max()
            if not pd.isna(current_min):
                min_temp_overall = min(min_temp_overall, current_min)
            if not pd.isna(current_max):
                max_temp_overall = max(max_temp_overall, current_max)
            valid_sample_count += 1
        except Exception:
            pass  # Ignore errors in range finding

    if (
        not np.isfinite(min_temp_overall)
        or not np.isfinite(max_temp_overall)
        or valid_sample_count == 0
    ):
        logger.error("Cannot determine temperature range for interpolation.")
        return None
    step = 0.1
    interpolated_temps = np.round(
        np.arange(min_temp_overall, max_temp_overall + step / 2, step), 1
    )
    # --- Determine Common Temp Grid --- End

    # --- Process Each Sample --- Start
    processed_data_list = []
    for sample_id, params in baseline_parameters.items():
        raw_data_list = raw_samples_dict.get(sample_id)
        if not raw_data_list:
            continue
        try:
            df_raw = pd.DataFrame(raw_data_list)
            if df_raw.empty or "Temperature" not in df_raw or "dCp" not in df_raw:
                continue

            df_subtracted = subtract_spline_baseline(
                df_raw, params.get("lower"), params.get("upper")
            )
            if df_subtracted is None or df_subtracted.empty:
                continue

            # Prepare for interpolation
            df_for_interp = df_subtracted[["Temperature", "dCp_subtracted"]].rename(
                columns={"dCp_subtracted": "dCp"}
            )
            df_interpolated = interpolate_thermogram(
                df_for_interp, temp_grid=interpolated_temps
            )

            if df_interpolated is not None and not df_interpolated.empty:
                df_interpolated["SampleID"] = sample_id
                # Select and rename for final long format
                final_sample_df = df_interpolated[
                    ["SampleID", "Temperature", "dCp"]
                ].rename(columns={"dCp": "dCp_subtracted"})
                processed_data_list.append(final_sample_df)

        except Exception as e:
            logger.error(
                f"Error processing sample '{sample_id}' for report generation: {e}",
                exc_info=True,
            )
    # --- Process Each Sample --- End

    if not processed_data_list:
        logger.error(f"No samples could be processed for dataset '{dataset_id}'.")
        return None

    # Combine all samples into one long dataframe
    final_df = pd.concat(processed_data_list, ignore_index=True)
    logger.info(
        f"Successfully generated processed data (long format) for '{dataset_id}'. Shape: {final_df.shape}"
    )
    return final_df


# --- Helper to Generate Processed Data --- End


# --- Callback to Calculate Metrics --- Start
@callback(
    Output(
        "report-calculated-metrics-temp", "data"
    ),  # Output result back to same store
    Input("report-calculated-metrics-temp", "data"),  # Triggered by previous callback
    State("processed-datasets-store", "data"),
    State("all-samples-data", "data"),
    prevent_initial_call=True,
)
def calculate_metrics_for_report(
    trigger_data: Optional[Dict[str, Any]],
    processed_datasets: Optional[Dict[str, Any]],
    all_samples_data: Optional[Dict[str, Any]],
) -> Dict:
    """Calculates metrics for the selected dataset when triggered."""
    # Check if triggered correctly
    if not trigger_data or "trigger" not in trigger_data:
        logger.debug("calculate_metrics callback triggered without valid trigger data.")
        # Return no_update or empty dict if it shouldn't clear?
        # Let's clear if input is invalid
        return {"error": "Invalid trigger data"}

    dataset_id = trigger_data.get("dataset_id")
    if not dataset_id:
        logger.error("Trigger data missing dataset_id.")
        return {"error": "Missing dataset_id in trigger"}

    logger.info(f"Calculating metrics for report, triggered for dataset: {dataset_id}")

    # --- Get Dataset Info --- Start
    if not processed_datasets or dataset_id not in processed_datasets:
        logger.error(f"Dataset '{dataset_id}' not found in processed-datasets-store.")
        return {"error": f"Dataset {dataset_id} not found"}
    dataset_info = processed_datasets[dataset_id]
    data_type = dataset_info.get("data_type", "saved_from_review")
    logger.info(f"Dataset type: {data_type}")
    # --- Get Dataset Info --- End

    # --- Get Processed Data --- Start
    processed_df: Optional[pd.DataFrame] = None
    if data_type == "uploaded_wide":
        df_json = dataset_info.get("data_json")
        if df_json:
            try:
                # ---> Manual JSON Parsing and DataFrame Creation <---
                # Load JSON manually
                json_data = json.loads(df_json)
                str_columns = json_data["columns"]
                sample_ids_index = json_data["index"]
                data_values = json_data["data"]

                # Convert columns (temperatures) explicitly BEFORE creating DataFrame
                try:
                    # Use errors='raise' during conversion to catch issues early
                    numeric_columns = pd.Index(
                        pd.to_numeric(str_columns, errors="raise"), dtype=np.float64
                    )
                    logger.info(
                        "Successfully converted JSON columns list to numeric float64 index."
                    )
                except Exception as e_col_conv:
                    logger.error(
                        f"Failed to convert string columns from JSON to numeric: {e_col_conv}"
                    )
                    return {"error": "Invalid numeric columns in JSON data"}

                # Create DataFrame
                processed_df = pd.DataFrame(
                    data_values, index=sample_ids_index, columns=numeric_columns
                )
                logger.info(
                    f"Loaded wide uploaded data MANUALLY. Shape: {processed_df.shape}"
                )

                # ---> Verify columns immediately after manual creation <---
                logger.debug(
                    f"After MANUAL load: processed_df.columns.dtype = {processed_df.columns.dtype}"
                )
                logger.debug(
                    f"After MANUAL load: processed_df.columns[:5] = {processed_df.columns[:5].tolist()}"
                )
                try:
                    test_cols_range = processed_df.columns.to_series().between(
                        50, 54, inclusive="both"
                    )
                    logger.info(
                        f"After MANUAL load: Number of columns between 50-54 = {test_cols_range.sum()}"
                    )
                except Exception as e_col_test:
                    logger.error(
                        f"Error testing columns range after manual load: {e_col_test}"
                    )
                # ---> End column verification <---
                # ---> END Manual JSON Parsing <---

            except json.JSONDecodeError as e_json:
                logger.error(
                    f"Failed to decode wide data JSON string: {e_json}", exc_info=True
                )
                return {"error": f"Corrupted JSON data for {dataset_id}"}
            except KeyError as e_key:
                logger.error(
                    f"Missing expected key ('columns', 'index', or 'data') in JSON: {e_key}",
                    exc_info=True,
                )
                return {"error": f"Incomplete JSON data structure for {dataset_id}"}
            except Exception as e_manual_load:
                logger.error(
                    f"Failed to manually load/process wide data JSON: {e_manual_load}",
                    exc_info=True,
                )
                return {"error": f"Failed to process wide data for {dataset_id}"}
        else:
            logger.error("Missing data_json for uploaded_wide dataset.")
            return {"error": "Missing data JSON"}

    elif data_type == "uploaded_long":
        df_json = dataset_info.get("data_json")
        if df_json:
            try:
                # Data is stored long, columns: SampleID, Temperature, dCp_subtracted
                # Use orient='table' for consistency
                processed_df = pd.read_json(df_json, orient="table")
                logger.info(f"Loaded long uploaded data. Shape: {processed_df.shape}")
            except Exception as e:
                logger.error(f"Failed to load long data JSON: {e}", exc_info=True)
                return {"error": f"Failed to load data for {dataset_id}"}
        else:
            logger.error("Missing data_json for uploaded_long dataset.")
            return {"error": "Missing data JSON"}

    elif data_type == "saved_from_review":
        processed_df = _get_processed_data_for_saved_review(
            dataset_id, dataset_info, all_samples_data
        )
        if processed_df is None:
            logger.error(f"Failed to generate processed data for '{dataset_id}'.")
            return {"error": f"Failed to generate data for {dataset_id}"}
        # Result is already in long format

    else:
        logger.error(f"Unknown data_type '{data_type}' for dataset '{dataset_id}'.")
        return {"error": f"Unknown data type {data_type}"}

    if processed_df is None or processed_df.empty:
        logger.error("Processed DataFrame is empty or None after loading/generation.")
        return {"error": "Processed data is empty"}
    # --- Get Processed Data --- End

    # --- Calculate Metrics Per Sample --- Start
    all_metrics = []
    sample_ids = []

    # Handle Wide vs Long format for iteration
    if data_type == "uploaded_wide":
        sample_ids = processed_df.index.unique().tolist()
        logger.info(
            f"Calculating metrics for {len(sample_ids)} samples from WIDE data."
        )
        for sample_id in sample_ids:
            try:
                # Extract sample data (Series), index=float Temperature, values=dCp
                sample_series = processed_df.loc[sample_id]

                # Convert Series to DataFrame (intermediate)
                df_temp = sample_series.reset_index()
                # Now columns are 'index' (Temperatures) and sample_id (dCp values)

                # ---> Create final sample_df with correct types directly <---
                try:
                    sample_df = pd.DataFrame(
                        {
                            "Temperature": pd.to_numeric(
                                df_temp["index"], errors="coerce"
                            ),
                            "dCp_subtracted": pd.to_numeric(
                                df_temp[sample_id], errors="coerce"
                            ),
                        }
                    ).astype({"Temperature": np.float64, "dCp_subtracted": np.float64})
                    logger.debug(
                        f"Created sample_df directly for {sample_id}, dtypes:\\n{sample_df.dtypes}"
                    )
                except Exception as e_create_df:
                    logger.error(
                        f"Error creating final sample_df for {sample_id}: {e_create_df}",
                        exc_info=True,
                    )
                    all_metrics.append(
                        {"SampleID": sample_id, "Error": "Dataframe creation error"}
                    )
                    continue  # Skip sample
                # ---> End direct creation <---

                # ---> Add logging BEFORE cleaning <---
                logger.debug(
                    f"BEFORE cleaning for {sample_id}, dtypes:\\n{sample_df.dtypes}"
                )
                logger.debug(
                    f"BEFORE cleaning for {sample_id}, head():\\n{sample_df.head().to_string()}"
                )
                logger.debug(
                    f"BEFORE cleaning for {sample_id}, isna().sum():\\n{sample_df.isna().sum()}"
                )
                # ---> END logging BEFORE cleaning <---

                # ---> Add detailed logging before peak detection <---
                logger.debug(
                    f"--- Preparing for Peak Detection (Sample: {sample_id}) ---"
                )
                logger.debug(f"sample_df dtypes:\\n{sample_df.dtypes}")
                logger.debug(f"sample_df head():\\n{sample_df.head().to_string()}")
                logger.debug(
                    f"sample_df describe():\\n{sample_df.describe().to_string()}"
                )
                logger.debug("--- End Prepare Log ---")
                # ---> End detailed logging <---

                if sample_df.empty:  # Check again after logging
                    logger.warning(
                        f"Skipping sample '{sample_id}' due to empty data before peak detection."
                    )
                    continue
                # ---> Baseline subtraction REMOVED (already subtracted in processed data) <---

                # ---> Use sample_df directly for Peaks/Metrics <---
                logger.debug(f"Detecting peaks for sample: {sample_id}")
                peaks_data = detect_thermogram_peaks(sample_df)  # Use sample_df
                logger.debug(f"Calculating final metrics for sample: {sample_id}")
                metrics = calculate_thermogram_metrics(
                    sample_df, peaks_data
                )  # Use sample_df
                metrics["SampleID"] = sample_id  # Add sample ID to results
                all_metrics.append(metrics)

            except Exception as e_sample:
                logger.error(
                    f"Error processing sample '{sample_id}' from uploaded wide data: {e_sample}",
                    exc_info=True,
                )
                placeholder_metrics = {
                    "SampleID": sample_id,
                    "Error": f"Processing error: {e_sample}",
                }
                all_metrics.append(placeholder_metrics)
                continue  # Skip to next sample

    else:  # Long format (uploaded_long or saved_from_review)
        # This part assumes long format also contains dCp_subtracted correctly
        sample_ids = processed_df["SampleID"].unique().tolist()
        logger.info(
            f"Calculating metrics for {len(sample_ids)} samples from LONG data."
        )
        for sample_id in sample_ids:
            sample_df = processed_df[processed_df["SampleID"] == sample_id]
            if sample_df.empty:
                logger.warning(f"Skipping sample '{sample_id}' due to empty data.")
                continue

            # Ensure required columns are present and sorted
            sample_df = sample_df[["Temperature", "dCp_subtracted"]].sort_values(
                "Temperature"
            )

            peaks_data = detect_thermogram_peaks(sample_df)
            metrics = calculate_thermogram_metrics(sample_df, peaks_data)
            metrics["SampleID"] = sample_id  # Add sample ID to results
            all_metrics.append(metrics)
    # --- Calculate Metrics Per Sample --- End

    if not all_metrics:
        logger.error("No metrics could be calculated for any sample.")
        return {"error": "Metric calculation failed for all samples"}

    logger.info(f"Successfully calculated metrics for {len(all_metrics)} samples.")
    # Store the results list back into the temp store
    return_data = {"results": all_metrics, "dataset_id": dataset_id}
    logger.debug(f"Returning data from calculate_metrics_for_report: {return_data}")
    return return_data


# --- Callback to Calculate Metrics --- End


# Re-define available metrics here for easy access in the preview callback
# TODO: Move this to a shared config/utils location later?
available_metrics_dict = {
    "Peak_F": "Peak F Height",
    "TPeak_F": "Peak F Temp (°C)",
    "Peak_1": "Peak 1 Height",
    "TPeak_1": "Peak 1 Temp (°C)",
    "Peak_2": "Peak 2 Height",
    "TPeak_2": "Peak 2 Temp (°C)",
    "Peak_3": "Peak 3 Height",
    "TPeak_3": "Peak 3 Temp (°C)",
    "Peak1_Peak2_Ratio": "Peak 1 / Peak 2 Ratio",
    "Peak1_Peak3_Ratio": "Peak 1 / Peak 3 Ratio",
    "Peak2_Peak3_Ratio": "Peak 2 / Peak 3 Ratio",
    "V1.2": "Valley 1-2 Height",
    "TV1.2": "Valley 1-2 Temp (°C)",
    "V1.2_Peak1_Ratio": "V1.2 / Peak 1 Ratio",
    "V1.2_Peak2_Ratio": "V1.2 / Peak 2 Ratio",
    "V1.2_Peak3_Ratio": "V1.2 / Peak 3 Ratio",
    "Max": "Max Height",
    "TMax": "Temp at Max Height (°C)",
    "Min": "Min Height",
    "TMin": "Temp at Min Height (°C)",
    "Median": "Median Height",
    "Area": "Area Under Curve",
    "TFM": "First Moment Temp (TFM) (°C)",
    "FWHM": "Full Width at Half Max (FWHM) (°C)",
}


# --- Callback to Update Preview Table --- Start
@callback(
    Output("report-preview-table-div", "children"),
    Output("generate-report-button", "disabled"),
    Input("report-calculated-metrics-temp", "data"),  # Calculated metrics (or error)
    Input("report-metric-selector", "value"),  # Selected metrics
    prevent_initial_call=True,
)
def update_report_preview(
    calculated_data: Optional[Dict[str, Any]],
    selected_metrics: List[str],
) -> Tuple[Any, bool]:
    """Updates the report preview table based on calculated metrics and selections."""
    logger.debug(f"update_report_preview received calculated_data: {calculated_data}")
    if not calculated_data:
        logger.debug("Report preview update: No calculated data.")
        return html.Em("Select a dataset to calculate metrics."), True

    # Check if calculation resulted in an error
    if "error" in calculated_data:
        error_msg = calculated_data["error"]
        logger.warning(f"Report preview update: Calculation error - {error_msg}")
        return dbc.Alert(
            f"Error calculating metrics: {error_msg}", color="danger"
        ), True

    # Check if results are present
    all_metrics_results = calculated_data.get("results")
    if not all_metrics_results or not isinstance(all_metrics_results, list):
        logger.warning(
            "Report preview update: No valid 'results' found in calculated data."
        )
        return html.Em("No metrics calculated or results are invalid."), True

    logger.info(
        f"Updating report preview with {len(all_metrics_results)} samples and selected metrics: {selected_metrics}"
    )

    # Prepare data for the table, filtering by selected metrics
    preview_data = []
    for sample_metrics in all_metrics_results:
        row = {"SampleID": sample_metrics.get("SampleID", "N/A")}
        for metric_key in selected_metrics:
            value = sample_metrics.get(metric_key)
            # Basic rounding for floats
            if isinstance(value, (float, np.floating)):
                # Apply more specific rounding based on metric type later if needed
                row[metric_key] = round(value, 3) if not pd.isna(value) else None
            else:
                row[metric_key] = value
        preview_data.append(row)

    # Define columns for the table based on selection
    preview_columns = [{"name": "SampleID", "id": "SampleID"}]
    for metric_key in selected_metrics:
        # Use the friendly name from our dict, default to key if not found
        friendly_name = available_metrics_dict.get(metric_key, metric_key)
        preview_columns.append({"name": str(friendly_name), "id": str(metric_key)})

    # Create the DataTable
    preview_table = dash_table.DataTable(
        id="report-preview-table",
        columns=preview_columns,
        data=preview_data,
        page_size=10,  # Add pagination for potentially long reports
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px", "fontSize": "0.8rem"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
    )

    return preview_table, False  # Return table, enable Generate button


# --- Callback to Update Preview Table --- End


# --- Callback to Generate Report File --- Start
@callback(
    Output("generated-reports-store", "data"),
    Output("report-selector-message", "children", allow_duplicate=True),
    Output("report-selector-message", "is_open", allow_duplicate=True),
    Output("report-selector-message", "color", allow_duplicate=True),
    Input("generate-report-button", "n_clicks"),
    State("report-calculated-metrics-temp", "data"),
    State("report-metric-selector", "value"),
    State("report-name-input", "value"),
    State("report-format-dropdown", "value"),
    State("generated-reports-store", "data"),
    prevent_initial_call=True,
)
def generate_report(
    n_clicks: Optional[int],
    calculated_data: Optional[Dict[str, Any]],
    selected_metrics: List[str],
    report_name: Optional[str],
    report_format: Optional[str],
    existing_reports: Optional[Dict[str, Any]],
) -> Tuple[Dict, str, bool, str]:
    """Generates report metadata and stores it, confirming to the user."""
    button_id = callback_context.triggered_id
    if not n_clicks or button_id != "generate-report-button":
        raise PreventUpdate

    # --- Validate Inputs --- Start
    if not calculated_data or "results" not in calculated_data:
        logger.error("Generate report: No valid metrics results found.")
        return no_update, "Error: Metrics not calculated.", True, "danger"
    all_metrics_results = calculated_data.get("results")
    dataset_id_from_calc = calculated_data.get("dataset_id")  # Get associated dataset
    if (
        not all_metrics_results
        or not isinstance(all_metrics_results, list)
        or not dataset_id_from_calc
    ):
        logger.error(
            "Generate report: Invalid metrics results structure or missing dataset_id."
        )
        return no_update, "Error: Invalid metric results.", True, "danger"
    if not selected_metrics:
        logger.warning("Generate report: No metrics selected.")
        return no_update, "Warning: No metrics selected for report.", True, "warning"
    if not report_name:
        report_name = "Thermogram_Report"
        logger.warning("No report name provided, using default.")
    if not report_format:
        logger.error("No report format selected.")
        return no_update, "Error: No report format selected.", True, "danger"
    # --- Validate Inputs --- End

    logger.info(
        f"Generating report metadata for '{report_name}' (Format: {report_format}) from dataset '{dataset_id_from_calc}'"
    )

    # --- Generate Filename & ID --- Start
    base_filename = report_name
    safe_filename_base = "".join(
        c for c in base_filename if c.isalnum() or c in ("_", "-")
    ).rstrip()
    if not safe_filename_base:
        safe_filename_base = "Thermogram_Report"
    report_filename = f"{safe_filename_base}.{report_format}"
    report_timestamp = datetime.now()
    report_id = f"{safe_filename_base}_{report_timestamp.strftime('%Y%m%d%H%M%S%f')}"
    # --- Generate Filename & ID --- End

    # --- Prepare Metadata to Store --- Start
    report_metadata = {
        "report_filename": report_filename,
        "format": report_format,
        "source_dataset_id": dataset_id_from_calc,
        "timestamp": report_timestamp.isoformat(),
        "selected_metric_keys": selected_metrics,
        # Store the actual calculated metric results needed to generate the file later
        "metrics_results": all_metrics_results,
    }
    # --- Prepare Metadata to Store --- End

    # --- Update Store --- Start
    updated_reports = (existing_reports or {}).copy()
    updated_reports[report_id] = report_metadata
    logger.info(f"Stored metadata for report ID: {report_id}")
    # --- Update Store --- End

    # --- Return updated store and confirmation message --- Start
    return (
        updated_reports,
        f"Successfully generated report '{report_filename}'. View in Data Overview.",
        True,  # Make alert visible
        "success",  # Set alert color
    )
    # --- Return updated store and confirmation message --- End

    # REMOVED: File content generation and dcc.send_...


# --- Callback to Generate Report File --- End


# --- Callback to Update Reports Overview Display --- Start
@callback(
    Output("reports-overview-display", "children"),
    Input("generated-reports-store", "data"),
    prevent_initial_call=True,
)
def update_reports_overview(
    generated_reports: Optional[Dict[str, Any]],
) -> List[dbc.ListGroupItem]:
    """Updates the overview display for generated reports."""
    if not generated_reports:
        return [dbc.ListGroupItem("No reports generated yet.")]

    items = []
    for report_id, metadata in sorted(
        generated_reports.items(), reverse=True
    ):  # Show newest first
        if not isinstance(metadata, dict):
            logger.warning(f"Skipping invalid metadata entry for report {report_id}")
            continue

        filename = metadata.get("report_filename", "Unknown Report")
        timestamp_str = metadata.get("timestamp", "N/A")
        source_dataset = metadata.get("source_dataset_id", "Unknown")
        report_format = metadata.get("format", "?").upper()

        # Format timestamp
        try:
            dt_obj = datetime.fromisoformat(timestamp_str)
            timestamp_formatted = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            timestamp_formatted = timestamp_str

        download_button = dbc.Button(
            f"Download {report_format}",
            id={"type": "download-report-btn", "index": report_id},
            color="info",
            outline=False,
            size="sm",
            className="me-2",
            # Disable PDF download for now
            disabled=(report_format == "PDF"),
        )
        delete_button = dbc.Button(
            "Delete",
            id={"type": "delete-report-btn", "index": report_id},
            color="danger",
            outline=True,
            size="sm",
        )

        item_content = dbc.Row(
            [
                dbc.Col(html.Strong(filename), width=5),
                dbc.Col(f"Source: {source_dataset}", width="auto"),
                dbc.Col(f"Created: {timestamp_formatted}", width="auto"),
                dbc.Col(
                    [download_button, delete_button], width="auto", className="ms-auto"
                ),
            ],
            className="d-flex align-items-center justify-content-between",
        )
        items.append(dbc.ListGroupItem(item_content))

    return items if items else [dbc.ListGroupItem("No reports generated yet.")]


# --- Callback to Update Reports Overview Display --- End


# --- Callback for Metric Selection Control Buttons --- Start
@callback(
    Output("report-metric-selector", "value", allow_duplicate=True),
    Input("report-metric-select-all", "n_clicks"),
    Input("report-metric-clear-all", "n_clicks"),
    Input("report-metric-reset", "n_clicks"),
    State("report-metric-selector", "options"),
    prevent_initial_call=True,
)
def update_metric_selection(
    select_all_clicks: Optional[int],
    clear_all_clicks: Optional[int],
    reset_clicks: Optional[int],
    available_options: List[Dict[str, Any]],
) -> List[str]:
    """Updates the selected metrics based on control button clicks."""
    ctx = callback_context
    if not ctx.triggered_id:
        raise PreventUpdate

    button_id = ctx.triggered_id
    logger.debug(f"Metric selection control triggered by: {button_id}")

    if button_id == "report-metric-select-all":
        if not available_options:
            return []
        all_values = [option["value"] for option in available_options]
        logger.info("Selecting all report metrics.")
        return all_values
    elif button_id == "report-metric-clear-all":
        logger.info("Clearing all report metrics selection.")
        return []
    elif button_id == "report-metric-reset":
        # Redefine defaults here or get from a central config if needed
        default_selected_metrics = [
            "TPeak_1",
            "TPeak_2",
            "TPeak_3",
            "Area",
            "FWHM",
            "TFM",
        ]
        # Ensure defaults are valid within the current options
        valid_defaults = [
            opt["value"]
            for opt in available_options
            if opt["value"] in default_selected_metrics
        ]
        logger.info(f"Resetting report metrics selection to defaults: {valid_defaults}")
        return valid_defaults
    else:
        raise PreventUpdate


# --- Callback for Metric Selection Control Buttons --- End


# --- Callback for Report Download Button --- Start
@callback(
    Output("download-report", "data"),
    Input({"type": "download-report-btn", "index": ALL}, "n_clicks"),
    State("generated-reports-store", "data"),
    prevent_initial_call=True,
)
def download_generated_report(
    n_clicks: List[Optional[int]], generated_reports: Optional[Dict[str, Any]]
) -> Optional[Dict]:
    """Handles clicks on report download buttons in the overview list."""
    triggered_id_dict = callback_context.triggered_id
    if not triggered_id_dict or not any(n for n in n_clicks if n):
        raise PreventUpdate

    report_id = triggered_id_dict.get("index")
    if not report_id or not generated_reports or report_id not in generated_reports:
        logger.error(f"Could not find report ID '{report_id}' in store for download.")
        raise PreventUpdate

    report_metadata = generated_reports[report_id]
    filename = report_metadata.get("report_filename", "report.file")
    report_format = report_metadata.get("format")
    metrics_results = report_metadata.get("metrics_results")
    selected_metrics = report_metadata.get("selected_metric_keys")

    if not report_format or not metrics_results or not selected_metrics:
        logger.error(f"Missing required metadata for downloading report '{report_id}'.")
        raise PreventUpdate

    logger.info(
        f"Download requested for report '{report_id}' ({filename}) Format: {report_format}"
    )

    # --- Prepare DataFrame --- Start
    try:
        report_df = pd.DataFrame(metrics_results)
        if "SampleID" in report_df.columns:
            cols = ["SampleID"] + [
                col
                for col in selected_metrics
                if col in report_df.columns and col != "SampleID"
            ]
        else:
            cols = [col for col in selected_metrics if col in report_df.columns]
        report_df_filtered = report_df[cols]
        report_df_renamed = report_df_filtered.rename(columns=available_metrics_dict)
    except Exception as e:
        logger.error(
            f"Error preparing report DataFrame for download: {e}", exc_info=True
        )
        raise PreventUpdate
    # --- Prepare DataFrame --- End

    # --- Generate File Content --- Start
    if report_format == "csv":
        return dcc.send_data_frame(
            report_df_renamed.to_csv, filename=filename, index=False
        )
    elif report_format == "xlsx":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            report_df_renamed.to_excel(writer, index=False, sheet_name="Metrics")
        excel_buffer.seek(0)
        return dcc.send_bytes(excel_buffer.getvalue(), filename=filename)
    elif report_format == "pdf":
        logger.error("PDF download requested but not implemented.")
        # Return alert?
        raise PreventUpdate  # Prevent download for unimplemented format
    else:
        logger.error(f"Unknown report format '{report_format}' for download.")
        raise PreventUpdate
    # --- Generate File Content --- End


# --- Callback for Report Download Button --- End


# --- Optional: Callback to Delete Report --- Start
@callback(
    Output("generated-reports-store", "data", allow_duplicate=True),
    Input({"type": "delete-report-btn", "index": ALL}, "n_clicks"),
    State("generated-reports-store", "data"),
    prevent_initial_call=True,
)
def delete_generated_report(
    n_clicks: List[Optional[int]], generated_reports: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handles clicks on report delete buttons."""
    triggered_id_dict = callback_context.triggered_id
    if (
        not triggered_id_dict
        or not any(n for n in n_clicks if n)
        or not generated_reports
    ):
        raise PreventUpdate

    report_id_to_delete = triggered_id_dict.get("index")
    if not report_id_to_delete or report_id_to_delete not in generated_reports:
        logger.warning(
            f"Attempted to delete non-existent report ID: {report_id_to_delete}"
        )
        raise PreventUpdate

    updated_reports = generated_reports.copy()
    if report_id_to_delete in updated_reports:
        del updated_reports[report_id_to_delete]
        logger.info(f"Deleted report metadata for ID: {report_id_to_delete}")

    return updated_reports


# --- Optional: Callback to Delete Report --- End
