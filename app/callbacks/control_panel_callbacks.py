"""
Callbacks for the main control panel, dataset selection, and review workflow.

Handles interactions within the 'Review Endpoints' tab, including:
- Populating the control panel based on the selected sample.
- Triggering review mode from the 'Data Overview' tab.
- Toggling the upload modal.
- Activating endpoint selection mode via buttons.
- Handling 'Discard Changes', 'Mark Reviewed & Next', 'Previous Sample' buttons.
- Loading data into the AG Grid for the selected dataset.
- Saving processed metadata.
- Updating UI elements like overviews and statuses.
"""

import logging
import os
import time
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, MATCH, Input, Output, State, callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dateutil import parser

# Need interpolation and baseline subtraction utils
from app.utils.data_processing import interpolate_thermogram
from core.baseline import subtract_spline_baseline

logger = logging.getLogger(__name__)

# --- Define namedtuple for control panel outputs ---
ControlPanelOutputs = namedtuple(
    "ControlPanelOutputs",
    [
        "style",
        "title",
        "lower_display",
        "lower_source",
        "upper_display",
        "upper_source",
        "exclude_value",
        "exclude_disabled",
        "cancel_disabled",
        "select_lower_disabled",
        "select_lower_color",
        "select_upper_disabled",
        "select_upper_color",
        "mark_reviewed_disabled",
        "previous_sample_disabled",
    ],
)


@callback(
    Output("sample-control-panel-content", "style", allow_duplicate=True),
    Output("selected-sample-display", "children", allow_duplicate=True),
    Output("display-lower-endpoint", "children", allow_duplicate=True),
    Output("lower-endpoint-source", "children", allow_duplicate=True),
    Output("display-upper-endpoint", "children", allow_duplicate=True),
    Output("upper-endpoint-source", "children", allow_duplicate=True),
    Output("edit-exclude-checkbox", "value", allow_duplicate=True),
    Output("edit-exclude-checkbox", "disabled", allow_duplicate=True),
    Output("cancel-sample-changes-btn", "disabled", allow_duplicate=True),
    Output("select-lower-btn", "disabled", allow_duplicate=True),
    Output("select-lower-btn", "color", allow_duplicate=True),
    Output("select-upper-btn", "disabled", allow_duplicate=True),
    Output("select-upper-btn", "color", allow_duplicate=True),
    Output("mark-reviewed-next-btn", "disabled", allow_duplicate=True),
    Output("previous-sample-btn", "disabled", allow_duplicate=True),
    Input("sample-grid", "selectedRows"),
    State("baseline-params", "data"),
    State("review-dataset-selector", "value"),
    prevent_initial_call=True,
)
def update_control_panel_on_selection(
    selected_rows: Optional[List[Dict[str, Any]]],
    baseline_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    selected_filename: Optional[str],
) -> ControlPanelOutputs:
    """Updates the control panel when a grid row is selected or baseline data changes.

    Determines the appropriate state for all control panel elements based on
    whether a valid sample is selected and its corresponding baseline parameters
    are available.

    Args:
        selected_rows: Data for the selected row in the AG Grid.
        baseline_data: The main baseline parameter store.
        selected_filename: The filename of the currently active dataset.

    Returns:
        A ControlPanelOutputs namedtuple containing the state for all output components.
    """
    # Define default state using the namedtuple
    default_outputs = ControlPanelOutputs(
        style={"display": "none"},
        title="Select a dataset and sample",
        lower_display="N/A",
        lower_source="Source: auto",
        upper_display="N/A",
        upper_source="Source: auto",
        exclude_value=False,
        exclude_disabled=True,
        cancel_disabled=True,
        select_lower_disabled=True,
        select_lower_color="primary",
        select_upper_disabled=True,
        select_upper_color="primary",
        mark_reviewed_disabled=True,
        previous_sample_disabled=True,
    )

    # --- Validation: Dataset and Row Selection --- Start
    if not selected_filename:
        logger.debug("No dataset selected, hiding control panel.")
        return default_outputs._replace(
            style={"display": "none"}, title="Select a dataset for review above"
        )

    if not selected_rows:
        logger.debug("No row selected, hiding control panel content.")
        return default_outputs._replace(
            style={"display": "none"}, title="Select a sample from the grid below"
        )

    selected_sample = selected_rows[0]
    sample_id = selected_sample.get("sample_id")
    if not sample_id:
        logger.warning("Selected row data missing 'sample_id'.")
        return default_outputs._replace(
            style={"display": "block"}, title="Error: Invalid row data"
        )
    # --- Validation: Dataset and Row Selection --- End

    # --- Validation: Data Availability --- Start
    if not baseline_data:
        logger.warning("Baseline data store is empty.")
        return default_outputs._replace(
            style={"display": "block"}, title="Error: Baseline store empty"
        )

    active_baseline_data = baseline_data.get(selected_filename)
    if not active_baseline_data:
        error_msg = f"Error: Baseline data not found for dataset '{selected_filename}'"
        logger.warning(f"Control panel update failed: {error_msg}")
        return default_outputs._replace(style={"display": "block"}, title=error_msg)

    sample_id_str = str(sample_id)  # Ensure string key access
    if sample_id_str not in active_baseline_data:
        error_msg = f"Error: Baseline data not found for sample '{sample_id_str}' in dataset '{selected_filename}'"
        logger.warning(f"Control panel update failed: {error_msg}")
        return default_outputs._replace(style={"display": "block"}, title=error_msg)
    # --- Validation: Data Availability --- End

    # --- Data Found: Populate Panel --- Start
    logger.info(
        f"Populating control panel for sample '{sample_id_str}' in '{selected_filename}'"
    )
    sample_params = active_baseline_data[sample_id_str]
    lower_temp = sample_params.get("lower")
    lower_source = sample_params.get("lower_source", "auto")
    upper_temp = sample_params.get("upper")
    upper_source = sample_params.get("upper_source", "auto")
    exclude = sample_params.get("exclude", False)

    # Format temps for display
    display_lower = f"{lower_temp:.1f}" if lower_temp is not None else "N/A"
    display_upper = f"{upper_temp:.1f}" if upper_temp is not None else "N/A"

    # Enable buttons now that a valid sample is selected
    # Return the populated state using the namedtuple
    return ControlPanelOutputs(
        style={"display": "block"},
        title=f"Reviewing Sample: {sample_id_str}",
        lower_display=display_lower,
        lower_source=f"Source: {lower_source}",
        upper_display=display_upper,
        upper_source=f"Source: {upper_source}",
        exclude_value=exclude,
        exclude_disabled=False,
        cancel_disabled=False,
        select_lower_disabled=False,
        select_lower_color="primary",
        select_upper_disabled=False,
        select_upper_color="primary",
        mark_reviewed_disabled=False,
        previous_sample_disabled=False,
    )
    # --- Data Found: Populate Panel --- End


@callback(
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Output("review-dataset-selector", "value"),
    Input({"type": "review-btn", "index": ALL}, "n_clicks"),  # Match any review button
    prevent_initial_call=True,
)
def handle_review_button_click(n_clicks: List[Optional[int]]) -> Tuple[str, str]:
    """Switches to Review tab and selects the corresponding dataset.

    Triggered when a 'Review Endpoints' button is clicked on the Data Overview tab.

    Args:
        n_clicks: List of click counts for all buttons matching the pattern.

    Returns:
        A tuple containing:
            - The target tab ID ('tab-review').
            - The filename (index) of the dataset to select in the dropdown.

    Raises:
        PreventUpdate: If no button click triggered the callback or if the
                       triggered button ID cannot be parsed.
    """
    # Check if any button was actually clicked
    if not ctx.triggered_id or not any(n is not None and n > 0 for n in n_clicks):
        logger.debug("handle_review_button_click: No relevant button click detected.")
        raise PreventUpdate

    # Get the context to find which specific button was clicked
    triggered_id_dict = ctx.triggered_id
    if not isinstance(triggered_id_dict, dict):
        logger.warning(
            f"Could not parse triggered ID for review button: {triggered_id_dict}"
        )
        raise PreventUpdate

    # Extract the filename (index) from the button ID
    filename = triggered_id_dict.get("index")
    if not filename:
        logger.warning(
            f"Could not get filename index from review button ID: {triggered_id_dict}"
        )
        raise PreventUpdate

    logger.info(
        f"Review button clicked for dataset: {filename}. Switching tab and selecting dataset."
    )
    return "tab-review", filename  # Set active tab and dropdown value


@callback(
    Output("upload-modal", "is_open"),
    Input("open-upload-modal-btn-overview", "n_clicks"),
    Input("open-upload-modal-btn-review", "n_clicks"),
    Input("close-upload-modal-btn", "n_clicks"),
    State("upload-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_upload_modal(
    n_open_overview: Optional[int],
    n_open_review: Optional[int],
    n_close: Optional[int],
    is_open: bool,
) -> bool:
    """Opens or closes the upload modal based on button clicks.

    Args:
        n_open_overview: Click count for the overview tab's upload button.
        n_open_review: Click count for the review tab's upload button.
        n_close: Click count for the modal's close button.
        is_open: Current state of the modal's `is_open` property.

    Returns:
        The new state for the modal's `is_open` property (True to open, False to close).
    """
    triggered_id = ctx.triggered_id
    logger.debug(f"toggle_upload_modal triggered by: {triggered_id}")

    # Check if any of the relevant buttons were clicked
    # Using ctx.triggered_id is more explicit than checking n_clicks > 0
    if triggered_id in [
        "open-upload-modal-btn-overview",
        "open-upload-modal-btn-review",
    ]:
        logger.info(f"Opening upload modal via {triggered_id}.")
        return True
    elif triggered_id == "close-upload-modal-btn":
        logger.info("Closing upload modal via close button.")
        return False

    # If triggered by something else (e.g., initial load, unrelated callback),
    # or if n_clicks were None/0, maintain the current state.
    logger.debug("Modal toggle callback fired but no relevant button press caused it.")
    return is_open


# --- Control Panel Button Callbacks ---


@callback(
    Output("endpoint-selection-mode", "data", allow_duplicate=True),
    Output("select-lower-btn", "color", allow_duplicate=True),
    Output("select-upper-btn", "color", allow_duplicate=True),
    Input("select-lower-btn", "n_clicks"),
    Input("select-upper-btn", "n_clicks"),
    State("endpoint-selection-mode", "data"),
    prevent_initial_call=True,
)
def activate_endpoint_selection(
    lower_clicks: Optional[int],
    upper_clicks: Optional[int],
    current_mode: Optional[str],
) -> Tuple[Optional[str], str, str]:
    """Activates/deactivates plot click mode for endpoint selection.

    Sets the `endpoint-selection-mode` store to 'lower', 'upper', or None.
    Highlights the active button with 'warning' color.

    Args:
        lower_clicks: Click count for the lower endpoint button.
        upper_clicks: Click count for the upper endpoint button.
        current_mode: The current value in the `endpoint-selection-mode` store.

    Returns:
        A tuple containing:
            - The new selection mode ('lower', 'upper', or None).
            - The color for the lower selection button ('primary' or 'warning').
            - The color for the upper selection button ('primary' or 'warning').
    """
    button_id = ctx.triggered_id
    new_mode: Optional[str] = None
    lower_color = "primary"
    upper_color = "primary"

    logger.debug(
        f"activate_endpoint_selection triggered by: {button_id}. Current mode: {current_mode}"
    )

    if button_id == "select-lower-btn":
        # Toggle lower mode: If already lower, turn off (None). Otherwise, turn on lower.
        new_mode = "lower" if current_mode != "lower" else None
        logger.info(f"Lower button clicked. New mode: {new_mode}")
    elif button_id == "select-upper-btn":
        # Toggle upper mode: If already upper, turn off (None). Otherwise, turn on upper.
        new_mode = "upper" if current_mode != "upper" else None
        logger.info(f"Upper button clicked. New mode: {new_mode}")
    else:
        # Should not happen if inputs are correct, but handle defensively
        logger.warning(
            f"activate_endpoint_selection triggered by unexpected ID: {button_id}"
        )
        raise PreventUpdate

    # Set button colors based on the *new* mode
    if new_mode == "lower":
        lower_color = "warning"  # Highlight active button
    elif new_mode == "upper":
        upper_color = "warning"
    # If new_mode is None, colors remain 'primary'

    return new_mode, lower_color, upper_color


# --- Discard Changes Callback ---
@callback(
    # Outputs to reset display fields
    Output("display-lower-endpoint", "children", allow_duplicate=True),
    Output("lower-endpoint-source", "children", allow_duplicate=True),
    Output("display-upper-endpoint", "children", allow_duplicate=True),
    Output("upper-endpoint-source", "children", allow_duplicate=True),
    Output("edit-exclude-checkbox", "value", allow_duplicate=True),
    # Outputs to reset button/mode states
    Output("select-lower-btn", "color", allow_duplicate=True),
    Output("select-upper-btn", "color", allow_duplicate=True),
    Output(
        "endpoint-selection-mode", "data", allow_duplicate=True
    ),  # Deactivate selection mode
    # Output for temporary params store (reset it)
    Output("temporary-baseline-params", "data", allow_duplicate=True),
    # Outputs for alert message
    Output("control-panel-alert", "children", allow_duplicate=True),
    Output("control-panel-alert", "is_open", allow_duplicate=True),
    Output("control-panel-alert", "color", allow_duplicate=True),
    # Input
    Input("cancel-sample-changes-btn", "n_clicks"),
    # States needed to retrieve the *original* saved parameters
    State("sample-grid", "selectedRows"),
    State("baseline-params", "data"),  # Main store with saved values
    State("review-dataset-selector", "value"),
    prevent_initial_call=True,
)
def discard_sample_changes(
    n_clicks: Optional[int],
    selected_rows: Optional[List[Dict[str, Any]]],
    baseline_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    selected_filename: Optional[str],
) -> Tuple[
    str,
    str,
    str,
    str,
    bool,  # Display fields
    str,
    str,
    None,  # Button colors, selection mode
    Optional[Dict[str, Any]],  # Temp params store update
    str,
    bool,
    str,  # Alert fields
]:
    """Resets control panel and temporary params to the last saved state.

    Retrieves the parameters for the selected sample from the main `baseline-params`
    store (for the active dataset) and updates the control panel display fields
    and the `temporary-baseline-params` store accordingly. Also resets selection mode.

    Args:
        n_clicks: Click count for the discard button.
        selected_rows: Data for the selected row in the AG Grid.
        baseline_data: The main store containing saved baseline parameters.
        selected_filename: The filename of the currently active dataset.

    Returns:
        A tuple containing updates for:
            - Lower endpoint display text.
            - Lower endpoint source text.
            - Upper endpoint display text.
            - Upper endpoint source text.
            - Exclude checkbox value.
            - Lower button color (reset).
            - Upper button color (reset).
            - Endpoint selection mode (None).
            - Temporary baseline parameters store (reset to saved values or None).
            - Alert message text.
            - Alert visibility (True).
            - Alert color ('warning').

    Raises:
        PreventUpdate: If button not clicked, no row/dataset selected, or data missing.
    """
    # --- Input Validation --- Start
    if not n_clicks:
        raise PreventUpdate  # Callback triggered but button not clicked

    if not selected_filename or not selected_rows:
        logger.warning("Discard changes clicked but no dataset or row selected.")
        message = "Select a dataset and sample first."
        # Return defaults for display, reset buttons/mode, clear temp, show error
        return (
            "N/A",
            "Source: -",
            "N/A",
            "Source: -",
            False,
            "primary",
            "primary",
            None,
            None,
            message,
            True,
            "danger",
        )

    sample_id = selected_rows[0].get("sample_id")
    if not sample_id:
        logger.warning("Discard changes clicked but selected row missing sample_id.")
        return (
            "N/A",
            "Source: -",
            "N/A",
            "Source: -",
            False,
            "primary",
            "primary",
            None,
            None,
            "Error: Invalid row data",
            True,
            "danger",
        )

    sample_id_str = str(sample_id)
    if (
        not baseline_data
        or selected_filename not in baseline_data
        or sample_id_str not in baseline_data[selected_filename]
    ):
        error_msg = f"Cannot discard: Original parameters not found for {sample_id_str} in {selected_filename}."
        logger.warning(error_msg)
        # Still reset displays/buttons/mode/temp, but show error
        return (
            "N/A",
            "Source: Error",
            "N/A",
            "Source: Error",
            False,
            "primary",
            "primary",
            None,
            None,
            error_msg,
            True,
            "danger",
        )
    # --- Input Validation --- End

    # --- Reset Logic --- Start
    logger.info(
        f"Discarding changes for sample '{sample_id_str}' in dataset '{selected_filename}'."
    )
    saved_params = baseline_data[selected_filename][sample_id_str]

    # Prepare display values from saved params
    lower_temp = saved_params.get("lower")
    lower_source = saved_params.get("lower_source", "auto")
    upper_temp = saved_params.get("upper")
    upper_source = saved_params.get("upper_source", "auto")
    exclude = saved_params.get("exclude", False)

    display_lower = f"{lower_temp:.1f}" if lower_temp is not None else "N/A"
    display_upper = f"{upper_temp:.1f}" if upper_temp is not None else "N/A"
    source_lower_display = f"Source: {lower_source}"
    source_upper_display = f"Source: {upper_source}"

    # Prepare the data to reset the temporary store (add sample_id)
    temp_params_reset = saved_params.copy()
    temp_params_reset["sample_id"] = sample_id_str

    message = f"Changes discarded for {sample_id_str}. Restored saved parameters."
    logger.debug(f"Resetting temporary parameters to: {temp_params_reset}")
    # --- Reset Logic --- End

    # Return display updates, reset buttons/mode, reset temp store, show success message
    return (
        display_lower,
        source_lower_display,
        display_upper,
        source_upper_display,
        exclude,
        "primary",  # Reset lower button color
        "primary",  # Reset upper button color
        None,  # Reset selection mode
        temp_params_reset,  # Update temporary store
        message,
        True,
        "warning",  # Use warning color for discard confirmation
    )


# --- Mark Reviewed & Next / Previous Callbacks ---


@callback(
    # Outputs to update the main store, grid, and selection
    Output("baseline-params", "data", allow_duplicate=True),
    Output(
        "sample-grid", "rowData", allow_duplicate=True
    ),  # Update grid row to reflect saved changes
    Output("sample-grid", "selectedRows", allow_duplicate=True),  # Select next row
    # Outputs for alert message
    Output("control-panel-alert", "children", allow_duplicate=True),
    Output("control-panel-alert", "is_open", allow_duplicate=True),
    Output("control-panel-alert", "color", allow_duplicate=True),
    # Input
    Input("mark-reviewed-next-btn", "n_clicks"),
    # States needed to get current sample info and all data
    State("sample-grid", "selectedRows"),
    State("temporary-baseline-params", "data"),  # Get the params to save
    State("edit-exclude-checkbox", "value"),  # Get the current exclude status
    State("sample-grid", "rowData"),  # Get current grid data to find next row
    State("baseline-params", "data"),  # Get the main store to update
    State("review-dataset-selector", "value"),  # Get active dataset filename
    prevent_initial_call="initial_duplicate",
)
def handle_mark_reviewed_next(
    n_clicks: Optional[int],
    selected_rows: Optional[List[Dict[str, Any]]],
    temp_params_data: Optional[Dict[str, Any]],
    exclude_checkbox_value: bool,
    grid_data: Optional[List[Dict[str, Any]]],
    baseline_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    selected_filename: Optional[str],
) -> Tuple[
    Union[Dict, type(no_update)],  # baseline-params store update
    Union[List[Dict], type(no_update)],  # grid rowData update
    Union[List[Dict], List],  # grid selectedRows update (empty list if last row)
    str,
    bool,
    str,  # Alert updates
]:
    """Saves changes for the current sample, marks as reviewed, and selects the next sample.

    Retrieves parameters from the temporary store, updates the main `baseline-params`
    store for the active dataset, updates the corresponding row in the grid's `rowData`,
    and advances the grid selection.

    Args:
        n_clicks: Click count for the button.
        selected_rows: Data for the currently selected row.
        temp_params_data: Data from the `temporary-baseline-params` store.
        exclude_checkbox_value: Current value of the exclude checkbox.
        grid_data: Current `rowData` of the AG Grid.
        baseline_data: The main baseline parameter store.
        selected_filename: Filename of the active dataset.

    Returns:
        A tuple containing updates for:
            - baseline-params store.
            - sample-grid rowData.
            - sample-grid selectedRows.
            - Alert message text.
            - Alert visibility.
            - Alert color.

    Raises:
        PreventUpdate: If button not clicked or required data is missing.
    """
    # --- Input Validation --- Start
    if not n_clicks:
        raise PreventUpdate

    if (
        not selected_filename
        or not selected_rows
        or not temp_params_data
        or not grid_data
        or not baseline_data
    ):
        logger.warning(
            "Mark Reviewed & Next clicked, but missing selected dataset, selected row, temp params, grid data, or baseline store."
        )
        message = "Cannot save: Missing required data. Select dataset and sample."
        return no_update, no_update, no_update, message, True, "danger"

    current_sample_id = temp_params_data.get("sample_id")
    if not current_sample_id:
        logger.warning("Cannot save: 'sample_id' missing in temporary parameters.")
        message = "Cannot save: Sample ID missing. Reselect sample."
        return no_update, no_update, no_update, message, True, "danger"

    if selected_filename not in baseline_data:
        logger.error(
            f"Dataset '{selected_filename}' not found in baseline store during save."
        )
        message = f"Error: Dataset '{selected_filename}' not found in store."
        return no_update, no_update, no_update, message, True, "danger"
    # --- Input Validation --- End

    # --- Save Logic --- Start
    logger.info(
        f"Saving changes for sample '{current_sample_id}' in dataset '{selected_filename}'."
    )
    updated_baseline_store = baseline_data.copy()
    if selected_filename not in updated_baseline_store:
        updated_baseline_store[selected_filename] = {}

    # Prepare parameters to save
    params_to_save = temp_params_data.copy()
    params_to_save["reviewed"] = True  # Mark as reviewed
    params_to_save["exclude"] = exclude_checkbox_value  # Update exclude status
    del params_to_save["sample_id"]  # Remove sample_id before saving in nested dict

    # Update the main store
    updated_baseline_store[selected_filename][current_sample_id] = params_to_save
    logger.debug(f"Updated baseline store for {current_sample_id}: {params_to_save}")
    # --- Save Logic --- End

    # --- Update Grid RowData --- Start
    updated_grid_data = grid_data.copy()
    row_found = False
    for i, row in enumerate(updated_grid_data):
        if row.get("sample_id") == current_sample_id:
            # Update the row in grid data with the saved values
            updated_grid_data[i] = {
                "sample_id": current_sample_id,
                "lower": params_to_save.get("lower"),
                "upper": params_to_save.get("upper"),
                "lower_source": params_to_save.get("lower_source", "auto"),
                "upper_source": params_to_save.get("upper_source", "auto"),
                "reviewed": params_to_save.get("reviewed", False),
                "exclude": params_to_save.get("exclude", False),
            }
            row_found = True
            logger.debug(f"Updated rowData for index {i} ({current_sample_id})")
            break
    if not row_found:
        logger.warning(
            f"Could not find row for {current_sample_id} in grid_data to update."
        )
        # Proceed with saving and navigation, but grid might be visually out of sync
    # --- Update Grid RowData --- End

    # --- Navigation Logic (Select Next Row) --- Start
    current_index = -1
    for i, row in enumerate(grid_data):
        if row.get("sample_id") == current_sample_id:
            current_index = i
            break

    next_selected_rows = []  # Default: deselect if last row or error
    if current_index != -1 and current_index < len(grid_data) - 1:
        next_index = current_index + 1
        next_selected_rows = [grid_data[next_index]]
        next_sample_id = grid_data[next_index].get("sample_id", "Unknown")
        logger.info(
            f"Advancing selection to next sample: {next_sample_id} (index {next_index})"
        )
    else:
        logger.info(
            "Last sample reviewed or current sample not found. Deselecting row."
        )
    # --- Navigation Logic (Select Next Row) --- End

    message = f"Saved changes for {current_sample_id} and marked as reviewed."
    alert_color = "success"

    return (
        updated_baseline_store,
        updated_grid_data,
        next_selected_rows,
        message,
        True,
        alert_color,
    )


@callback(
    Output("sample-grid", "selectedRows", allow_duplicate=True),
    Output("control-panel-alert", "children", allow_duplicate=True),
    Output("control-panel-alert", "is_open", allow_duplicate=True),
    Output("control-panel-alert", "color", allow_duplicate=True),
    Input("previous-sample-btn", "n_clicks"),
    State("sample-grid", "selectedRows"),
    State("sample-grid", "rowData"),
    prevent_initial_call="initial_duplicate",
)
def handle_previous_sample(
    n_clicks: Optional[int],
    selected_rows: Optional[List[Dict[str, Any]]],
    grid_data: Optional[List[Dict[str, Any]]],
) -> Tuple[Union[List[Dict], type(no_update)], str, bool, str]:
    """Selects the previous sample in the AG Grid.

    Args:
        n_clicks: Click count for the button.
        selected_rows: Data for the currently selected row.
        grid_data: Current `rowData` of the AG Grid.

    Returns:
        A tuple containing updates for:
            - sample-grid selectedRows.
            - Alert message text.
            - Alert visibility.
            - Alert color.

    Raises:
        PreventUpdate: If button not clicked or required data is missing.
    """
    if not n_clicks:
        raise PreventUpdate

    if not selected_rows or not grid_data:
        logger.warning("Previous Sample clicked, but no row selected or no grid data.")
        return no_update, "Select a sample first.", True, "warning"

    current_sample_id = selected_rows[0].get("sample_id")
    current_index = -1
    for i, row in enumerate(grid_data):
        if row.get("sample_id") == current_sample_id:
            current_index = i
            break

    if current_index == -1:
        logger.warning("Could not find current sample in grid data.")
        return no_update, "Error: Current sample not found.", True, "danger"

    if current_index == 0:
        logger.info("Already at the first sample.")
        return no_update, "Already at the first sample.", True, "info"

    prev_index = current_index - 1
    prev_selected_rows = [grid_data[prev_index]]
    prev_sample_id = grid_data[prev_index].get("sample_id", "Unknown")
    logger.info(f"Selecting previous sample: {prev_sample_id} (index {prev_index})")

    return (
        prev_selected_rows,
        f"Selected previous sample: {prev_sample_id}",
        True,
        "info",
    )


# --- Callback to Save Processed Data Metadata ---
@callback(
    Output("processed-datasets-store", "data", allow_duplicate=True),
    Output("control-panel-alert", "children", allow_duplicate=True),
    Output("control-panel-alert", "is_open", allow_duplicate=True),
    Output("control-panel-alert", "color", allow_duplicate=True),
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input("save-processed-data-btn", "n_clicks"),
    State("all-samples-data", "data"),
    State("baseline-params", "data"),
    State("review-dataset-selector", "value"),
    State("processed-datasets-store", "data"),
    prevent_initial_call=True,
)
def save_processed_data_to_store(
    n_clicks: Optional[int],
    all_samples_data: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]],
    saved_baseline_params: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    raw_filename: Optional[str],
    existing_processed_data: Optional[Dict[str, Any]],
) -> Tuple[Dict, str, bool, str, Union[str, type(no_update)]]:
    """Saves the final reviewed baseline parameters to the processed data store.

    Generates the key based on the raw filename and current date.
    Marks the corresponding raw dataset as processed.

    Args:
        n_clicks: Click count for the save button.
        all_samples_data: The main store containing raw data for all datasets.
        saved_baseline_params: The main store containing reviewed baseline parameters.
        raw_filename: The filename of the raw dataset being reviewed.
        existing_processed_data: The current state of the processed data store.

    Returns:
        A tuple containing updates for:
            - processed-datasets-store.
            - Alert message text.
            - Alert visibility.
            - Alert color.
            - Active main tab (switches to overview on success).

    Raises:
        PreventUpdate: If button not clicked or required data is missing.
    """
    if not n_clicks:
        raise PreventUpdate

    # --- Input Validation & Filename Generation --- Start
    if not raw_filename:
        msg = "Cannot save: No raw dataset selected."
        logger.warning(msg)
        return no_update, msg, True, "warning", no_update

    try:
        # Generate the processed filename/key
        filename_str = str(raw_filename)
        base_name = os.path.splitext(filename_str)[0]
        today_date = datetime.now().strftime("%d%m%Y")
        processed_key = f"{base_name}_processed_{today_date}.csv"
        logger.info(f"Generated processed data key: {processed_key}")
    except Exception as e:
        msg = f"Error generating processed filename for {raw_filename}: {e}"
        logger.error(msg, exc_info=True)
        return no_update, msg, True, "danger", no_update

    if not saved_baseline_params or raw_filename not in saved_baseline_params:
        msg = f"Cannot save: Baseline parameters for the current raw dataset ('{raw_filename}') not found."
        logger.warning(msg)
        return no_update, msg, True, "danger", no_update
    # --- Input Validation & Filename Generation --- End

    # --- Save Logic --- Start
    logger.info(
        f"Saving processed data for raw dataset '{raw_filename}' under key '{processed_key}'."
    )
    updated_processed_store = (existing_processed_data or {}).copy()

    # Check if overwriting
    if processed_key in updated_processed_store:
        logger.warning(
            f"Overwriting existing processed data with key: '{processed_key}'"
        )

    # Store the relevant baseline parameters under the new key
    # Add metadata about the source raw file
    baseline_params_for_file = saved_baseline_params[raw_filename]
    data_to_save = {
        "source_raw_file": raw_filename,
        "baseline_parameters": baseline_params_for_file,
        "num_samples": len(baseline_params_for_file),  # Add sample count
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        ),  # Add formatted timestamp
        # "saved_timestamp": time.time(), # Keep epoch timestamp if preferred
    }
    updated_processed_store[processed_key] = data_to_save
    logger.info(f"Successfully saved processed data under key '{processed_key}'.")
    # --- Save Logic --- End

    msg = f"Processed data saved successfully as '{processed_key}'."
    # Switch back to the overview tab after saving
    return updated_processed_store, msg, True, "success", "tab-overview"


# --- Callback to Update Processed Data Overview Display ---
@callback(
    Output("processed-data-overview-display", "children"),
    Input("processed-datasets-store", "data"),
    prevent_initial_call=True,
)
def update_processed_data_overview(
    processed_datasets: Optional[Dict[str, Any]],
) -> List[dbc.ListGroupItem]:
    """Updates the overview display for processed datasets.

    Handles both datasets saved from the review process and those uploaded directly.

    Args:
        processed_datasets: The store containing metadata about processed datasets.

    Returns:
        A list of dbc.ListGroupItem components for the processed data overview.
    """
    if not processed_datasets:
        return [dbc.ListGroupItem("No data processed yet.")]

    items = []
    for dataset_id, metadata in sorted(processed_datasets.items()):
        if not isinstance(metadata, dict):
            logger.warning(f"Skipping invalid metadata entry for {dataset_id}")
            continue

        # Determine data type
        data_type = metadata.get(
            "data_type", "saved_from_review"
        )  # Default to old type if flag missing

        # Default values
        display_name = dataset_id
        details = []
        report_disabled = False  # Default to enabled
        download_disabled = False  # Default to enabled

        # Format timestamp helper
        def format_timestamp(ts_str):
            try:
                # Use dateutil.parser for flexible ISO parsing
                dt_obj = parser.isoparse(ts_str)
                return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return ts_str  # Return original string if parsing fails

        if data_type == "uploaded_wide":
            original_filename = metadata.get("original_filename", "N/A")
            upload_time_str = metadata.get("upload_time", "N/A")
            details.append("Type: Uploaded (Wide)")
            details.append(f"Original Name: {original_filename}")
            details.append(f"Uploaded: {format_timestamp(upload_time_str)}")
            report_disabled = False
            download_disabled = False
        elif data_type == "uploaded_long":
            original_filename = metadata.get("original_filename", "N/A")
            upload_time_str = metadata.get("upload_time", "N/A")
            details.append("Type: Uploaded (Long)")
            details.append(f"Original Name: {original_filename}")
            details.append(f"Uploaded: {format_timestamp(upload_time_str)}")
            report_disabled = False
            download_disabled = False
        elif data_type == "saved_from_review":
            num_samples = metadata.get("num_samples", "N/A")
            created_at_str = metadata.get("created_at", "N/A")
            source_raw = metadata.get("source_raw_file", "Unknown")
            details.append("Type: Processed Internally")
            details.append(f"Samples: {num_samples}")
            details.append(f"Created: {format_timestamp(created_at_str)}")
            details.append(f"Source Raw File: {source_raw}")
            # Implicitly keep report_disabled=False, download_disabled=False
        else:
            # Handle unexpected structure
            logger.warning(
                f"Unrecognized structure or data_type '{data_type}' for dataset {dataset_id}"
            )
            details.append(f"Unknown format (Type: {data_type})")
            report_disabled = True  # Disable buttons for unknown format
            download_disabled = True

        # Create buttons
        report_button = dbc.Button(
            "Generate Report",
            id={"type": "go-to-report-builder-btn", "index": dataset_id},
            color="success",
            outline=False,
            size="sm",
            className="me-2",
            disabled=report_disabled,
        )
        download_button = dbc.Button(
            "Download Data",
            id={"type": "download-processed-btn", "index": dataset_id},
            color="secondary",
            outline=False,
            size="sm",
            className="me-2",
            disabled=download_disabled,  # Use disabled flag
        )

        item_content = dbc.Row(
            [
                dbc.Col(html.Strong(display_name), width=5),
                dbc.Col(", ".join(details), width="auto"),  # Combine details
                dbc.Col(
                    [report_button, download_button], width="auto", className="ms-auto"
                ),
            ],
            className="d-flex align-items-center justify-content-between",
        )

        items.append(dbc.ListGroupItem(item_content))

    return items if items else [dbc.ListGroupItem("No processed datasets found.")]


# --- Callback to Update Raw Data Status on Process ---
@callback(
    Output({"type": "raw-status", "index": MATCH}, "children"),
    Input("processed-datasets-store", "data"),
    State({"type": "raw-status", "index": MATCH}, "id"),
    prevent_initial_call=True,
)
def update_raw_data_status(
    processed_datasets: Optional[Dict[str, Any]], status_id: Dict
) -> str:
    """Updates the status display ('Processed'/'Not Processed') for raw datasets.

    Checks if a raw dataset filename (from the component ID's index)
    exists as a `source_raw_file` value within the `processed-datasets-store`.

    Args:
        processed_datasets: The dictionary stored in `processed-datasets-store`.
        status_id: The dictionary ID of the `html.Small` component being updated.
                   Contains `{"type": "raw-status", "index": <filename>}`.

    Returns:
        The status string: 'Processed' or 'Not Processed'.
    """
    if not processed_datasets:
        return "Not Processed"

    raw_filename_to_check = status_id.get("index")
    if not raw_filename_to_check:
        logger.warning(f"Could not get filename index from raw status ID: {status_id}")
        return "Error"

    is_processed = any(
        data.get("source_raw_file") == raw_filename_to_check
        for data in processed_datasets.values()
        if isinstance(data, dict)  # Check if value is a dict before accessing
    )

    status = "Processed" if is_processed else "Not Processed"
    # logger.debug(f"Status for raw file '{raw_filename_to_check}': {status}") # Can be noisy
    return status


# --- NEW Callback to Load Data for Review based on Dropdown Selection ---
@callback(
    Output("sample-grid", "rowData", allow_duplicate=True),
    Output("select-first-row-trigger", "data", allow_duplicate=True),
    Output("review-selector-message", "children", allow_duplicate=True),
    Output("review-selector-message", "is_open", allow_duplicate=True),
    Output("review-selector-message", "color", allow_duplicate=True),
    Input("review-dataset-selector", "value"),
    State("baseline-params", "data"),
    prevent_initial_call="initial_duplicate",
)
def load_data_for_review(
    selected_filename: Optional[str],
    baseline_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], int, str, bool, str]:
    """Loads AG Grid data when a dataset is selected for review.

    Constructs `rowData` for the grid based on the baseline parameters
    for the selected dataset filename. Triggers selection of the first row.
    Also updates the message below the dataset selector.

    Args:
        selected_filename: The filename selected in the dataset dropdown.
        baseline_data: The main baseline parameter store.

    Returns:
        A tuple containing:
            - List of dictionaries for the grid `rowData`.
            - A trigger value (timestamp) for `select-first-row-trigger`.
            - An informational message for the selector alert.
            - Visibility state for the selector alert.
            - Color for the selector alert.
    """
    if not selected_filename:
        logger.debug("No dataset selected, clearing grid.")
        # Return empty grid, no trigger, hide message
        # Corrected return values
        return [], no_update, "Select a dataset to load samples.", False, "info"

    if not baseline_data or selected_filename not in baseline_data:
        error_msg = f"Error: Baseline parameters not found for dataset '{selected_filename}'. Upload or reprocess data."
        logger.warning(error_msg)
        # Return empty grid, no trigger, show error message
        # Corrected return values
        return [], no_update, error_msg, True, "danger"

    logger.info(f"Loading grid data for selected dataset: {selected_filename}")
    dataset_params = baseline_data[selected_filename]
    row_data = []
    for sample_id_str, params in sorted(dataset_params.items()):
        if isinstance(params, dict):
            row_data.append(
                {
                    "sample_id": sample_id_str,
                    "lower": params.get("lower"),
                    "upper": params.get("upper"),
                    "lower_source": params.get("lower_source", "auto"),
                    "upper_source": params.get("upper_source", "auto"),
                    "reviewed": params.get("reviewed", False),
                    "exclude": params.get("exclude", False),
                }
            )
        else:
            logger.warning(
                f"Skipping invalid entry in baseline-params for {sample_id_str} in {selected_filename}"
            )

    # Trigger selection of the first row using current time as value
    # Use time.time() for trigger to ensure it changes
    trigger_value = int(time.time())
    message = f"Loaded {len(row_data)} samples for dataset: {selected_filename}"
    logger.info(message + f" (Trigger: {trigger_value})")

    # --- ADDED LOG: Log the data being returned ---\n    logger.debug(f"Returning row_data for grid: {row_data}")\n    # ----------------------------------------------\n\n    # Corrected return values\n    return row_data, trigger_value, message, True, "info"

    # Corrected return values
    return row_data, trigger_value, message, True, "info"


# --- NEW Callback to Select First Row on Trigger ---
@callback(
    Output("sample-grid", "selectedRows", allow_duplicate=True),
    Input("select-first-row-trigger", "data"),  # Triggered by load_data_for_review
    State("sample-grid", "rowData"),
    prevent_initial_call="initial_duplicate",
)
def select_first_grid_row(trigger_value, grid_data):
    """Selects the first row in the grid when the trigger store changes."""
    if not trigger_value or not grid_data:
        logger.debug(
            "First row selection trigger received, but no trigger value or grid data. No row selected."
        )
        return None  # Clear selection

    logger.info(f"Trigger {trigger_value} received. Selecting first row in grid.")
    first_row = grid_data[0]
    return [first_row]


# --- Callback to Update Default Download Filename Placeholder ---
@callback(
    Output("download-filename-input", "placeholder"),
    Input("review-dataset-selector", "value"),
    prevent_initial_call=True,
)
def update_download_filename_placeholder(selected_raw_filename: Optional[str]) -> str:
    """Updates the placeholder text for the download filename input.

    Takes the selected raw filename, removes the extension, and appends
    _processed_[DDMMYYYY].csv.

    Args:
        selected_raw_filename: The filename selected in the review dataset dropdown.

    Returns:
        The generated placeholder string.
    """
    if not selected_raw_filename:
        # Default placeholder if no file is selected
        return "select_dataset_first.csv"

    try:
        # Ensure it's a string before processing
        filename_str = str(selected_raw_filename)
        base_name = os.path.splitext(filename_str)[0]
        # Get current date formatted as DDMMYYYY
        today_date = datetime.now().strftime("%d%m%Y")
        placeholder = f"{base_name}_processed_{today_date}.csv"
        logger.debug(f"Setting download placeholder to: {placeholder}")
        return placeholder
    except Exception as e:
        logger.error(
            f"Error generating download placeholder for input '{selected_raw_filename}' (type: {type(selected_raw_filename)}): {e}",
            exc_info=True,
        )
        return "error_generating_name.csv"


# --- Callbacks for Processed Data Actions (Download, Report) ---


@callback(
    Output("download-data", "data"),
    # Use pattern-matching input for the download buttons
    Input({"type": "download-processed-btn", "index": ALL}, "n_clicks"),
    State("processed-datasets-store", "data"),
    State("all-samples-data", "data"),
    prevent_initial_call=True,
)
def download_processed_data_csv(
    n_clicks: List[Optional[int]],
    processed_datasets: Optional[Dict[str, Any]],
    all_samples_data: Optional[Dict[str, Any]],
) -> Optional[Dict]:
    """Handles clicks on 'Download Data' buttons for processed datasets.

    Generates a wide-format CSV with interpolated, baseline-subtracted data.

    Args:
        n_clicks: List of click counts for all buttons matching the pattern.
        processed_datasets: The store containing processed dataset metadata and parameters.
        all_samples_data: The store containing raw sample data and upload metadata.

    Returns:
        A dictionary suitable for dcc.Download, or None if no button was clicked
        or data is missing.
    """
    triggered_id_dict = ctx.triggered_id
    if not triggered_id_dict or not any(n for n in n_clicks if n):
        logger.debug("Download callback triggered without a specific button click.")
        raise PreventUpdate

    # Extract the dataset name (index) from the triggered button ID
    dataset_name = triggered_id_dict.get("index")
    if not dataset_name:
        logger.error("Could not determine dataset name from triggered download button.")
        raise PreventUpdate  # Or return an error message?

    logger.info(f"Download requested for processed dataset: {dataset_name}")

    # --- Data Validation and Type Check --- Start
    if not processed_datasets or dataset_name not in processed_datasets:
        logger.error(f"Processed dataset '{dataset_name}' not found in store.")
        raise PreventUpdate

    dataset_info = processed_datasets[dataset_name]
    data_type = dataset_info.get("data_type", "saved_from_review")  # Check type
    download_filename = (
        dataset_name if dataset_name.lower().endswith(".csv") else f"{dataset_name}.csv"
    )
    logger.info(f"Dataset '{dataset_name}' identified as type: '{data_type}'")

    # --- Handle Uploaded Wide --- Start
    if data_type == "uploaded_wide":  # Use correct flag
        df_json = dataset_info.get("data_json")
        original_filename = dataset_info.get("original_filename", dataset_name)
        if not df_json:
            logger.error(
                f"Missing 'data_json' for wide uploaded dataset '{dataset_name}'. Cannot download."
            )
            raise PreventUpdate
        try:
            # Load the WIDE-format DataFrame
            # orient='split' is suitable for DataFrames with index
            df_wide = pd.read_json(df_json, orient="split")
            logger.info(
                f"Loaded uploaded WIDE DataFrame '{dataset_name}' with shape {df_wide.shape}"
            )

            # --- Apply Edge Column Adjustment --- Start
            if df_wide.shape[1] >= 2:
                first_temp_col = df_wide.columns[0]
                second_temp_col = df_wide.columns[1]
                last_temp_col = df_wide.columns[-1]
                second_last_temp_col = df_wide.columns[-2]

                logger.info(
                    f"Adjusting edge columns for uploaded wide data: {first_temp_col} and {last_temp_col}"
                )
                df_wide[first_temp_col] = 0.5 * df_wide[second_temp_col]
                df_wide[last_temp_col] = 0.5 * df_wide[second_last_temp_col]
            else:
                logger.warning(
                    "Skipping edge column adjustment for uploaded wide data: < 2 temp columns."
                )
            # --- Edge Column Adjustment --- End

            # Prepare Download filename
            download_filename = (
                original_filename
                if original_filename.lower().endswith(".csv")
                else f"{original_filename}.csv"
            )
            logger.info(
                f"Preparing download for uploaded wide file '{original_filename}' as '{download_filename}'"
            )
            return dcc.send_data_frame(
                df_wide.to_csv,
                filename=download_filename,
                index=True,  # SampleID is the index
            )

        except Exception as e:
            logger.error(
                f"Error processing uploaded wide dataset '{dataset_name}' for download: {e}",
                exc_info=True,
            )
            raise PreventUpdate
    # --- Handle Uploaded Wide --- End

    # --- Handle Uploaded Long --- Start
    elif data_type == "uploaded_long":  # Add handler for long format
        df_json = dataset_info.get("data_json")
        original_filename = dataset_info.get("original_filename", dataset_name)
        if not df_json:
            logger.error(
                f"Missing 'data_json' for long uploaded dataset '{dataset_name}'. Cannot download."
            )
            raise PreventUpdate
        try:
            # Load the LONG-format DataFrame
            df_long = pd.read_json(df_json, orient="split")
            logger.info(
                f"Loaded uploaded LONG DataFrame '{dataset_name}' with shape {df_long.shape}"
            )

            # Pivot to wide format
            df_wide = df_long.pivot(
                index="SampleID", columns="Temperature", values="dCp_subtracted"
            )
            logger.info(
                f"Pivoted LONG DataFrame '{dataset_name}' to WIDE format shape: {df_wide.shape}"
            )

            # Apply Edge Column Adjustment
            if df_wide.shape[1] >= 2:
                first_temp_col = df_wide.columns[0]
                second_temp_col = df_wide.columns[1]
                last_temp_col = df_wide.columns[-1]
                second_last_temp_col = df_wide.columns[-2]

                logger.info(
                    f"Adjusting edge columns for pivoted long data: {first_temp_col} and {last_temp_col}"
                )
                df_wide[first_temp_col] = 0.5 * df_wide[second_temp_col]
                df_wide[last_temp_col] = 0.5 * df_wide[second_last_temp_col]
            else:
                logger.warning(
                    "Skipping edge column adjustment for pivoted long data: < 2 temp columns."
                )

            # Prepare Download filename
            download_filename = (
                original_filename
                if original_filename.lower().endswith(".csv")
                else f"{original_filename}.csv"
            )
            logger.info(
                f"Preparing download for pivoted long file '{original_filename}' as '{download_filename}'"
            )
            return dcc.send_data_frame(
                df_wide.to_csv,
                filename=download_filename,
                index=True,
            )

        except Exception as e:
            logger.error(
                f"Error processing or pivoting uploaded long dataset '{dataset_name}' for download: {e}",
                exc_info=True,
            )
            raise PreventUpdate
    # --- Handle Uploaded Long --- End

    # --- Process Internally Saved Data --- Start
    elif data_type == "saved_from_review":
        logger.info(
            f"Processing internally saved dataset '{dataset_name}' for download."
        )
        baseline_parameters = dataset_info.get("baseline_parameters")
        source_raw_file = dataset_info.get("source_raw_file")

        if not baseline_parameters or not source_raw_file:
            logger.error(
                f"Missing baseline parameters or source file info for '{dataset_name}'. Cannot download."
            )
            raise PreventUpdate

        if not all_samples_data or source_raw_file not in all_samples_data:
            logger.error(
                f"Source raw data file '{source_raw_file}' not found for '{dataset_name}'. Cannot download."
            )
            raise PreventUpdate

        raw_file_data = all_samples_data[source_raw_file]
        raw_samples_dict = raw_file_data.get("samples")
        if not raw_samples_dict:
            logger.error(
                f"'samples' key missing in raw data for '{source_raw_file}'. Cannot download."
            )
            raise PreventUpdate
        # --- Data Validation --- End (Moved validation inside the type check)

        # --- Data Processing for Download --- Start (Existing logic)
        # ... (Keep the existing multi-pass processing and interpolation logic here) ...
        # First pass: Determine the overall temperature range
        logger.debug("Download: First pass to determine overall temperature range.")
        min_temp_overall = np.inf
        max_temp_overall = -np.inf
        valid_sample_count = 0
        for sample_id, params in baseline_parameters.items():
            raw_data_list = raw_samples_dict.get(sample_id)
            if not raw_data_list:
                continue  # Skip samples without raw data
            try:
                df_raw = pd.DataFrame(raw_data_list)
                if df_raw.empty or "Temperature" not in df_raw or "dCp" not in df_raw:
                    continue  # Skip invalid/empty raw data

                # Find min/max for this sample
                current_min = df_raw["Temperature"].min()
                current_max = df_raw["Temperature"].max()

                if not pd.isna(current_min):
                    min_temp_overall = min(min_temp_overall, current_min)
                if not pd.isna(current_max):
                    max_temp_overall = max(max_temp_overall, current_max)
                valid_sample_count += 1  # Count samples with valid temp range
            except Exception as e:
                logger.warning(
                    f"Download Range Check: Error processing sample '{sample_id}': {e}"
                )

        if (
            not np.isfinite(min_temp_overall)
            or not np.isfinite(max_temp_overall)
            or valid_sample_count == 0
        ):
            logger.error(
                "Could not determine a valid overall temperature range for interpolation."
            )
            raise PreventUpdate

        # Define the interpolation grid using np.arange for 0.1 steps
        # Add a small epsilon to max_temp_overall to ensure inclusion if it's a multiple of step
        step = 0.1
        interpolated_temps_raw = np.arange(
            min_temp_overall, max_temp_overall + step / 2, step
        )
        # Round the grid to 1 decimal place to fix floating point inaccuracies
        interpolated_temps = np.round(interpolated_temps_raw, 1)
        logger.info(
            f"Generated download interpolation grid: {len(interpolated_temps)} points from {interpolated_temps[0]:.1f} to {interpolated_temps[-1]:.1f} with step {step}"
        )

        # Second pass: Process each sample and interpolate onto the common grid
        logger.debug("Download: Second pass to process and interpolate samples.")
        # Initialize dictionary to hold data for the final DataFrame
        data_for_final_df = {"Temperature": interpolated_temps}

        for sample_id, params in baseline_parameters.items():
            logger.debug(
                f"Processing sample '{sample_id}' for download (interpolation phase)."
            )
            raw_data_list = raw_samples_dict.get(sample_id)
            if not raw_data_list:
                logger.warning(
                    f"Raw data not found for sample '{sample_id}' in second pass. Skipping."
                )
                continue

            try:
                df_raw = pd.DataFrame(raw_data_list)
                if df_raw.empty or "Temperature" not in df_raw or "dCp" not in df_raw:
                    logger.warning(
                        f"Raw data for '{sample_id}' is invalid or empty in second pass. Skipping."
                    )
                    continue

                # 1. Apply Baseline Subtraction
                df_subtracted_with_baseline = subtract_spline_baseline(
                    df_raw,
                    lower_endpoint=params.get("lower"),
                    upper_endpoint=params.get("upper"),
                )
                if (
                    df_subtracted_with_baseline is None
                    or df_subtracted_with_baseline.empty
                ):
                    logger.warning(
                        f"Spline baseline subtraction failed for '{sample_id}'. Skipping."
                    )
                    continue

                # Prepare DataFrame for interpolation
                df_for_interp = df_subtracted_with_baseline[
                    ["Temperature", "dCp_subtracted"]
                ].rename(columns={"dCp_subtracted": "dCp"})

                # 2. Interpolate onto the COMMON grid
                df_interpolated = interpolate_thermogram(
                    df_for_interp,
                    temp_grid=interpolated_temps,  # Use the common grid
                )

                if df_interpolated is None or df_interpolated.empty:
                    logger.warning(f"Interpolation failed for '{sample_id}'. Skipping.")
                    continue

                # Add the interpolated dCp numpy array to the dictionary
                if "dCp" in df_interpolated.columns and len(
                    df_interpolated["dCp"]
                ) == len(interpolated_temps):
                    data_for_final_df[sample_id] = df_interpolated["dCp"].to_numpy()
                    logger.debug(
                        f"Successfully processed and interpolated '{sample_id}' for download."
                    )
                else:
                    logger.warning(
                        f"'dCp' column missing or length mismatch after interpolation for '{sample_id}'. Skipping."
                    )

            except Exception as e:
                logger.error(
                    f"Error processing sample '{sample_id}' for download (interpolation phase): {e}",
                    exc_info=True,
                )

        # Check if any samples were successfully processed
        if len(data_for_final_df) <= 1:
            logger.error(
                "No samples could be processed successfully for download after interpolation."
            )
            raise PreventUpdate

        # Combine into Wide DataFrame
        logger.info("Combining processed samples into final DataFrame for download.")
        try:
            df_final = pd.DataFrame(data_for_final_df)
            df_final = df_final.set_index("Temperature")
            df_wide = df_final.transpose()
            df_wide.index.name = "SampleID"
        except Exception as e:
            logger.error(
                f"Error creating final DataFrame for download: {e}", exc_info=True
            )
            raise PreventUpdate

        logger.info(
            f"Generated wide DataFrame for download with shape: {df_wide.shape}"
        )
        # --- Edge Column Adjustment --- Start (Existing logic)
        if df_wide.shape[1] >= 2:
            first_temp_col = df_wide.columns[0]
            second_temp_col = df_wide.columns[1]
            last_temp_col = df_wide.columns[-1]
            second_last_temp_col = df_wide.columns[-2]

            logger.info(
                f"Adjusting edge columns for saved data: {first_temp_col} and {last_temp_col}"
            )
            df_wide[first_temp_col] = 0.5 * df_wide[second_temp_col]
            df_wide[last_temp_col] = 0.5 * df_wide[second_last_temp_col]
        else:
            logger.warning(
                "Skipping edge column adjustment: DataFrame has less than 2 temperature columns."
            )
        # --- Edge Column Adjustment --- End

        # --- Prepare Download --- Start (Existing logic)
        return dcc.send_data_frame(
            df_wide.to_csv,
            filename=download_filename,
            index=True,
        )
        # --- Prepare Download --- End

    else:
        logger.error(
            f"Unknown data_type '{data_type}' for dataset '{dataset_name}'. Cannot download."
        )
        raise PreventUpdate
