"""
Callbacks related to baseline parameter selection, adjustment, and application.

This module includes:
- Callbacks to handle user interaction for selecting/adjusting baseline endpoints:
    - Clicking the thermogram plot (`handle_endpoint_click_ag`).
    - Directly editing flags (Reviewed, Exclude) in the AG Grid (`update_flags_from_grid`).
- UI helper callbacks (e.g., toggling advanced options).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from dash import Input, Output, State, callback, ctx, no_update

# Set up logger
logger = logging.getLogger(__name__)


# --- Grid Cell Change Callback ---
@callback(
    Output("baseline-params", "data", allow_duplicate=True),
    Input("sample-grid", "cellValueChanged"),
    State("baseline-params", "data"),
    State(
        "review-dataset-selector", "value"
    ),  # ADDED: Need filename to update correct dataset
    prevent_initial_call=True,
)
def update_flags_from_grid(
    cell_change_data: Optional[Dict[str, Any]],
    all_saved_params: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    selected_filename: Optional[str],
) -> Union[Dict[str, Dict[str, Dict[str, Any]]], type(no_update)]:
    """Updates the main baseline-params store when 'Reviewed' or 'Exclude' is changed in the AG Grid.

    Args:
        cell_change_data: Dictionary provided by AG Grid on cell value change.
                          Contains row data, column ID, and new value.
        all_saved_params: The current state of the main baseline parameter store.
        selected_filename: The filename of the currently active dataset.

    Returns:
        The updated baseline parameter store dictionary or `no_update` if change is invalid.
    """
    if not cell_change_data or not selected_filename:
        logger.debug(
            "Grid cell change triggered, but no data or no selected dataset. No update."
        )
        return no_update

    col_id = cell_change_data.get("colId")
    new_value = cell_change_data.get("value")
    row_data = cell_change_data.get("data")
    sample_id = row_data.get("sample_id") if row_data else None

    # Only process changes to 'reviewed' or 'exclude' columns
    if not sample_id or col_id not in ["reviewed", "exclude"]:
        logger.debug(
            f"Ignoring grid change in column: {col_id} for sample: {sample_id}"
        )
        return no_update

    logger.info(
        f"Grid cell changed: Sample '{sample_id}', Column '{col_id}', New Value: {new_value} in Dataset '{selected_filename}'"
    )

    # Ensure the main store exists and the dataset exists within it
    updated_params = (all_saved_params or {}).copy()
    if selected_filename not in updated_params:
        updated_params[selected_filename] = {}

    # Ensure the sample exists within the dataset's parameters
    sample_id_str = str(sample_id)
    if sample_id_str not in updated_params[selected_filename]:
        logger.warning(
            f"Attempted to update flag for sample '{sample_id_str}' which doesn't exist in baseline params for dataset '{selected_filename}'. Creating entry."
        )
        # Initialize with default if missing - might indicate an issue upstream
        updated_params[selected_filename][sample_id_str] = {
            "lower": None,
            "upper": None,
            "lower_source": "auto",
            "upper_source": "auto",
            "reviewed": False,
            "exclude": False,
        }

    # Update the specific flag (reviewed or exclude)
    if col_id in updated_params[selected_filename][sample_id_str]:
        updated_params[selected_filename][sample_id_str][col_id] = new_value
        logger.info(
            f"Updated '{col_id}' flag for sample '{sample_id_str}' in dataset '{selected_filename}' to {new_value}."
        )
        return updated_params
    else:
        logger.error(
            f"Column '{col_id}' not found in params for sample '{sample_id_str}'. Cannot update flag."
        )
        return no_update


# --- UI Helper Callbacks ---
@callback(
    Output("advanced-options-collapse", "is_open"),
    Input("advanced-options-toggle", "value"),
    State("advanced-options-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_advanced_options(toggle_value: List[int], is_open: bool) -> bool:
    """Toggles the visibility of the advanced options section in the upload modal.

    Args:
        toggle_value: List containing the value of the switch (1 if checked, empty otherwise).
        is_open: The current open/closed state of the collapse component.

    Returns:
        The new state for the `is_open` property of the collapse component.
    """
    if toggle_value:
        logger.debug("Advanced options toggled open.")
        return not is_open  # Toggle state if switch is interacted with
    logger.debug("Advanced options toggle value empty, returning current state.")
    return is_open  # Should remain closed if toggle_value is empty initially


# --- Plot Click Handler (Main Interaction) ---
@callback(
    # Outputs for temporary state
    Output("temporary-baseline-params", "data", allow_duplicate=True),
    Output("endpoint-selection-mode", "data", allow_duplicate=True),
    # Outputs for control panel display UPDATES (children only)
    Output("display-lower-endpoint", "children", allow_duplicate=True),
    Output("display-upper-endpoint", "children", allow_duplicate=True),
    # Outputs to reset button colors
    Output("select-lower-btn", "color", allow_duplicate=True),
    Output("select-upper-btn", "color", allow_duplicate=True),
    # Inputs (Listen to both plots)
    Input("raw-plot-graph", "clickData"),
    Input("processed-plot-graph", "clickData"),
    # States
    State("endpoint-selection-mode", "data"),
    State("temporary-baseline-params", "data"),
    prevent_initial_call="initial_duplicate",
)
def handle_endpoint_click_ag(
    raw_click_data: Optional[Dict[str, Any]],
    processed_click_data: Optional[Dict[str, Any]],
    current_selection_mode: Optional[str],
    current_temp_params_store: Optional[Dict[str, Any]],
) -> Tuple[
    Union[Dict[str, Any], type(no_update)],  # temporary-baseline-params
    Optional[str],  # endpoint-selection-mode (set to None)
    Union[str, type(no_update)],  # display-lower-endpoint
    Union[str, type(no_update)],  # display-upper-endpoint
    str,  # select-lower-btn color
    str,  # select-upper-btn color
]:
    """Handles clicks on EITHER the raw or processed thermogram plot.

    Updates the temporary baseline parameters based on the clicked temperature
    and the active selection mode ('lower' or 'upper'). Also updates the
    endpoint display values and resets the selection mode and button colors.

    Args:
        raw_click_data: Data from the raw plot click event.
        processed_click_data: Data from the processed plot click event.
        current_selection_mode: The active endpoint selection mode ('lower' or 'upper').
        current_temp_params_store: The current dictionary stored in `temporary-baseline-params`.

    Returns:
        A tuple containing updates for relevant components.
    """
    triggered_id = ctx.triggered_id
    logger.debug(f"handle_endpoint_click_ag triggered by: {triggered_id}")

    # Determine which plot was clicked and get its data
    active_click_data = None
    if triggered_id == "raw-plot-graph":
        active_click_data = raw_click_data
    elif triggered_id == "processed-plot-graph":
        active_click_data = processed_click_data

    # --- Input Validation --- Start
    # Check if selection mode is active, if click data exists for the triggered plot,
    # and if temp params store has data
    if (
        not current_selection_mode
        or not active_click_data
        or not current_temp_params_store
    ):
        logger.debug(
            f"Plot click ignored: No selection mode ('{current_selection_mode}'), no click data ('{bool(active_click_data)}'), or no temp params ('{bool(current_temp_params_store)}')."
        )
        # Prevent update, reset button colors to default
        return (no_update, no_update, no_update, no_update, "primary", "primary")

    sample_id = current_temp_params_store.get("sample_id")
    if not sample_id:
        logger.warning(
            "Plot click ignored: 'sample_id' missing in temporary parameters."
        )
        # Prevent update, reset button colors
        return (no_update, None, no_update, no_update, "primary", "primary")

    try:
        # Extract clicked temperature from the active plot's click data
        clicked_temp = float(active_click_data["points"][0]["x"])
    except (TypeError, KeyError, IndexError, ValueError) as e:
        logger.warning(
            f"Could not extract valid temperature from click data: {active_click_data}, Error: {e}"
        )
        # Prevent update, reset button colors
        return (no_update, None, no_update, no_update, "primary", "primary")
    # --- Input Validation --- End

    logger.info(
        f"Handling plot click for {current_selection_mode} endpoint. Clicked Temp: {clicked_temp:.2f} for Sample: {sample_id}"
    )

    # Make a copy to modify
    updated_temp_params = current_temp_params_store.copy()

    # Update the clicked endpoint and its source to 'manual'
    if current_selection_mode == "lower":
        updated_temp_params["lower"] = clicked_temp
        updated_temp_params["lower_source"] = "manual"
        logger.debug(f"Set temporary lower endpoint to {clicked_temp:.2f} (manual)")
    elif current_selection_mode == "upper":
        updated_temp_params["upper"] = clicked_temp
        updated_temp_params["upper_source"] = "manual"
        logger.debug(f"Set temporary upper endpoint to {clicked_temp:.2f} (manual)")
    else:
        logger.warning(
            f"Plot clicked with unknown selection mode: {current_selection_mode}"
        )
        # Prevent update but reset mode/colors
        return (
            no_update,
            None,
            no_update,
            no_update,
            "primary",
            "primary",
        )

    # --- Ensure Lower <= Upper --- Start
    # Check if both endpoints exist now and swap if necessary
    lower = updated_temp_params.get("lower")
    upper = updated_temp_params.get("upper")
    if lower is not None and upper is not None and lower > upper:
        logger.info(f"Swapping endpoints: Lower ({lower:.2f}) > Upper ({upper:.2f})")
        updated_temp_params["lower"], updated_temp_params["upper"] = upper, lower
        # Also swap the sources
        lower_src = updated_temp_params.get("lower_source")
        upper_src = updated_temp_params.get("upper_source")
        updated_temp_params["lower_source"], updated_temp_params["upper_source"] = (
            upper_src,
            lower_src,
        )
        # Update local variables after potential swap
        lower = updated_temp_params["lower"]
        upper = updated_temp_params["upper"]
    # --- Ensure Lower <= Upper --- End

    # --- Prepare Display Updates (Values Only) --- Start
    final_lower = updated_temp_params.get("lower")
    final_upper = updated_temp_params.get("upper")
    lower_display = f"{final_lower:.1f}" if final_lower is not None else "N/A"
    upper_display = f"{final_upper:.1f}" if final_upper is not None else "N/A"
    # --- Prepare Display Updates --- End

    logger.debug(f"Final temporary params after click: {updated_temp_params}")
    # Return updated temp store, reset mode, updated display values, reset colors
    return (
        updated_temp_params,  # temp params
        None,  # selection mode (reset)
        lower_display,  # lower display value
        upper_display,  # upper display value
        "primary",  # lower color
        "primary",  # upper color
    )
