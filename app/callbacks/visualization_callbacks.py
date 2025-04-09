"""
Callbacks related to thermogram visualization.

This module defines functions for creating Plotly figures for raw and processed
thermogram data and includes callbacks triggered by user interactions like selecting
a dataset, selecting a sample from the grid, changing plot tabs, and updating
temporary baseline parameters via plot clicks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

# Utils
from app.utils.debug_utils import debug_callback
# from app.utils.plotting_utils import (
#     # create_processed_figure, # Defined locally
#     # create_raw_thermogram_figure, # Defined locally
# )
# Core baseline functions
from core.baseline import subtract_spline_baseline  # Import the new function

# from core.baseline import simple_baseline_subtraction # Keep old import commented if needed

# Set up logger
logger = logging.getLogger(__name__)

# --- Figure Creation Functions ---


def create_raw_thermogram_figure(
    df_raw: Optional[pd.DataFrame],
    title: str = "Raw Thermogram",
    baseline_params: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Creates a Plotly figure for the raw thermogram data.

    Args:
        df_raw: DataFrame containing the raw thermogram data with columns
                'Temperature' and 'dCp'. Can be None or empty.
        title: The title for the plot.
        baseline_params: Dictionary containing baseline parameters ('lower', 'upper',
                         'lower_source', 'upper_source') to draw vertical lines
                         and annotations. Can be None.

    Returns:
        A Plotly Figure object representing the raw thermogram.
    """
    fig = go.Figure()
    min_y: Optional[float] = None
    max_y: Optional[float] = None

    if (
        df_raw is not None
        and not df_raw.empty
        and "Temperature" in df_raw
        and "dCp" in df_raw
    ):
        fig.add_trace(
            go.Scatter(
                x=df_raw["Temperature"],
                y=df_raw["dCp"],
                mode="lines",
                name="Raw dCp",
            )
        )
        min_y = df_raw["dCp"].min()
        max_y = df_raw["dCp"].max()
    else:
        logger.debug("No valid raw data to plot, creating empty figure.")
        # Create empty plot area if no data
        min_y, max_y = 0, 1  # Placeholder Y range
        fig.add_trace(go.Scatter(x=[], y=[], name="No Data"))  # Add trace for legend

    # Add baseline endpoint lines using provided baseline_params
    if baseline_params:
        lower_temp = baseline_params.get("lower")
        upper_temp = baseline_params.get("upper")
        lower_source = baseline_params.get("lower_source", "auto")  # Default to auto
        upper_source = baseline_params.get("upper_source", "auto")  # Default to auto

        lower_line_color = "red" if lower_source == "manual" else "grey"
        upper_line_color = "red" if upper_source == "manual" else "grey"
        line_width = 1.5

        # Add lower line/annotation
        if lower_temp is not None:
            fig.add_shape(
                type="line",
                x0=lower_temp,
                x1=lower_temp,
                y0=0,
                y1=1,
                yref="paper",  # Reference paper to span full height
                line=dict(color=lower_line_color, width=line_width, dash="dash"),
                name="baseline-start-line",
            )
            fig.add_annotation(
                x=lower_temp,
                y=1.05,  # Position above plot area
                yref="paper",
                text=f"{lower_temp:.1f}°C",
                showarrow=False,
                font=dict(color=lower_line_color),
                name="lower-annotation",  # Add name for potential updates
            )
        # Add upper line/annotation
        if upper_temp is not None:
            fig.add_shape(
                type="line",
                x0=upper_temp,
                x1=upper_temp,
                y0=0,
                y1=1,
                yref="paper",  # Reference paper to span full height
                line=dict(color=upper_line_color, width=line_width, dash="dash"),
                name="baseline-end-line",
            )
            fig.add_annotation(
                x=upper_temp,
                y=1.05,  # Position above plot area
                yref="paper",
                text=f"{upper_temp:.1f}°C",
                showarrow=False,
                font=dict(color=upper_line_color),
                name="upper-annotation",  # Add name for potential updates
            )

    # Update layout
    y_range = None
    if min_y is not None and max_y is not None:
        padding = (max_y - min_y) * 0.05  # Add 5% padding
        y_range = [min_y - padding, max_y + padding]

    fig.update_layout(
        title=title,
        xaxis_title="Temperature (°C)",
        yaxis_title="Excess Heat Capacity (dCp)",
        template="plotly_white",
        dragmode="pan",  # Enable panning by default
        margin=dict(l=40, r=20, t=60, b=40),
        yaxis_range=y_range,
    )
    return fig


def create_processed_figure(
    df_processed: Optional[pd.DataFrame],
    baseline_params: Optional[Dict[str, Any]] = None,
    title: str = "Baseline Subtracted Thermogram",
) -> go.Figure:
    """Creates a Plotly figure for the baseline-subtracted thermogram data.

    Args:
        df_processed: DataFrame containing the processed data with columns
                      'Temperature', 'dCp_subtracted'. Can be None or empty.
        baseline_params: Dictionary containing baseline parameters ('lower', 'upper',
                         'lower_source', 'upper_source') to add endpoint
                         annotations. Can be None.
        title: The title for the plot.

    Returns:
        A Plotly Figure object representing the processed thermogram.
    """
    fig = go.Figure()
    plot_suffix = ""
    min_y: Optional[float] = None
    max_y: Optional[float] = None

    if df_processed is not None and not df_processed.empty:
        try:
            # Check if expected columns exist
            required_cols = ["Temperature", "dCp_subtracted"]
            if not all(col in df_processed.columns for col in required_cols):
                logger.warning("Processed data missing required columns for plotting.")
                # Create empty plot with warning
                fig.add_trace(go.Scatter(x=[], y=[], name="Missing Data"))
                plot_suffix = " (Plotting Error: Missing Data)"
            else:
                # Plot baseline subtracted data
                fig.add_trace(
                    go.Scatter(
                        x=df_processed["Temperature"],
                        y=df_processed["dCp_subtracted"],
                        mode="lines",
                        name="Baseline Subtracted dCp",
                    )
                )
                min_y = df_processed["dCp_subtracted"].min()
                max_y = df_processed["dCp_subtracted"].max()
        except Exception as e:
            logger.error(f"Error plotting processed data: {e}", exc_info=True)
            fig.add_trace(go.Scatter(x=[], y=[], name="Plotting Error"))
            plot_suffix = " (Plotting Error)"  # Keep suffix concise
    else:
        logger.debug("No valid processed data to plot, creating empty figure.")
        # Create empty plot area if no data
        min_y, max_y = -1, 1  # Placeholder Y range for processed plot
        fig.add_trace(go.Scatter(x=[], y=[], name="No Data"))
        plot_suffix = " (No Data)"

    # --- ADD Endpoint Annotations --- Start
    if baseline_params:
        lower_temp = baseline_params.get("lower")
        upper_temp = baseline_params.get("upper")
        lower_source = baseline_params.get("lower_source", "auto")
        upper_source = baseline_params.get("upper_source", "auto")

        lower_color = "red" if lower_source == "manual" else "grey"
        upper_color = "red" if upper_source == "manual" else "grey"

        # Add lower annotation (position slightly above 0)
        if lower_temp is not None:
            fig.add_annotation(
                x=lower_temp,
                y=0.05,  # Position slightly above baseline
                yref="paper",
                text=f"{lower_temp:.1f}°C",
                showarrow=False,
                font=dict(color=lower_color, size=10),  # Smaller font
                bgcolor="rgba(255,255,255,0.7)",  # Slight background for readability
                name="lower-annotation",  # Add name
            )
        # Add upper annotation
        if upper_temp is not None:
            fig.add_annotation(
                x=upper_temp,
                y=0.05,  # Position slightly above baseline
                yref="paper",
                text=f"{upper_temp:.1f}°C",
                showarrow=False,
                font=dict(color=upper_color, size=10),
                bgcolor="rgba(255,255,255,0.7)",
                name="upper-annotation",  # Add name
            )
    # --- ADD Endpoint Annotations --- End

    # Update layout
    y_range = None
    if min_y is not None and max_y is not None:
        padding = abs(max_y - min_y) * 0.05  # Add 5% padding, handle negative min_y
        y_range = [min_y - padding, max_y + padding]

    fig.update_layout(
        title=title + plot_suffix,
        xaxis_title="Temperature (°C)",
        yaxis_title="Baseline Subtracted dCp",
        template="plotly_white",
        dragmode="pan",
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_range=y_range,
    )
    return fig


# --- Dataset Selection Callbacks ---
# Callback to update dataset selector options
@debug_callback
@callback(
    Output("review-dataset-selector", "options"),
    Output("review-dataset-selector", "value", allow_duplicate=True),
    Input("all-samples-data", "data"),
    State("review-dataset-selector", "value"),
    State("review-dataset-selector", "options"),
    prevent_initial_call=True,
)
def update_dataset_selector_options(
    all_samples_data: Optional[Dict[str, Any]],
    current_value: Optional[str],
    previous_options: Optional[List[Dict[str, str]]],
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Updates the dataset selector dropdown options and selects newly added dataset.

    Args:
        all_samples_data: The main store containing data for all uploaded files (datasets).
        current_value: The currently selected value in the dropdown.
        previous_options: The previous list of options in the dropdown.

    Returns:
        A tuple containing:
            - A list of dictionaries for the dropdown options.
            - The value for the dropdown (newly added dataset or current if still valid).
    """
    if not all_samples_data:
        logger.info("No datasets found in store, returning empty options and no value.")
        return [], None

    # Get current and previous dataset names
    new_dataset_names = set(all_samples_data.keys())
    previous_dataset_names = (
        set(opt["value"] for opt in previous_options) if previous_options else set()
    )

    # Create new options list (sorted)
    sorted_names = sorted(list(new_dataset_names))
    options = [{"label": name, "value": name} for name in sorted_names]
    logger.info(f"Updating dataset selector options: {sorted_names}")

    # Determine the value to set
    new_value = None
    newly_added_datasets = new_dataset_names - previous_dataset_names

    if newly_added_datasets:
        # If new datasets were added, select the first new one (alphabetically)
        new_value = sorted(list(newly_added_datasets))[0]
        logger.info(
            f"New dataset(s) detected: {newly_added_datasets}. Selecting: {new_value}"
        )
    elif current_value in new_dataset_names:
        # If no new datasets, keep current value if still valid
        new_value = current_value
        logger.debug(f"No new datasets. Keeping current value: {new_value}")
    else:
        # If current value is no longer valid (and no new datasets), set to None
        logger.info(
            f"Current selector value '{current_value}' is no longer valid and no new datasets added. Resetting value."
        )
        new_value = None

    return options, new_value


# Callback to show/hide the main review content area and update message
@debug_callback
@callback(
    Output("review-selector-message", "children"),
    Output("review-selector-message", "is_open"),
    Output("review-content-area", "style"),
    Input("review-dataset-selector", "value"),
    State("review-content-area", "style"),
    prevent_initial_call="initial_duplicate",
)
def update_review_panel_visibility(
    selected_dataset: Optional[str], current_style: Optional[Dict]
) -> Tuple[str, bool, Dict]:
    """Shows or hides the review panel based on dataset selection.

    Args:
        selected_dataset: The filename of the dataset selected in the dropdown.
        current_style: The current style dictionary of the review content area.

    Returns:
        A tuple containing:
            - A message for the alert below the selector.
            - A boolean indicating whether the alert should be open.
            - The updated style dictionary for the review content area.
    """
    new_style = current_style or {}
    if selected_dataset:
        logger.info(f"Dataset '{selected_dataset}' selected. Showing review panel.")
        if new_style.get("display") == "none":
            return f"Reviewing dataset: {selected_dataset}", True, {"display": "block"}
        else:
            # Already visible, just update message potentially
            return f"Reviewing dataset: {selected_dataset}", True, no_update
    else:
        logger.info("No dataset selected. Hiding review panel.")
        if new_style.get("display") != "none":
            return (
                "Select a dataset from the dropdown to begin review.",
                True,
                {"display": "none"},
            )
        else:
            # Already hidden
            return no_update, False, no_update


# --- Row Selection Callback (AG Grid) --- Split Callbacks Approach ---


@debug_callback
@callback(
    Output("raw-plot-graph", "figure", allow_duplicate=True),
    Output("processed-plot-graph", "figure", allow_duplicate=True),
    Output("plot-tabs", "active_tab"),
    Output("raw-plot-content", "style", allow_duplicate=True),
    Output("processed-plot-content", "style", allow_duplicate=True),
    Input("sample-grid", "selectedRows"),
    State("all-samples-data", "data"),
    State("baseline-params", "data"),
    State("review-dataset-selector", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_plots_on_grid_row_select(
    selected_rows: Optional[List[Dict[str, Any]]],
    all_samples_data: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]],
    saved_baseline_params: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    selected_filename: Optional[str],
) -> Tuple[go.Figure, go.Figure, str, Dict, Dict]:
    """Update plots, active tab, and visibility when a row is selected.

    Generates raw and processed plots based on saved baseline parameters.
    Sets the active plot tab to 'tab-processed' and updates plot div styles.

    Args:
        selected_rows: ...
        all_samples_data: ...
        saved_baseline_params: ...
        selected_filename: ...

    Returns:
        A tuple containing:
            - Plotly figure for the raw thermogram.
            - Plotly figure for the processed thermogram.
            - The active plot tab ID (always "tab-processed").
            - Style dictionary for the raw plot container (hidden).
            - Style dictionary for the processed plot container (visible).
    """
    logger.debug("update_plots_on_grid_row_select triggered.")
    # Define styles
    style_visible = {"display": "block", "height": "45vh"}
    style_hidden = {"display": "none", "height": "45vh"}
    default_tab = "tab-processed"  # Always default to processed

    # Define default empty figures
    empty_fig = go.Figure().update_layout(
        title="Select a sample", template="plotly_white", height=400
    )
    empty_processed_fig = go.Figure().update_layout(
        title="Select a sample", template="plotly_white", height=400
    )

    # --- Input Validation --- Start
    if not selected_filename or not selected_rows:
        logger.debug(
            "update_plots: No dataset or row selected. Returning empty figs and default view."
        )
        return empty_fig, empty_processed_fig, default_tab, style_hidden, style_visible

    sample_id = selected_rows[0].get("sample_id")
    if (
        not sample_id
        or not all_samples_data
        or selected_filename not in all_samples_data
        or not saved_baseline_params
        or selected_filename not in saved_baseline_params
    ):
        logger.warning(
            "update_plots: Missing required data for plotting. Returning empty figs and default view."
        )
        return empty_fig, empty_processed_fig, default_tab, style_hidden, style_visible

    # --- Data Extraction --- Start (Adjusted based on stored format)
    file_data = all_samples_data.get(selected_filename, {})
    if not isinstance(file_data, dict):
        logger.warning(
            f"update_plots: Invalid format for file_data '{selected_filename}'"
        )
        return empty_fig, empty_processed_fig, default_tab, style_hidden, style_visible

    samples_dict = file_data.get("samples", {})  # Samples are under the 'samples' key
    raw_sample_data_list = samples_dict.get(str(sample_id))
    # --- Data Extraction --- End

    dataset_params = saved_baseline_params.get(selected_filename, {})
    current_sample_params = dataset_params.get(str(sample_id))

    if not raw_sample_data_list or not current_sample_params:
        logger.warning(
            f"update_plots: Missing raw data list or params for {sample_id}. Returning empty figs and default view."
        )
        return empty_fig, empty_processed_fig, default_tab, style_hidden, style_visible

    try:
        df_raw_sample = pd.DataFrame(raw_sample_data_list)
        if not all(col in df_raw_sample.columns for col in ["Temperature", "dCp"]):
            raise ValueError("Missing columns in raw data")
    except Exception as e:
        logger.error(f"update_plots: Error creating DataFrame for {sample_id}: {e}")
        return empty_fig, empty_processed_fig, default_tab, style_hidden, style_visible
    # --- Input Validation --- End

    # --- Generate Plots --- Start
    logger.info(f"update_plots: Generating plots for sample '{sample_id}'")
    logger.debug(f"Data for raw plot: {df_raw_sample.head()} ...")
    logger.debug(f"Params for raw plot: {current_sample_params}")
    fig_raw = create_raw_thermogram_figure(
        df_raw=df_raw_sample,
        title=f"Raw Thermogram: {sample_id}",
        baseline_params=current_sample_params,
    )
    try:
        logger.debug(f"Attempting baseline subtraction for {sample_id}")
        # Use the new spline baseline subtraction
        df_processed_with_baseline = subtract_spline_baseline(
            df_raw_sample,
            lower_endpoint=current_sample_params.get("lower"),
            upper_endpoint=current_sample_params.get("upper"),
            # Pass other params like smooth factor if needed/available
        )

        if (
            df_processed_with_baseline is not None
            and not df_processed_with_baseline.empty
        ):
            logger.debug(
                f"Data for processed plot: {df_processed_with_baseline.head()} ..."
            )
            logger.debug(f"Params for processed plot: {current_sample_params}")
            # create_processed_figure likely expects 'dCp_subtracted'
            fig_processed = create_processed_figure(
                df_processed=df_processed_with_baseline,  # Pass the result
                baseline_params=current_sample_params,
                title=f"Processed: {sample_id}",
            )
        else:
            logger.warning(
                f"Spline baseline subtraction failed or returned empty for {sample_id}"
            )
            fig_processed = go.Figure().update_layout(
                title=f"Processing Error: {sample_id}",
                template="plotly_white",
                height=400,
            )

    except Exception as e:
        logger.error(
            f"update_plots: Error during spline baseline processing for {sample_id}: {e}",
            exc_info=True,
        )
        fig_processed = go.Figure().update_layout(
            title=f"Processing Error: {sample_id}", template="plotly_white", height=400
        )
    # --- Generate Plots --- End

    logger.debug("update_plots: Returning generated figures and default view.")
    # Return figures, default tab, and styles
    return fig_raw, fig_processed, default_tab, style_hidden, style_visible


@debug_callback
@callback(
    Output("temporary-baseline-params", "data", allow_duplicate=True),
    Input("sample-grid", "selectedRows"),
    State("baseline-params", "data"),
    State("review-dataset-selector", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_stores_on_grid_row_select(
    selected_rows: Optional[List[Dict[str, Any]]],
    saved_baseline_params: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    selected_filename: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Update temporary baseline params store ONLY when a row is selected.

    Loads saved baseline parameters into the temporary store.

    Args: (Simplified vs combined callback)
        selected_rows: ...
        saved_baseline_params: ...
        selected_filename: ...

    Returns:
        Dictionary of baseline parameters for the selected sample (or None).
    """
    # --- Input Validation --- Start
    if not selected_filename or not selected_rows:
        logger.debug("update_stores: No dataset or row selected.")
        return None

    sample_id = selected_rows[0].get("sample_id")
    if (
        not sample_id
        or not saved_baseline_params
        or selected_filename not in saved_baseline_params
    ):
        logger.warning("update_stores: Missing required data for store update.")
        return None

    dataset_params = saved_baseline_params.get(selected_filename, {})
    current_sample_params = dataset_params.get(str(sample_id))

    if not current_sample_params:
        logger.warning(f"update_stores: Saved params not found for {sample_id}.")
        return None
    # --- Input Validation --- End

    # --- Store current params in temporary store --- Start
    temp_params_to_store = current_sample_params.copy()
    temp_params_to_store["sample_id"] = str(sample_id)
    logger.debug(f"update_stores: Storing temporary parameters: {temp_params_to_store}")
    # --- Store current params in temporary store --- End

    return temp_params_to_store


# --- Separate callback to reset endpoint selection mode ---
@debug_callback
@callback(
    Output("endpoint-selection-mode", "data", allow_duplicate=True),
    Input("sample-grid", "selectedRows"),
    prevent_initial_call="initial_duplicate",
)
def reset_selection_mode_on_grid_row_select(
    _selected_rows: Optional[List[Dict[str, Any]]],
) -> None:
    """Resets the endpoint selection mode whenever a new grid row is selected.

    This ensures that clicking the grid cancels any active endpoint selection.

    Args:
        _selected_rows: Input trigger (value not used).

    Returns:
        None: Always resets the endpoint-selection-mode store to None.
    """
    logger.debug("Resetting endpoint selection mode due to grid row selection.")
    return None


# --- Callback to Update Raw Plot Based on Temporary Parameters ---
@debug_callback
@callback(
    Output("raw-plot-graph", "figure", allow_duplicate=True),
    Output("processed-plot-graph", "figure", allow_duplicate=True),
    Input("temporary-baseline-params", "data"),
    State("all-samples-data", "data"),
    State("review-dataset-selector", "value"),
    State("raw-plot-graph", "figure"),
    State("processed-plot-graph", "figure"),
    prevent_initial_call="initial_duplicate",
)
def update_raw_plot_from_temp_params(
    temp_params_data: Optional[Dict[str, Any]],
    all_samples_data: Optional[Dict[str, Dict[str, Any]]],
    selected_filename: Optional[str],
    current_raw_figure_dict: Optional[Dict],
    current_processed_figure_dict: Optional[Dict],
) -> Tuple[Union[go.Figure, type(no_update)], Union[go.Figure, type(no_update)]]:
    """Updates BOTH raw and processed plots when temporary baseline parameters change.

    Triggered after a user clicks on a plot to set a new endpoint.
    Retrieves raw data, redraws the raw plot with updated annotations, recalculates
    the baseline subtraction using temporary params, and redraws the processed plot.

    Args:
        temp_params_data: Dict from `temporary-baseline-params` store.
        all_samples_data: Main data store containing samples and metadata.
        selected_filename: Filename of the currently selected dataset.
        current_raw_figure_dict: Current state of the raw plot figure dict.
        current_processed_figure_dict: Current state of the processed plot figure dict.

    Returns:
        Tuple containing updated figures for the raw and processed plots, or `no_update`.
    """
    if not temp_params_data:
        logger.debug("Temporary parameters cleared, preventing plot updates.")
        raise PreventUpdate

    sample_id = temp_params_data.get("sample_id")

    # --- Input Validation --- Start
    if not sample_id:
        logger.warning("'sample_id' not found in temp params. Cannot update plots.")
        raise PreventUpdate
    if not selected_filename:
        logger.warning("No dataset selected. Cannot fetch raw data to update plots.")
        raise PreventUpdate
    if not all_samples_data or selected_filename not in all_samples_data:
        logger.warning(
            f"Raw data store missing '{selected_filename}'. Cannot update plots."
        )
        raise PreventUpdate
    # --- Input Validation --- End

    logger.info(
        f"Updating plots for '{sample_id}' using temp params: {temp_params_data}"
    )

    # --- Get Raw Data --- Start (Corrected Access)
    file_data = all_samples_data.get(selected_filename, {})
    if not isinstance(file_data, dict):
        logger.warning(f"Invalid format for file_data '{selected_filename}' in store.")
        raise PreventUpdate
    samples_dict = file_data.get("samples", {})
    raw_sample_data_list = samples_dict.get(str(sample_id))

    if not raw_sample_data_list:
        logger.warning(
            f"Raw data for sample '{sample_id}' not found in dataset '{selected_filename}' when updating from temp params."
        )
        raise PreventUpdate

    try:
        df_raw_sample = pd.DataFrame(raw_sample_data_list)
        if not all(col in df_raw_sample.columns for col in ["Temperature", "dCp"]):
            raise ValueError("Missing columns in raw data")
    except Exception as e:
        logger.error(f"Error creating DataFrame for '{sample_id}' (temp update): {e}")
        raise PreventUpdate
    # --- Get Raw Data --- End

    # --- Generate Updated Raw Plot --- Start
    fig_raw_updated = create_raw_thermogram_figure(
        df_raw=df_raw_sample,
        title=f"Raw Thermogram: {sample_id} (Adjusting)",
        baseline_params=temp_params_data,
    )
    # Preserve zoom/pan from the previous figure state if possible
    if current_raw_figure_dict and "layout" in current_raw_figure_dict:
        previous_layout = current_raw_figure_dict["layout"]
        if "xaxis.range" in previous_layout:
            fig_raw_updated.update_layout(xaxis_range=previous_layout["xaxis.range"])
        if "yaxis.range" in previous_layout:
            fig_raw_updated.update_layout(yaxis_range=previous_layout["yaxis.range"])
        logger.debug("Preserved layout range from previous raw figure.")

    # --- Generate Updated Processed Plot --- Start
    fig_processed_updated = go.Figure()  # Default empty figure
    try:
        logger.debug(
            f"Recalculating baseline subtraction for '{sample_id}' using temp params."
        )
        # Use the new spline baseline subtraction with TEMP params
        df_processed_with_baseline = subtract_spline_baseline(
            df_raw_sample,  # Use the raw dataframe we just fetched
            lower_endpoint=temp_params_data.get("lower"),
            upper_endpoint=temp_params_data.get("upper"),
            # Pass other params if needed
        )

        if (
            df_processed_with_baseline is not None
            and not df_processed_with_baseline.empty
        ):
            fig_processed_updated = create_processed_figure(
                df_processed=df_processed_with_baseline,
                baseline_params=temp_params_data,  # Use temp params for annotations
                title=f"Processed: {sample_id} (Adjusting)",
            )
            # Preserve layout from previous processed figure
            if (
                current_processed_figure_dict
                and "layout" in current_processed_figure_dict
            ):
                previous_layout = current_processed_figure_dict["layout"]
                if "xaxis.range" in previous_layout:
                    fig_processed_updated.update_layout(
                        xaxis_range=previous_layout["xaxis.range"]
                    )
                if "yaxis.range" in previous_layout:
                    fig_processed_updated.update_layout(
                        yaxis_range=previous_layout["yaxis.range"]
                    )
                logger.debug("Preserved layout range from previous processed figure.")
        else:
            logger.warning(
                f"Processed dataframe was empty for '{sample_id}' after temp subtraction (spline). "
            )
            fig_processed_updated.update_layout(title=f"Processed Error: {sample_id}")

    except Exception as e_proc:
        logger.error(
            f"Error generating updated processed plot (spline) for '{sample_id}': {e_proc}",
            exc_info=True,
        )
        fig_processed_updated.update_layout(title=f"Processing Error: {sample_id}")
    # --- Generate Updated Processed Plot --- End

    logger.info(f"Successfully updated both plots for {sample_id} from temp params.")
    return fig_raw_updated, fig_processed_updated


# --- NEW Callback to Update Tab Content ---
@debug_callback
@callback(
    Output("raw-plot-content", "style", allow_duplicate=True),
    Output("processed-plot-content", "style", allow_duplicate=True),
    Input("plot-tabs", "active_tab"),
    prevent_initial_call="initial_duplicate",
)
def handle_plot_tab_change(
    active_tab: str,
) -> Tuple[Dict, Dict]:
    """Shows/hides plot divs based on active tab.

    Does NOT regenerate any plots.

    Args:
        active_tab: The ID of the currently active plot tab ('tab-raw' or 'tab-processed').

    Returns:
        A tuple containing:
            - Style dictionary for the raw plot container.
            - Style dictionary for the processed plot container.
    """
    logger.info(f"Plot tab changed to: {active_tab}")
    style_visible = {"display": "block", "height": "45vh"}  # Match layout height
    style_hidden = {"display": "none", "height": "45vh"}

    if active_tab == "tab-raw":
        # Show raw, hide processed.
        return style_visible, style_hidden

    elif active_tab == "tab-processed":
        # Show processed, hide raw.
        return style_hidden, style_visible

    else:
        # Unknown tab ID, return current state (or default? no_update safer)
        logger.warning(f"Unknown plot tab ID: {active_tab}")
        return no_update, no_update
