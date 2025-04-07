"""
Callbacks for baseline subtraction.
"""

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from app.components import create_thermogram_figure
from app.utils import simple_baseline_subtraction


@callback(
    Output("thermogram-plot", "figure", allow_duplicate=True),
    Output("baseline-info", "children"),
    Input("baseline-range", "value"),
    Input("apply-baseline", "n_clicks"),
    State("thermogram-data", "data"),
    State("thermogram-plot", "figure"),
    prevent_initial_call=True,
)
def update_baseline(baseline_range, n_clicks, data_json, current_figure):
    """Update the baseline subtraction."""
    # Determine which input triggered the callback
    ctx = callback.ctx
    triggered_id = ctx.triggered_id if ctx else None

    if triggered_id is None or data_json is None:
        raise PreventUpdate

    # Load data from JSON
    df = pd.read_json(data_json, orient="split")

    # If the range slider changed, just update the endpoints in the plot
    if triggered_id == "baseline-range":
        lower_temp, upper_temp = baseline_range

        # Update the plot to show new endpoints
        fig = create_thermogram_figure(
            df,
            title="Thermogram with Baseline Endpoints",
            endpoints=(lower_temp, upper_temp),
        )

        # Update baseline info
        info = html.Div(
            [
                html.P(
                    f"Selected baseline endpoints: {lower_temp:.1f}°C - {upper_temp:.1f}°C"
                ),
                html.P("Click 'Apply Baseline Subtraction' to process the data."),
            ]
        )

        return fig, info

    # If the apply button was clicked, perform baseline subtraction
    if triggered_id == "apply-baseline" and n_clicks:
        lower_temp, upper_temp = baseline_range

        # Perform baseline subtraction
        baseline_df = simple_baseline_subtraction(df, lower_temp, upper_temp)

        # Create figure with baseline
        fig = create_thermogram_figure(
            df,
            title="Thermogram with Baseline Subtraction",
            show_baseline=True,
            baseline_df=baseline_df,
            endpoints=(lower_temp, upper_temp),
        )

        # Calculate some stats about the subtraction
        original_max = df["dCp"].max()
        subtracted_max = baseline_df["dCp_subtracted"].max()

        # Update baseline info
        info = html.Div(
            [
                html.P(
                    f"Baseline subtracted with endpoints: {lower_temp:.1f}°C - {upper_temp:.1f}°C"
                ),
                html.P(
                    f"Original max dCp: {original_max:.4f}, After subtraction: {subtracted_max:.4f}"
                ),
                dbc.Button(
                    "Download Processed Data",
                    id="download-button",
                    color="success",
                    className="mt-2",
                ),
                dbc.Tooltip(
                    "Download the baseline-subtracted data", target="download-button"
                ),
            ]
        )

        return fig, info

    # Default case, should not happen
    return current_figure, html.Div("Select baseline endpoints to continue")


@callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("thermogram-data", "data"),
    State("baseline-range", "value"),
    prevent_initial_call=True,
)
def download_processed_data(n_clicks, data_json, baseline_range):
    """Generate download data when the download button is clicked."""
    if not n_clicks or data_json is None:
        raise PreventUpdate

    # Load data
    df = pd.read_json(data_json, orient="split")

    # Apply baseline subtraction
    lower_temp, upper_temp = baseline_range
    processed_df = simple_baseline_subtraction(df, lower_temp, upper_temp)

    # Prepare for download
    return dict(
        content=processed_df.to_csv(index=False),
        filename="baseline_subtracted_thermogram.csv",
    )
