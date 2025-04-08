"""
Callbacks for baseline subtraction.
"""

from io import StringIO

import dash
import pandas as pd
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from app.components import create_thermogram_figure


def simple_baseline_subtraction(df, lower_temp, upper_temp):
    """
    Perform a simple linear baseline subtraction.

    Args:
        df: DataFrame with Temperature and dCp columns
        lower_temp: Lower temperature endpoint
        upper_temp: Upper temperature endpoint

    Returns:
        DataFrame with baseline-subtracted data
    """
    try:
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Get values at endpoints
        lower_idx = (df["Temperature"] - lower_temp).abs().idxmin()
        upper_idx = (df["Temperature"] - upper_temp).abs().idxmin()

        lower_temp_actual = df.loc[lower_idx, "Temperature"]
        upper_temp_actual = df.loc[upper_idx, "Temperature"]

        lower_dcp = df.loc[lower_idx, "dCp"]
        upper_dcp = df.loc[upper_idx, "dCp"]

        # Calculate slope and intercept for linear baseline
        slope = (upper_dcp - lower_dcp) / (upper_temp_actual - lower_temp_actual)
        intercept = lower_dcp - slope * lower_temp_actual

        # Calculate baseline for each point
        baseline = slope * df["Temperature"] + intercept

        # Subtract baseline
        result["dCp_baseline"] = baseline
        result["dCp_subtracted"] = result["dCp"] - baseline

        return result
    except Exception as e:
        print(f"Error in baseline subtraction: {str(e)}")
        return None


@callback(
    [
        Output("thermogram-plot", "figure", allow_duplicate=True),
        Output("baseline-info", "children"),
    ],
    [
        Input("apply-baseline", "n_clicks"),
        State("thermogram-data", "data"),
        State("baseline-range", "value"),
    ],
    prevent_initial_call=True,
)
def apply_baseline_subtraction(n_clicks, data_json, baseline_range):
    """Apply baseline subtraction when button is clicked."""
    if not n_clicks or not data_json:
        raise PreventUpdate

    try:
        # Load data from JSON
        df = pd.read_json(StringIO(data_json), orient="split")

        # Get baseline endpoints
        lower_temp, upper_temp = baseline_range

        # Perform baseline subtraction
        baseline_df = simple_baseline_subtraction(df, lower_temp, upper_temp)

        if baseline_df is None:
            return dash.no_update, html.Div(
                "Error in baseline subtraction", style={"color": "red"}
            )

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

        # Create info display
        info = html.Div(
            [
                html.P(
                    f"Baseline subtracted with endpoints: {lower_temp:.1f}°C - {upper_temp:.1f}°C"
                ),
                html.P(
                    f"Original max dCp: {original_max:.4f}, After subtraction: {subtracted_max:.4f}"
                ),
                html.Button(
                    "Download Processed Data",
                    id="download-button",
                    className="btn btn-success mt-2",
                ),
            ]
        )

        return fig, info

    except Exception as e:
        print(f"Error applying baseline subtraction: {str(e)}")
        return dash.no_update, html.Div(
            [
                "Error applying baseline subtraction:",
                html.Pre(str(e)),
            ],
            style={"color": "red"},
        )


@callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    [State("thermogram-data", "data"), State("baseline-range", "value")],
    prevent_initial_call=True,
)
def download_processed_data(n_clicks, data_json, baseline_range):
    """Generate download data when the download button is clicked."""
    if not n_clicks or not data_json:
        raise PreventUpdate

    try:
        # Load data
        df = pd.read_json(StringIO(data_json), orient="split")

        # Apply baseline subtraction
        lower_temp, upper_temp = baseline_range
        processed_df = simple_baseline_subtraction(df, lower_temp, upper_temp)

        if processed_df is None:
            return None

        # Select columns for export
        export_df = processed_df[
            ["Temperature", "dCp", "dCp_baseline", "dCp_subtracted"]
        ]

        # Prepare for download
        return dict(
            content=export_df.to_csv(index=False),
            filename="baseline_subtracted_thermogram.csv",
        )

    except Exception as e:
        print(f"Error generating download: {str(e)}")
        return None
