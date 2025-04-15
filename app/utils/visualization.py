"""
Visualization utilities for thermogram analysis.
"""

import logging
from typing import Dict, Optional, Tuple

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import html

logger = logging.getLogger(__name__)


def create_thermogram_figure(
    df: Optional[pd.DataFrame], title: str = "Thermogram Analysis"
) -> go.Figure:
    """Creates a simple thermogram visualization.

    Args:
        df (Optional[pd.DataFrame]): DataFrame containing thermogram data with
            'Temperature' and 'dCp' columns. If None or empty, an empty figure
            with a message is returned.

        title (str): The title for the plot. Defaults to "Thermogram Analysis".

    Returns:
        go.Figure: A Plotly Figure object representing the thermogram.
    """
    fig = go.Figure()

    if df is None or df.empty:
        logger.warning("Input DataFrame is None or empty for create_thermogram_figure.")
        fig.update_layout(
            title=title,
            xaxis_title="Temperature (°C)",
            yaxis_title="Excess Heat Capacity (dCp)",
            template="plotly_white",
            height=550,
            annotations=[
                dict(
                    text="No data available to display.",
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
        )
        return fig

    if "Temperature" not in df.columns or "dCp" not in df.columns:
        logger.error("DataFrame missing required columns 'Temperature' or 'dCp'.")
        fig.update_layout(
            title=title,
            xaxis_title="Temperature (°C)",
            yaxis_title="Excess Heat Capacity (dCp)",
            template="plotly_white",
            height=550,
            annotations=[
                dict(
                    text="Error: Input data missing required columns.",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
            ],
        )
        return fig

    # Add trace for the data
    try:
        fig.add_trace(
            go.Scatter(
                x=df["Temperature"],
                y=df["dCp"],
                mode="lines+markers",
                name="Thermogram Data",
                marker=dict(size=4, opacity=0.7),
                line=dict(width=1.5),
            )
        )
    except Exception as e:
        logger.error(f"Error adding trace to thermogram figure: {e}")
        fig.update_layout(
            annotations=[
                dict(
                    text=f"Error creating plot: {e}",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
            ]
        )
        # Keep layout settings from above if possible
        fig.update_layout(
            title=title,
            xaxis_title="Temperature (°C)",
            yaxis_title="Excess Heat Capacity (dCp)",
            template="plotly_white",
            height=550,
        )
        return fig

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Temperature (°C)",
        yaxis_title="Excess Heat Capacity (dCp)",
        template="plotly_white",
        height=550,
    )

    return fig


def create_baseline_figure(
    df: Optional[pd.DataFrame],
    baseline_df: Optional[pd.DataFrame],
    title: str,
    endpoints: Optional[Tuple[float, float]],
) -> go.Figure:
    """Creates a figure showing baseline subtraction with two subplots.

    Args:
        df (Optional[pd.DataFrame]): Original DataFrame with 'Temperature' and 'dCp' columns.
        baseline_df (Optional[pd.DataFrame]): DataFrame with baseline-subtracted data,
            expecting 'Temperature', 'dCp_baseline', and 'dCp_subtracted' columns.
        title (str): The title for the plot.
        endpoints (Optional[Tuple[float, float]]): Tuple of (lower_temp, upper_temp)
            for baseline endpoints. If None, vertical lines are not drawn.

    Returns:
        go.Figure: A Plotly Figure object with two subplots: raw data + baseline,
            and baseline-subtracted data. Returns an empty figure with a message
            on error or if input data is invalid.

    """
    fig = sp.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Raw Data with Baseline", "Baseline Subtracted"),
        row_heights=[0.6, 0.4],
    )

    # Basic validation
    error_message = None
    if df is None or df.empty:
        error_message = "Original data DataFrame is missing or empty."
    elif "Temperature" not in df.columns or "dCp" not in df.columns:
        error_message = "Original data DataFrame missing 'Temperature' or 'dCp'."
    elif baseline_df is None or baseline_df.empty:
        error_message = "Baseline data DataFrame is missing or empty."
    elif not all(
        col in baseline_df.columns
        for col in ["Temperature", "dCp_baseline", "dCp_subtracted"]
    ):
        error_message = "Baseline DataFrame missing required columns."
    elif endpoints is not None and (
        not isinstance(endpoints, tuple) or len(endpoints) != 2
    ):
        error_message = "Endpoints must be a tuple of two numbers."
        endpoints = None  # Prevent further errors using endpoints

    if error_message:
        logger.error(f"Error creating baseline figure: {error_message}")
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=550,
            annotations=[
                dict(
                    text=f"Error: {error_message}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
            ],
        )
        # Try to set axis labels even on error
        fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Excess Heat Capacity (dCp)", row=1, col=1)
        fig.update_yaxes(title_text="Baseline Subtracted dCp", row=2, col=1)
        return fig

    try:
        # Add raw data to top plot
        fig.add_trace(
            go.Scatter(
                x=df["Temperature"],
                y=df["dCp"],
                mode="lines+markers",
                name="Raw Data",
                marker=dict(size=4, opacity=0.7),
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )

        # Add baseline to top plot
        fig.add_trace(
            go.Scatter(
                x=baseline_df["Temperature"],
                y=baseline_df["dCp_baseline"],
                mode="lines",
                name="Baseline",
                line=dict(color="red", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Add baseline-subtracted data to bottom plot
        fig.add_trace(
            go.Scatter(
                x=baseline_df["Temperature"],
                y=baseline_df["dCp_subtracted"],
                mode="lines+markers",
                name="Baseline Subtracted",
                marker=dict(size=4, opacity=0.7, color="green"),
                line=dict(width=1, color="green"),
            ),
            row=2,
            col=1,
        )

        # Add vertical lines at endpoints if provided
        if endpoints:
            lower_temp, upper_temp = endpoints
            for temp, text, pos in [
                (lower_temp, "Lower Endpoint", "top right"),
                (upper_temp, "Upper Endpoint", "top left"),
            ]:
                fig.add_vline(
                    x=temp,
                    line=dict(color="red", width=1.5),
                    annotation_text=text,
                    annotation_position=pos,
                    row=1,
                    col=1,
                )

        # Add zero line to baseline-subtracted plot
        fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)

    except Exception as e:
        logger.error(f"Error adding traces to baseline figure: {e}")
        fig.update_layout(
            annotations=[
                dict(
                    text=f"Error generating plot: {e}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
            ]
        )
        # Keep layout settings from above if possible
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=550,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.update_xaxes(
            title_text="Temperature (°C)", showgrid=True, row=2, col=1
        )  # Show axis title on bottom plot
        fig.update_yaxes(
            title_text="Excess Heat Capacity (dCp)", showgrid=True, row=1, col=1
        )
        fig.update_yaxes(
            title_text="Baseline Subtracted dCp", showgrid=True, row=2, col=1
        )
        return fig

    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",  # Improve hover interaction
    )

    # Update axis labels (explicitly target bottom x-axis)
    fig.update_xaxes(
        title_text="Temperature (°C)",
        showgrid=True,
        row=2,
        col=1,  # Ensure title is on the bottom axis
    )

    # Update y-axis labels
    fig.update_yaxes(
        title_text="Excess Heat Capacity (dCp)",
        showgrid=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Baseline Subtracted dCp",
        showgrid=True,
        row=2,
        col=1,
    )

    return fig


def create_data_preview(df: Optional[pd.DataFrame], max_rows: int = 10) -> html.Div:
    """Creates a data preview component with summary and table.

    Args:
        df: DataFrame to preview. Expects 'Temperature' and 'dCp' columns
            for the summary section. If None or empty, shows a message.

        max_rows: Maximum number of rows to show in the preview table.
                  Defaults to 10.

    Returns:
        A Dash html.Div component containing the data summary and table preview.
    """
    if df is None or df.empty:
        return html.Div(
            [
                html.H6("Data Preview", className="mt-3"),
                dbc.Alert("No data loaded.", color="warning"),
            ]
        )

    # Create a summary component
    summary_content = []
    shape_info = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"
    summary_content.append(shape_info)

    try:
        if (
            "Temperature" in df.columns
            and pd.api.types.is_numeric_dtype(df["Temperature"])
            and not df["Temperature"].isnull().all()
        ):
            temp_range = f"Temperature range: {df['Temperature'].min():.1f}°C to {df['Temperature'].max():.1f}°C"
            summary_content.extend([html.Br(), temp_range])
        if (
            "dCp" in df.columns
            and pd.api.types.is_numeric_dtype(df["dCp"])
            and not df["dCp"].isnull().all()
        ):
            dcp_range = f"dCp range: {df['dCp'].min():.4f} to {df['dCp'].max():.4f}"
            summary_content.extend([html.Br(), dcp_range])
    except Exception as e:
        logger.warning(f"Could not generate full data summary: {e}")
        summary_content.append(html.Br())
        summary_content.append(html.Em(f"(Could not generate range summary: {e})"))

    summary = html.Div(
        [html.H6("Data Summary", className="mt-3"), html.P(summary_content)]
    )

    # Create a table preview with a smaller subset of rows
    preview_rows = min(max_rows, len(df))
    try:
        table_component = dbc.Table.from_dataframe(
            df.head(preview_rows).round(4),  # Round data for display
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm",
        )
    except Exception as e:
        logger.error(f"Error creating preview table: {e}")
        table_component = dbc.Alert(
            f"Error creating table preview: {e}", color="danger"
        )

    table = html.Div(
        [
            html.H6(f"Data Preview (first {preview_rows} rows)"),
            table_component,
        ]
    )

    return html.Div([summary, table])


def create_comparison_figure(
    samples: Optional[Dict[str, Optional[pd.DataFrame]]],
    mode: str = "raw",
    title: str = "Thermogram Comparison",
) -> go.Figure:
    """Creates a comparison figure of multiple thermogram samples.

    Args:
        samples (Optional[Dict[str, Optional[pd.DataFrame]]]): Dictionary where keys
            are sample_ids (str) and values are DataFrames. Each DataFrame should
            contain 'Temperature' and 'dCp'. If mode is 'baseline', it should also
            contain 'dCp_subtracted'. If None or empty, returns an empty figure
            with a message.

        mode (str): Comparison mode. Options: 'raw', 'baseline', 'normalized'.
            Defaults to 'raw'.

        title (str): The title for the plot. Defaults to "Thermogram Comparison".

    Returns:
        go.Figure: A Plotly Figure object comparing the samples. Returns an empty
            figure with a message if input is invalid or no valid samples are found.
    """
    fig = go.Figure()

    if not samples:
        logger.warning("No samples provided for comparison figure.")
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=600,
            annotations=[
                dict(
                    text="No samples selected for comparison.",
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
        )
        return fig

    valid_modes = ["raw", "baseline", "normalized"]
    if mode not in valid_modes:
        logger.warning(f"Invalid mode '{mode}' provided. Defaulting to 'raw'.")
        mode = "raw"

    # Create color scale for samples (can be extended or made dynamic)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]  # Plotly default colors

    valid_sample_count = 0
    for i, (sample_id, df) in enumerate(samples.items()):
        if df is None or df.empty:
            logger.warning(f"Sample '{sample_id}' has no data. Skipping.")
            continue
        if "Temperature" not in df.columns or "dCp" not in df.columns:
            logger.warning(
                f"Sample '{sample_id}' missing 'Temperature' or 'dCp'. Skipping."
            )
            continue

        color = colors[i % len(colors)]
        y_col = "dCp"
        y_data = None
        current_df = df  # Avoid modifying original dict df

        try:
            if mode == "baseline":
                if "dCp_subtracted" in current_df.columns:
                    y_col = "dCp_subtracted"
                    y_data = current_df[y_col]
                else:
                    logger.warning(
                        f"Mode 'baseline' selected but 'dCp_subtracted' missing for sample '{sample_id}'. Using raw 'dCp'."
                    )
                    y_data = current_df["dCp"]  # Fallback to raw
            elif mode == "normalized":
                # Normalize by dividing by max absolute value to handle negative peaks
                max_abs_val = current_df["dCp"].abs().max()
                if max_abs_val > 1e-9:  # Avoid division by zero or near-zero
                    # Create a copy only if normalization is performed
                    current_df = current_df.copy()
                    normalized_y = current_df["dCp"] / max_abs_val
                    current_df["dCp_normalized"] = normalized_y
                    y_col = "dCp_normalized"
                    y_data = current_df[y_col]
                else:
                    logger.warning(
                        f"Max absolute dCp is near zero for sample '{sample_id}'. Cannot normalize. Using raw 'dCp'."
                    )
                    y_data = current_df["dCp"]  # Fallback to raw
            else:  # mode == "raw"
                y_data = current_df[y_col]

            if y_data is not None:
                # Add trace for this sample
                fig.add_trace(
                    go.Scatter(
                        x=current_df["Temperature"],
                        y=y_data,
                        mode="lines",
                        name=str(sample_id),  # Ensure name is string
                        line=dict(color=color, width=1.5),  # Slightly thinner lines
                        legendgroup=str(sample_id),  # Group hover info
                    )
                )
                valid_sample_count += 1
        except Exception as e:
            logger.error(f"Error processing sample '{sample_id}' for comparison: {e}")
            continue  # Skip this sample

    # Update y-axis title based on mode
    y_title = "Excess Heat Capacity (dCp)"
    if mode == "baseline":
        y_title = "Baseline Subtracted dCp"
    elif mode == "normalized":
        y_title = "Normalized dCp (by max abs value)"

    # Final layout updates
    final_title = title
    if valid_sample_count == 0:
        final_title = f"{title} (No valid data)"
        fig.add_annotation(text="No valid sample data to display.", showarrow=False)

    fig.update_layout(
        title=final_title,
        xaxis_title="Temperature (°C)",
        yaxis_title=y_title,
        template="plotly_white",
        height=600,
        legend=dict(
            traceorder="normal",  # Keep legend order same as trace addition
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        hovermode="closest",
    )

    return fig
