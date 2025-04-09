"""
Visualization utilities for thermogram analysis.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import html


def create_thermogram_figure(df, title="Thermogram Analysis"):
    """
    Create a simple thermogram visualization.

    Args:
        df: DataFrame with Temperature and dCp columns
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add trace for the data
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

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Temperature (°C)",
        yaxis_title="Excess Heat Capacity (dCp)",
        template="plotly_white",
        height=550,
    )

    return fig


def create_baseline_figure(df, baseline_df, title, endpoints):
    """
    Create a figure showing baseline subtraction with two subplots.

    Args:
        df: Original DataFrame with Temperature and dCp columns
        baseline_df: DataFrame with baseline-subtracted data
        title: Plot title
        endpoints: Tuple of (lower_temp, upper_temp) for baseline endpoints

    Returns:
        Plotly Figure object
    """
    # Create a subplot with two rows
    fig = sp.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Raw Data with Baseline", "Baseline Subtracted"),
        row_heights=[0.6, 0.4],
    )

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

    # Add vertical lines at endpoints
    lower_temp, upper_temp = endpoints
    fig.add_vline(
        x=lower_temp,
        line=dict(color="red", width=1.5),
        annotation_text="Lower Endpoint",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=upper_temp,
        line=dict(color="red", width=1.5),
        annotation_text="Upper Endpoint",
        annotation_position="top left",
        row=1,
        col=1,
    )

    # Add zero line to baseline-subtracted plot
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axis labels
    fig.update_xaxes(
        title_text="Temperature (°C)",
        showgrid=True,
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


def create_data_preview(df, max_rows=10):
    """
    Create a data preview component with summary and table.

    Args:
        df: DataFrame to preview
        max_rows: Maximum number of rows to show in preview table

    Returns:
        Dash HTML component
    """
    # Create a summary component
    summary = html.Div(
        [
            html.H6("Data Summary", className="mt-3"),
            html.P(
                [
                    f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                    html.Br(),
                    f"Temperature range: {df['Temperature'].min():.1f}°C to {df['Temperature'].max():.1f}°C",
                    html.Br(),
                    f"dCp range: {df['dCp'].min():.4f} to {df['dCp'].max():.4f}",
                ]
            ),
        ]
    )

    # Create a table preview with a smaller subset of rows
    preview_rows = min(max_rows, len(df))
    table = html.Div(
        [
            html.H6(f"Data Preview (first {preview_rows} rows)"),
            dbc.Table.from_dataframe(
                df.head(preview_rows),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm",
            ),
        ]
    )

    return html.Div([summary, table])


def create_comparison_figure(samples, mode="raw", title="Thermogram Comparison"):
    """
    Create a comparison figure of multiple thermogram samples.

    Args:
        samples: Dictionary of sample_id -> DataFrame
        mode: Comparison mode ('raw', 'baseline', 'normalized')
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if not samples:
        fig.update_layout(
            title="No samples to compare",
            annotations=[
                dict(
                    text="No sample data available", showarrow=False, font=dict(size=14)
                )
            ],
        )
        return fig

    # Create color scale for samples
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "teal",
        "magenta",
        "cyan",
        "pink",
        "gray",
    ]

    # Add each sample as a trace
    for i, (sample_id, df) in enumerate(samples.items()):
        if df is None or len(df) == 0:
            continue

        color = colors[i % len(colors)]

        # Get y column based on mode
        y_col = "dCp"
        if mode == "baseline" and "dCp_subtracted" in df.columns:
            y_col = "dCp_subtracted"
        elif mode == "normalized":
            # Normalize by dividing by max value
            max_val = df["dCp"].max()
            if max_val > 0:
                normalized_y = df["dCp"] / max_val
                df = df.copy()
                df["dCp_normalized"] = normalized_y
                y_col = "dCp_normalized"

        # Add trace for this sample
        fig.add_trace(
            go.Scatter(
                x=df["Temperature"],
                y=df[y_col],
                mode="lines",
                name=sample_id,
                line=dict(color=color, width=2),
            )
        )

    # Update y-axis title based on mode
    y_title = "Excess Heat Capacity (dCp)"
    if mode == "baseline":
        y_title = "Baseline Subtracted dCp"
    elif mode == "normalized":
        y_title = "Normalized dCp"

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Temperature (°C)",
        yaxis_title=y_title,
        template="plotly_white",
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="closest",
    )

    return fig
