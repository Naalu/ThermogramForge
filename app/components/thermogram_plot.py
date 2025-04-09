"""
Enhanced thermogram visualization component.
"""

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import dcc, html


def create_thermogram_figure(
    df: pd.DataFrame,
    title: str = "Thermogram Analysis",
    show_baseline: bool = False,
    baseline_df: pd.DataFrame = None,
    endpoints: tuple = None,
    highlight_peaks: bool = False,
    peaks: dict = None,
):
    """
    Create an enhanced thermogram visualization.

    Args:
        df: DataFrame with Temperature and dCp columns
        title: Plot title
        show_baseline: Whether to show baseline subtraction
        baseline_df: DataFrame with baseline-subtracted data
        endpoints: Tuple of (lower_temp, upper_temp) for baseline endpoints
        highlight_peaks: Whether to highlight detected peaks
        peaks: Dictionary of peak information {name: {peak_temp, peak_height}}

    Returns:
        Plotly figure object
    """
    if show_baseline and baseline_df is not None:
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

        # Add baseline to top plot if available
        if "dCp_baseline" in baseline_df.columns:
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
        if "dCp_subtracted" in baseline_df.columns:
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

            # Add peak markers to the baseline-subtracted data if available
            if highlight_peaks and peaks:
                for peak_name, peak_info in peaks.items():
                    if (
                        peak_name != "FWHM"
                        and "peak_temp" in peak_info
                        and "peak_height" in peak_info
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=[peak_info["peak_temp"]],
                                y=[peak_info["peak_height"]],
                                mode="markers",
                                name=f"{peak_name} ({peak_info['peak_temp']:.1f}°C)",
                                marker=dict(
                                    size=10,
                                    color="red",
                                    symbol="star",
                                    line=dict(width=1, color="black"),
                                ),
                            ),
                            row=2,
                            col=1,
                        )

        # Add endpoints if provided
        if endpoints:
            lower_temp, upper_temp = endpoints

            # Add vertical lines at endpoints
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
            fig.add_hline(
                y=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1
            )
    else:
        # Create a single plot for raw data only
        fig = go.Figure()

        # Add raw data
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

        # Add peaks if available
        if highlight_peaks and peaks:
            for peak_name, peak_info in peaks.items():
                if (
                    peak_name != "FWHM"
                    and "peak_temp" in peak_info
                    and "peak_height" in peak_info
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=[peak_info["peak_temp"]],
                            y=[peak_info["peak_height"]],
                            mode="markers",
                            name=f"{peak_name}",
                            marker=dict(
                                size=10,
                                color="red",
                                symbol="star",
                                line=dict(width=1, color="black"),
                            ),
                        )
                    )

        # Add vertical lines for endpoints if provided
        if endpoints:
            lower_temp, upper_temp = endpoints
            fig.add_vline(
                x=lower_temp,
                line=dict(color="red", width=1.5, dash="dash"),
                annotation_text="Lower Endpoint",
                annotation_position="top right",
            )
            fig.add_vline(
                x=upper_temp,
                line=dict(color="red", width=1.5, dash="dash"),
                annotation_text="Upper Endpoint",
                annotation_position="top left",
            )

    # Update layout for better appearance
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )

    # Update axis labels
    fig.update_xaxes(
        title_text="Temperature (°C)",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="gray",
        tickformat=".1f",
    )

    # Update y-axis for top plot (or single plot)
    y_title = "Excess Heat Capacity (dCp)"
    if show_baseline and baseline_df is not None:
        fig.update_yaxes(
            title_text=y_title,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
            row=1,
            col=1,
        )

        # Update y-axis for bottom plot
        fig.update_yaxes(
            title_text="Baseline Subtracted dCp",
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
            row=2,
            col=1,
        )
    else:
        fig.update_yaxes(
            title_text=y_title,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        )

    return fig


def create_comparison_figure(
    samples: dict,
    mode: str = "raw",
    title: str = "Thermogram Comparison",
    highlight_shared_features: bool = False,
):
    """
    Create a comparison figure of multiple thermogram samples.

    Args:
        samples: Dictionary of sample_id -> DataFrame with Temperature and dCp columns
        mode: Comparison mode ('raw', 'baseline', 'normalized')
        title: Plot title
        highlight_shared_features: Whether to highlight features common across samples

    Returns:
        Plotly figure object
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
    color_scale = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "teal",
        "magenta",
        "brown",
        "pink",
        "gray",
    ]

    # Add each sample as a trace
    for i, (sample_id, df) in enumerate(samples.items()):
        if df is None or len(df) == 0:
            continue

        color = color_scale[i % len(color_scale)]

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


def create_data_preview(df: pd.DataFrame, max_rows: int = 10):
    """
    Create a data preview component.

    Args:
        df: DataFrame to preview
        max_rows: Maximum number of rows to show

    Returns:
        Dash component with data preview
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


def thermogram_card(id="thermogram-plot", figure=None):
    """
    Create a card containing the thermogram plot.

    Args:
        id: ID for the graph component
        figure: Initial figure to display

    Returns:
        Card component with the thermogram plot
    """
    return dbc.Card(
        [
            dbc.CardHeader("Thermogram Visualization"),
            dbc.CardBody(
                [
                    dcc.Graph(
                        id=id,
                        figure=figure or {},
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "thermogram_plot",
                                "height": 800,
                                "width": 1200,
                                "scale": 2,
                            },
                        },
                        style={"height": "550px"},
                    )
                ]
            ),
        ]
    )


def data_preview_card(id="data-preview", children=None):
    """
    Create a card for the data preview.

    Args:
        id: ID for the component
        children: Initial children to display

    Returns:
        Card component with the data preview
    """
    return dbc.Card(
        [
            dbc.CardHeader("Data Preview"),
            dbc.CardBody(id=id, children=children or html.Div("No data loaded yet.")),
        ]
    )


def metrics_table(metrics: dict, multiple_samples: bool = False):
    """
    Create a metrics table component.

    Args:
        metrics: Dictionary of metrics or DataFrame of metrics for multiple samples
        multiple_samples: Whether the metrics are for multiple samples

    Returns:
        Dash component with metrics table
    """
    if not metrics:
        return html.Div("No metrics available")

    if multiple_samples:
        # Create a table with samples as rows and metrics as columns
        return html.Div(
            [
                html.H6("Metrics Comparison"),
                dbc.Table.from_dataframe(
                    pd.DataFrame(metrics),
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )
    else:
        # Create a table with metrics as rows
        metric_df = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
        )

        return html.Div(
            [
                html.H6("Calculated Metrics"),
                dbc.Table.from_dataframe(
                    metric_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )
