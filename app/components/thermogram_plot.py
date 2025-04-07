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
):
    """
    Create an enhanced thermogram visualization.

    Args:
        df: DataFrame with Temperature and dCp columns
        title: Plot title
        show_baseline: Whether to show baseline subtraction
        baseline_df: DataFrame with baseline-subtracted data
        endpoints: Tuple of (lower_temp, upper_temp) for baseline endpoints

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
                marker=dict(size=6, opacity=0.7),
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
                    marker=dict(size=6, opacity=0.7, color="green"),
                    line=dict(width=1, color="green"),
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
                marker=dict(size=6, opacity=0.7),
                line=dict(width=1.5),
            )
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
    fig.update_yaxes(
        title_text="Excess Heat Capacity (dCp)",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="gray",
        row=1,
        col=1,
    )

    # Update y-axis for bottom plot if using subplots
    if show_baseline and baseline_df is not None:
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
