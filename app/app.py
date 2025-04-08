"""
Dash app instance defined separately to avoid circular imports.
"""

import dash
import dash_bootstrap_components as dbc

# Create the Dash app instance
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ThermogramForge v1.0",
    suppress_callback_exceptions=True,
    # Enable better error messages during development
    assets_folder="assets",
    include_assets_files=True,
)

# Create a server variable for deployment
server = app.server
