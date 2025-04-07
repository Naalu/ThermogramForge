"""
Dash app instance defined separately to avoid circular imports.
"""

import os

import dash
import dash_bootstrap_components as dbc

# Create the Dash app instance
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ThermogramForge v1.0",
    suppress_callback_exceptions=True,
)

# server variable for deployment
server = app.server

# Configure upload folder for dash-uploader
UPLOAD_FOLDER_ROOT = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(UPLOAD_FOLDER_ROOT):
    os.makedirs(UPLOAD_FOLDER_ROOT, exist_ok=True)
