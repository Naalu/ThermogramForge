"""
Initialize the app package.

This file ensures proper initialization order of app components.
"""

import os
from typing import Final

import dash
import dash_bootstrap_components as dbc
import dash_uploader as du

# Create the Dash app instance first
app: dash.Dash = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MATERIA, dbzc.icons.FONT_AWESOME],
    title="ThermogramForge v1.0",
    suppress_callback_exceptions=True,
)

# Define server for deployment
server = app.server

# Configure upload folder
UPLOAD_FOLDER_ROOT: Final[str] = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(UPLOAD_FOLDER_ROOT):
    os.makedirs(UPLOAD_FOLDER_ROOT, exist_ok=True)

# Configure dash-uploader
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

# Optional: Add server health endpoint if needed for deployment checks
# @server.route('/health')
# def health():
#     return 'OK'
