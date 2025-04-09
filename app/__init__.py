"""
Initialize ThermogramForge application.
"""

import logging
import os

import dash
import dash_bootstrap_components as dbc
import dash_uploader as du

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing ThermogramForge application")

# Create app instance with proper configuration
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Configure upload folder
UPLOAD_FOLDER_ROOT = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER_ROOT, exist_ok=True)
logger.info(f"Upload folder configured at: {UPLOAD_FOLDER_ROOT}")

# Configure dash-uploader
try:
    du.configure_upload(app, UPLOAD_FOLDER_ROOT)
    logger.info("Dash uploader configured successfully")
except Exception as e:
    logger.error(f"Error configuring dash-uploader: {str(e)}", exc_info=True)
