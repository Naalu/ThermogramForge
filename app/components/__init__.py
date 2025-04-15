"""
Initialize components for the ThermogramForge app.
"""

from .renderers import CheckboxRenderer  # Ensure renderers are available
from .upload_processed_modal import create_upload_processed_modal

__all__ = [
    # Removed thermogram_plot exports
    "create_upload_processed_modal",
    "CheckboxRenderer",
]
