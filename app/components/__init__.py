"""
Initialize components for the ThermogramForge app.
"""

# Import individual component creation functions/layouts
# Removed import from deleted thermogram_plot.py
from .renderers import CheckboxRenderer  # Ensure renderers are available
from .upload_processed_modal import create_upload_processed_modal

__all__ = [
    # Removed thermogram_plot exports
    "create_upload_processed_modal",
    "CheckboxRenderer",
]
