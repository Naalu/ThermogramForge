"""
Callbacks for ThermogramForge.

This package contains callback definitions for the application.
"""

# Import all callbacks to register them
from . import (
    baseline_callbacks,
    control_panel_callbacks,
    processed_upload_callbacks,
    report_builder_callbacks,
    upload_callbacks,
    visualization_callbacks,
)


# Function to initialize all callbacks
def init_callbacks():
    """
    Initialize all callbacks.
    """
    # All callbacks are automatically registered when imported
    print("All callbacks registered")
