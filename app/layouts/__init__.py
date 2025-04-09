"""
Layout definitions for ThermogramForge.

This package contains layout definitions for the application.
"""

# Import the main layout to automatically register it
from .main_layout import register_layout


# Function to initialize all layouts
def init_layouts(app):
    """
    Initialize all layouts with the app instance.

    Args:
        app: Dash app instance
    """
    register_layout(app)

    # Add future layout registrations here

    print("All layouts registered")
