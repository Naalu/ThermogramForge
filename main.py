"""
ThermogramForge application entry point.

This script initializes the Dash application, configures logging, registers the layout,
initializes callbacks, and starts the development server.
It ensures that components are imported and initialized in the correct order
to prevent circular dependencies or errors.
"""

import logging
import sys

# Configure logging (Consider moving to a separate config file/function for complex setups)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    # Import app instance first
    from app import app

    logger.info("App instance imported successfully")

    # Import layout registration function
    from app.layouts.main_layout import register_layout

    logger.info("Layout module imported successfully")

    # Register the layout with the app
    register_layout(app)
    logger.info("Layout registered with app using register_layout function")

    # Import callback initializer AFTER layout is set
    from app.callbacks import init_callbacks

    logger.info("Callback initializer imported")

    # Initialize all callbacks (this imports the individual callback modules)
    init_callbacks()
    logger.info("All callbacks initialized")

    # Simple server function
    def main() -> None:
        """Configures and runs the Dash development server."""
        print("Starting ThermogramForge application...")
        print("Visit http://127.0.0.1:8050/ in your browser")
        # Set debug=True for development features (hot-reloading, error debugging)
        # Set debug=False for production or performance testing
        app.run(debug=True)  # Keep True during active development

    # Run the server
    if __name__ == "__main__":
        main()

except ImportError as e:
    logger.error(f"ImportError during app initialization: {str(e)}", exc_info=True)
    logger.error(
        "Please ensure all dependencies are installed and Python path is correct."
    )
    sys.exit(1)
except Exception as e:
    logger.error(
        f"An unexpected error occurred during app initialization: {str(e)}",
        exc_info=True,
    )
    sys.exit(1)
