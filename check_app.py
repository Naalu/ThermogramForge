"""
Utility script to check application for common problems
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    # Import app instance
    logger.info("App instance imported successfully")

    # Import layout creation function
    from app.layouts.main_layout import create_layout

    logger.info("Layout module imported successfully")

    # Import layout checker
    from app.utils.layout_checker import find_duplicate_ids

    # Check the layout for duplicate IDs
    layout = create_layout()
    logger.info("Checking layout for duplicate IDs...")
    duplicates = find_duplicate_ids(layout)

    if duplicates:
        logger.error(f"DUPLICATE IDs FOUND: {duplicates}")
        logger.error(
            "You must fix these duplicate IDs before the app will work correctly"
        )
        sys.exit(1)
    else:
        logger.info("No duplicate IDs found in layout - good!")

    logger.info("Application check complete - no issues found")

except Exception as e:
    logger.error(f"Error during app check: {str(e)}", exc_info=True)
    sys.exit(1)
