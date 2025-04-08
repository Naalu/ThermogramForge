"""
Run the ThermogramForge web application.

This script starts the Dash web application in debug mode for development.
"""

# Import the app instance
import app.callbacks.baseline_callbacks

# Import callbacks to register them
import app.callbacks.upload_callbacks

# Import the main layout module to register the layout
import app.main
from app.app import app

if __name__ == "__main__":
    print("Starting ThermogramForge application...")
    print("Visit http://127.0.0.1:8050/ in your browser")
    app.run(debug=True)
