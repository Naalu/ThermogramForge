"""
Convenience script to run the Thermogram Analysis application.
"""

from thermogram_app.app import app

if __name__ == "__main__":
    print("Starting ThermogramForge application...")
    print("Visit http://127.0.0.1:8050/ in your browser")
    app.run_server(debug=True)
