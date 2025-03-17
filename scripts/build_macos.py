"""
Script to build a standalone macOS application using PyInstaller.
"""

import os
import shutil
import subprocess
import sys


def build_macos_app() -> int:
    """Build macOS application."""
    print("Building macOS application...")

    # Define paths
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    dist_dir = os.path.join(project_root, "dist")
    build_dir = os.path.join(project_root, "build")

    # Clean up previous builds
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    # Create PyInstaller command
    pyinstaller_cmd = [
        "pyinstaller",
        "--clean",
        "--name=ThermogramAnalysis",
        "--windowed",  # Use a windowed application
        "--onefile",  # Create a single executable
        "--icon=thermogram_app/assets/icon.icns",  # Add when we have an icon
        "thermogram_app/app.py",  # Main entry point
    ]

    # Run PyInstaller
    subprocess.run(pyinstaller_cmd, check=True)

    print(f"Build completed! Application created at {dist_dir}/ThermogramAnalysis.app")
    return 0


if __name__ == "__main__":
    sys.exit(build_macos_app())
