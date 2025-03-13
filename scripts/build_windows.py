"""
Script to build a standalone Windows executable using PyInstaller.
"""

import os
import shutil
import subprocess
import sys


def build_windows_exe() -> int:
    """Build Windows executable."""
    print("Building Windows executable...")

    # Define paths
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    dist_dir = os.path.join(project_root, "dist")
    build_dir = os.path.join(project_root, "build")
    # spec_file = os.path.join(project_root, "thermogram_app.spec")

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
        "--icon=thermogram_app/assets/icon.ico",  # Add when we have an icon
        "thermogram_app/app.py",  # Main entry point
    ]

    # Run PyInstaller
    subprocess.run(pyinstaller_cmd, check=True)

    print(f"Build completed! Executable created at {dist_dir}/ThermogramAnalysis.exe")
    return 0


if __name__ == "__main__":
    sys.exit(build_windows_exe())
