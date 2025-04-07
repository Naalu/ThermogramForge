#!/bin/bash
# Quick development environment setup

# Create virtual environment if it doesn't exist
if [ ! -d "venv-v1" ]; then
    python -m venv venv-v1
fi

# Activate virtual environment
source venv-v1/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Set up pre-commit
pre-commit install

echo "Development environment ready! Run 'python run_app.py' to start the application."