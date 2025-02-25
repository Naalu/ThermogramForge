# ThermogramForge

A Python toolkit for thermal biopsy data analysis, including baseline subtraction, parameter calculation, and interactive visualization.

## Overview

ThermogramForge is a modern Python implementation of thermogram analysis tools, designed to replace legacy R-based tools with improved performance, usability, and cross-platform compatibility.

## Components

- **thermogram_baseline**: Package for baseline detection, subtraction, and signal processing
- **tlbparam**: Package for parameter calculation and statistical analysis
- **Web Application**: Interactive Dash-based interface for data visualization and processing

## Installation

*Coming soon*

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/ThermogramForge.git
cd ThermogramForge

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
