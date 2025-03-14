# Thermogram Analysis

A Python implementation of thermogram analysis tools for thermal liquid biopsy (TLB) data.

## Overview

This project provides two main packages:

1. **thermogram_baseline**: For baseline subtraction in thermogram data
2. **tlbparam**: For calculating metrics from thermogram data

It also includes a web-based user interface for interactive analysis.

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thermogram-analysis.git
cd thermogram-analysis

# Create a virtual environment with uv
uv venv

# Install the package and its dependencies
uv sync
