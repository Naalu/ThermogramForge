"""
Global pytest configuration and fixtures.
"""

from importlib.util import find_spec

import numpy as np
import polars as pl
import pytest

# Set random seed for reproducibility
np.random.seed(42)


@pytest.fixture
def sample_thermogram_data():
    """Generate synthetic thermogram data for testing."""
    # Create temperature range
    temps = np.linspace(45, 90, 100)

    # Create synthetic data with peaks
    peak1 = 0.3 * np.exp(-0.5 * ((temps - 63) / 2) ** 2)
    peak2 = 0.2 * np.exp(-0.5 * ((temps - 70) / 2) ** 2)
    peak3 = 0.15 * np.exp(-0.5 * ((temps - 77) / 2.5) ** 2)

    # Add baseline and noise
    baseline = 0.02 * (temps - 65)
    noise = 0.01 * np.random.randn(len(temps))

    # Combined signal
    dcp = peak1 + peak2 + peak3 + baseline + noise

    # Use polars DataFrame
    return pl.DataFrame({"Temperature": temps, "dCp": dcp})


@pytest.fixture
def has_r():
    """Check if R is available for testing."""
    return find_spec("rpy2") is not None
