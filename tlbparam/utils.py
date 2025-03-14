"""
Utility functions for thermogram data analysis.

This module provides utility functions for loading, formatting, and
processing thermogram data for the tlbparam package.
"""

import pathlib
from typing import List, Optional, Union

import numpy as np
import polars as pl


def load_thermograms(
    file: Union[str, pathlib.Path],
    sheet: str = "Sheet1",
    blank_row: int = 1,
    temps: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Load thermogram data from Excel or CSV files.

    This function replicates the R load_thermograms function, reading and
    transposing data appropriately.

    Args:
        file: Path to Excel or CSV file
        sheet: Sheet name (for Excel files)
        blank_row: Number of blank rows to skip
        temps: List of temperature column names. If None, defaults to
            ['T45.0', 'T45.1', ..., 'T90.0']

    Returns:
        DataFrame with thermogram data in wide format
    """
    # This is a placeholder implementation that will be fully developed in Sprint 5
    # For now, return an empty DataFrame with expected structure
    if temps is None:
        temps = [f"T{temp:.1f}" for temp in np.arange(45.0, 90.1, 0.1)]

    # Create an empty DataFrame with SampleCode column
    result = pl.DataFrame({"SampleCode": []})

    # Add temperature columns with empty data
    for temp in temps:
        result = result.with_columns(pl.lit(None).alias(temp))

    return result


def add_thermogram_lag(
    thermogram_data: pl.DataFrame,
    lag_data: pl.DataFrame,
    sample_id_col: str = "SampleCode",
    lag_id_col: str = "SampleCode",
) -> pl.DataFrame:
    """
    Add lag data to thermogram metrics.

    This function replicates the R add_thermogram_lag function, adding
    lag data to thermogram metrics.

    Args:
        thermogram_data: DataFrame with thermogram metrics
        lag_data: DataFrame with lag data
        sample_id_col: Column name containing sample IDs in thermogram data
        lag_id_col: Column name containing sample IDs in lag data

    Returns:
        DataFrame with thermogram metrics and lag data
    """
    # This is a placeholder implementation that will be fully developed in Sprint 7
    # For now, return the original data
    return thermogram_data
