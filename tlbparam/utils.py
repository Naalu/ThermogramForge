"""
Utility functions for thermogram data analysis.

This module provides utility functions for loading, formatting, and
processing thermogram data for the tlbparam package.
"""

import concurrent.futures
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import polars as pl


def load_thermograms(
    file: Union[str, Path],
    sheet: str = "Sheet1",
    blank_row: int = 1,
    temps: Optional[List[str]] = None,
    use_lazy: bool = True,
) -> pl.DataFrame:
    """
    Load processed thermogram data from Excel or CSV files.

    This function replicates the R load_thermograms function, reading and
    transposing data appropriately. It assumes the input file has temperatures
    as rows and samples as columns, and will transpose the data to have samples
    as rows and temperatures as columns.

    Args:
        file: Path to Excel or CSV file
        sheet: Sheet name (for Excel files)
        blank_row: Number of blank rows to skip
        temps: List of temperature column names. If None, defaults to
            ['T45.0', 'T45.1', ..., 'T90.0']
        use_lazy: Whether to use lazy evaluation for loading (better for large files)

    Returns:
        DataFrame with thermogram data in wide format, where:
            - Each row represents a sample
            - The first column is 'SampleCode' with sample identifiers
            - Remaining columns are temperature columns (T45.0, T45.1, etc.)

    Raises:
        ValueError: If the file format is unsupported or if the file cannot be read
        FileNotFoundError: If the specified file does not exist

    Examples:
        >>> # Load from Excel
        >>> data = load_thermograms("data/thermogram_data.xlsx", sheet="DSC Data")
        >>>
        >>> # Load from CSV with custom temperature range
        >>> temps = [f"T{t:.1f}" for t in np.arange(50, 85, 0.1)]
        >>> data = load_thermograms("data/thermogram_data.csv", temps=temps)
    """
    # Convert file to Path object for easier handling
    file_path = Path(file) if isinstance(file, str) else file

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Set default temperature columns if not provided
    if temps is None:
        temps = [f"T{temp:.1f}" for temp in np.arange(45.0, 90.1, 0.1)]

    # Determine file type based on extension
    suffix = file_path.suffix.lower()

    try:
        # Load data based on file type
        if suffix == ".xlsx" or suffix == ".xls":
            # Lazy loading not supported for Excel, use regular method
            raw_data = pl.read_excel(file_path, sheet_name=sheet)
        elif suffix == ".csv":
            # Use lazy loading if requested (better for large files)
            if use_lazy:
                raw_data = pl.scan_csv(file_path).collect()
            else:
                raw_data = pl.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Check if file is empty
        if raw_data.height == 0 or raw_data.width == 0:
            # Return empty DataFrame with expected schema
            empty_df = pl.DataFrame({"SampleCode": []})
            for temp in temps:
                empty_df = empty_df.with_columns(pl.lit(None).alias(temp))
            return empty_df

        # Manual transpose operation (without using pandas)
        # First get column names (except first column if it exists)
        if raw_data.width > 1:
            # First column might be row labels
            sample_names = raw_data.columns[1:]
            # Get values from the first column as temperature indices
            temp_indices = (
                raw_data.select(pl.col(raw_data.columns[0])).to_series().to_list()
            )
        else:
            # No sample labels, use generic names
            sample_names = [f"Sample{i + 1}" for i in range(raw_data.width)]
            # Generate temperature indices
            temp_indices = [f"Row{i + 1}" for i in range(raw_data.height)]

        # Skip blank rows if specified
        if blank_row > 0:
            if blank_row < len(temp_indices):
                temp_indices = temp_indices[blank_row:]
                # Get data rows after blank rows
                raw_data = raw_data.slice(blank_row)
            else:
                # Not enough rows
                empty_df = pl.DataFrame({"SampleCode": []})
                for temp in temps:
                    empty_df = empty_df.with_columns(pl.lit(None).alias(temp))
                return empty_df

        # Create result dictionary with SampleCode column
        result_dict = {"SampleCode": sample_names}

        # Process all temperature values at once using expression-based operations
        for i, _ in enumerate(temp_indices):
            if i < len(temps):
                temp_col = temps[i]
                # Skip if we don't have enough rows
                if i < raw_data.height:
                    # Convert row to a Series of values for all samples
                    if raw_data.width > 1:
                        # Skip first column (used for temp indices)
                        result_dict[temp_col] = raw_data.row(i)[1:]  # type: ignore
                    else:
                        result_dict[temp_col] = raw_data.row(i)  # type: ignore

        # Create result DataFrame
        result = pl.DataFrame(result_dict)

        # Ensure all expected temperature columns exist (more efficiently)
        missing_temps = [temp for temp in temps if temp not in result.columns]
        if missing_temps:
            # Could we add all missing columns at once with a single operation w/
            # missing_cols={temp: [None] * len(sample_names) for temp in missing_temps}
            result = result.with_columns(
                [pl.lit(None).alias(temp) for temp in missing_temps]
            )

        return result

    except Exception as e:
        raise ValueError(f"Error loading thermogram data: {str(e)}") from e


def load_raw_thermograms(
    file: Union[str, Path],
    sheet: str = "Sheet1",
    use_lazy: bool = True,
    num_threads: int = 4,
) -> pl.DataFrame:
    """
    Load raw thermogram data from lab output files (CSV or Excel).

    This function loads thermogram data where columns come in pairs:
    - T[SampleID]: Temperature values for the sample
    - [SampleID]: Measurement values for the sample

    It converts this format to a "long" format with columns:
    - SampleID: Identifier for the sample
    - Temperature: Temperature at which measurement was taken
    - dCp: Measured value

    Args:
        file: Path to Excel or CSV file
        sheet: Sheet name (for Excel files)
        use_lazy: Whether to use lazy evaluation for CSV loading (for large files)
        num_threads: Number of threads to use for parallel processing of samples

    Returns:
        DataFrame with thermogram data in long format

    Raises:
        ValueError: If the file format is unsupported or if the file cannot be read
        FileNotFoundError: If the specified file does not exist

    Examples:
        >>> # Load from CSV
        >>> data = load_raw_thermograms("raw_lab_data.csv")
        >>>
        >>> # Load from Excel
        >>> data = load_raw_thermograms("raw_lab_data.xlsx", sheet="Raw Data")
        >>>
        >>> # Load large file efficiently
        >>> data = load_raw_thermograms("large_data.csv", use_lazy=True, num_threads=8)
    """
    # Convert file to Path object for easier handling
    file_path = Path(file) if isinstance(file, str) else file

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type based on extension
    suffix = file_path.suffix.lower()

    try:
        # Load data based on file type
        if suffix == ".xlsx" or suffix == ".xls":
            # Excel files - lazy loading not supported
            raw_data = pl.read_excel(file_path, sheet_name=sheet)
        elif suffix == ".csv":
            # CSV files - can use lazy loading
            if use_lazy:
                # Use schema inference hints for better performance
                schema_overrides = {
                    f"col_{i}": pl.Float64 for i in range(1000)
                }  # Large number to cover all columns
                raw_data = pl.scan_csv(
                    file_path, schema_overrides=schema_overrides
                ).collect()
            else:
                raw_data = pl.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Identify paired columns (Temperature and Value)
        # Find all columns that start with 'T' and check if the
        # corresponding value column exists
        sample_pairs = []

        # More efficient column pairing detection using vectorized operations
        temp_cols = [col for col in raw_data.columns if col.startswith("T")]
        for temp_col in temp_cols:
            # Extract sample ID by removing 'T' prefix
            sample_id = temp_col[1:]
            if sample_id in raw_data.columns:
                sample_pairs.append((temp_col, sample_id))

        if not sample_pairs:
            raise ValueError(
                "No valid thermogram column pairs found (T[SampleID], [SampleID])"
            )

        # Process sample pairs in parallel for large datasets
        if len(sample_pairs) > 1 and num_threads > 1:
            # Process each sample pair in a separate thread
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                # Submit tasks for each sample pair
                future_to_sample = {
                    executor.submit(
                        _process_sample_pair, raw_data, temp_col, value_col
                    ): (temp_col, value_col)
                    for temp_col, value_col in sample_pairs
                }

                # Collect results as they complete
                sample_dataframes = []
                for future in concurrent.futures.as_completed(future_to_sample):
                    pair = future_to_sample[future]
                    try:
                        result = future.result()
                        if result is not None and len(result) > 0:
                            sample_dataframes.append(result)
                    except Exception as e:
                        print(f"Error processing sample pair {pair}: {e}")

                # Combine results
                if sample_dataframes:
                    result = pl.concat(sample_dataframes)
                else:
                    # No valid data found
                    result = pl.DataFrame(
                        {"SampleID": [], "Temperature": [], "dCp": []}
                    )
        else:
            # Single-threaded processing for small datasets
            sample_dataframes = []

            for temp_col, value_col in sample_pairs:
                sample_df = _process_sample_pair(raw_data, temp_col, value_col)
                if sample_df is not None and len(sample_df) > 0:
                    sample_dataframes.append(sample_df)

            # Combine results
            if sample_dataframes:
                result = pl.concat(sample_dataframes)
            else:
                # No valid data found
                result = pl.DataFrame({"SampleID": [], "Temperature": [], "dCp": []})

        # Sort by SampleID and Temperature
        result = result.sort(["SampleID", "Temperature"])

        return result

    except Exception as e:
        raise ValueError(f"Error loading raw thermogram data: {str(e)}") from e


def _process_sample_pair(
    raw_data: pl.DataFrame, temp_col: str, value_col: str
) -> pl.DataFrame | None:
    """
    Process a single sample pair (temperature and value columns) from raw data.

    Args:
        raw_data: DataFrame containing the raw data
        temp_col: Name of the temperature column
        value_col: Name of the value column

    Returns:
        DataFrame with processed data for this sample
    """
    # Extract sample ID from value column name
    sample_id = value_col

    # Efficiently filter out null values using expressions
    # Note: We use pl.col().is_not_null() to filter out nulls and NaNs
    valid_data = raw_data.filter(
        pl.col(temp_col).is_not_null()
        & pl.col(value_col).is_not_null()
        & ~pl.col(temp_col).is_nan()
        & ~pl.col(value_col).is_nan()
    )

    # Extract temperature and value columns
    temps = valid_data.select(pl.col(temp_col)).to_series()
    values = valid_data.select(pl.col(value_col)).to_series()

    # Check if we have any valid data
    if len(temps) == 0:
        return None

    # Create result DataFrame directly with correct structure
    return pl.DataFrame(
        {"SampleID": [sample_id] * len(temps), "Temperature": temps, "dCp": values}
    )


def convert_long_to_wide(
    data: pl.DataFrame,
    sample_id_col: str = "SampleID",
    temp_col: str = "Temperature",
    value_col: str = "dCp",
    temp_grid: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """
    Convert thermogram data from long format to wide format.

    This function converts data from long format (one row per measurement)
    to wide format (one row per sample, columns for each temperature).

    Args:
        data: DataFrame in long format
        sample_id_col: Column name for sample identifiers
        temp_col: Column name for temperature values
        value_col: Column name for measured values
        temp_grid: Optional temperature grid for interpolation
                  If None, uses actual temperatures from data

    Returns:
        DataFrame in wide format with:
        - First column: sample_id_col with sample identifiers
        - Remaining columns: temperature values with T{temp} names

    Examples:
        >>> # Convert from long to wide
        >>> long_data = load_raw_thermograms("raw_data.csv")
        >>> wide_data = convert_long_to_wide(long_data)
        >>>
        >>> # Convert with specific temperature grid
        >>> grid = np.arange(45, 90.1, 0.1)
        >>> wide_data = convert_long_to_wide(long_data, temp_grid=grid)
    """
    # Check required columns
    if not all(col in data.columns for col in [sample_id_col, temp_col, value_col]):
        raise ValueError(
            f"Required columns missing. Need "
            f"{sample_id_col}, {temp_col}, and {value_col}"
        )

    # Get unique sample IDs - use to_series() for better performance
    sample_ids = (
        data.select(pl.col(sample_id_col))
        .unique()
        .sort(by=sample_id_col)
        .to_series()
        .to_list()
    )

    if not sample_ids:
        # No samples found
        return pl.DataFrame({sample_id_col: []})

    # Determine temperature grid
    if temp_grid is None:
        # Use actual temperatures from data - sorted for consistency
        temps = (
            data.select(pl.col(temp_col))
            .unique()
            .sort(by=temp_col)
            .to_series()
            .to_numpy()
        )
    else:
        temps = temp_grid

    # Create column names for temperatures
    temp_cols = [f"T{t:.1f}" for t in temps]

    # More efficient pivoting using Polars built-in pivot functionality
    # This is much faster than manual for-loops
    try:
        # If temp_grid is None, we can use direct pivoting
        if temp_grid is None:
            # Create temporary column with temperature labels for pivoting
            temp_labels = data.with_columns(
                pl.col(temp_col).map(lambda t: f"T{t:.1f}").alias("temp_label")
            )

            # Use Polars pivot operation
            result = temp_labels.pivot(
                index=sample_id_col,
                on="temp_label",
                values=value_col,
                aggregate_function="first",  # Use first value if duplicates exist
            )

            # Rename the index column back to sample_id_col if needed
            if result.columns[0] != sample_id_col:
                result = result.rename({result.columns[0]: sample_id_col})

            # Ensure all expected columns exist
            for col in temp_cols:
                if col not in result.columns:
                    result = result.with_columns(pl.lit(None).alias(col))

            return result

        else:
            # When temp_grid is specified, we need to map data to grid points
            # Group by sample ID
            sample_groups = data.group_by(sample_id_col)

            # Initialize result dictionary
            result_dict = {sample_id_col: sample_ids}
            result_dict.update(
                {temp_col: [None] * len(sample_ids) for temp_col in temp_cols}
            )

            # Create mapping of sample ID to index
            sample_id_to_idx = {sample_id: i for i, sample_id in enumerate(sample_ids)}

            # Process each group efficiently with apply
            for sample_id, group in zip(
                sample_groups.groups[sample_id_col], sample_groups.groups.data
            ):
                sample_idx = sample_id_to_idx[sample_id]

                # For each temperature/value pair, find closest grid point
                for temp, value in zip(group[temp_col], group[value_col]):
                    # Find closest grid temp
                    grid_idx = np.abs(temps - temp).argmin()
                    grid_temp = temps[grid_idx]
                    temp_col_name = f"T{grid_temp:.1f}"

                    # Update result
                    if temp_col_name in result_dict:
                        result_dict[temp_col_name][sample_idx] = value

            # Create DataFrame from result dict
            return pl.DataFrame(result_dict)

    except Exception:
        # Fallback method if pivot fails
        # We'll implement a more manual approach but still optimize where possible

        # Initialize result dictionary
        result_dict = {sample_id_col: sample_ids}

        # Initialize all temperature columns with None (avoids repetitive dict updates)
        for temp_col_name in temp_cols:
            result_dict[temp_col_name] = [None] * len(sample_ids)

        # Create mapping of sample ID to index
        sample_id_to_idx = {sample_id: i for i, sample_id in enumerate(sample_ids)}

        # Efficiently batch process rows by temperature range
        # This reduces the number of dict updates which are expensive
        if temp_grid is not None:
            # When using a specific grid, we need to map temps to nearest grid point
            for i in range(len(temps)):
                grid_temp = temps[i]
                temp_col_name = f"T{grid_temp:.1f}"

                # Find data points near this grid temperature
                # We look for points within a small range of the grid point
                lower_bound = (
                    grid_temp - 0.05 if i == 0 else (grid_temp + temps[i - 1]) / 2
                )
                upper_bound = (
                    grid_temp + 0.05
                    if i == len(temps) - 1
                    else (grid_temp + temps[i + 1]) / 2
                )

                temp_data = data.filter(
                    (pl.col(temp_col) >= lower_bound) & (pl.col(temp_col) < upper_bound)
                )

                # Process all points for this temperature at once
                for row in temp_data.rows():
                    sample = row[temp_data.columns.index(sample_id_col)]
                    value = row[temp_data.columns.index(value_col)]

                    if sample in sample_id_to_idx:
                        sample_idx = sample_id_to_idx[sample]
                        result_dict[temp_col_name][sample_idx] = value
        else:
            # When using actual temperatures, we can directly map
            for _, row in enumerate(data.iter_rows(named=True)):
                sample = row[sample_id_col]
                temp = row[temp_col]
                value = row[value_col]

                # Format temperature column name
                temp_col_name = f"T{temp:.1f}"

                # Update result dict
                if temp_col_name in result_dict and sample in sample_id_to_idx:
                    sample_idx = sample_id_to_idx[sample]
                    result_dict[temp_col_name][sample_idx] = value

        # Create result DataFrame
        return pl.DataFrame(result_dict)


def convert_wide_to_long(
    data: pl.DataFrame,
    sample_id_col: str = "SampleCode",
) -> pl.DataFrame:
    """
    Convert thermogram data from wide format to long format.

    This function converts data from wide format (one row per sample,
    columns for each temperature) to long format (one row per measurement).

    Args:
        data: DataFrame in wide format
        sample_id_col: Column name for sample identifiers

    Returns:
        DataFrame in long format with columns:
        - SampleID: Sample identifier
        - Temperature: Temperature value
        - dCp: Measured value

    Examples:
        >>> # Convert from wide to long
        >>> wide_data = load_thermograms("processed_data.csv")
        >>> long_data = convert_wide_to_long(wide_data)
    """
    # Check if sample_id_col exists
    if sample_id_col not in data.columns:
        raise ValueError(f"Sample ID column '{sample_id_col}' not found in data")

    # Identify temperature columns (starting with 'T')
    temp_cols = [
        col for col in data.columns if col.startswith("T") and col != sample_id_col
    ]

    if not temp_cols:
        raise ValueError("No temperature columns found (columns starting with 'T')")

    # Use Polars' unpivot operation for efficient conversion to long format
    # This is much faster than manual loops
    long_data = data.unpivot(
        index=[sample_id_col],
        on=temp_cols,
        variable_name="temp_col",
        value_name="dCp",
    )

    # Extract temperature values from column names
    long_data = long_data.with_columns(
        pl.col("temp_col").str.replace("T", "").cast(pl.Float64).alias("Temperature")
    )

    # Filter out null or NaN values efficiently
    long_data = long_data.filter(pl.col("dCp").is_not_null() & ~pl.col("dCp").is_nan())

    # Rename SampleCode to SampleID for consistency with load_raw_thermograms
    if sample_id_col != "SampleID":
        long_data = long_data.rename({sample_id_col: "SampleID"})

    # Select only needed columns and sort
    result = long_data.select(["SampleID", "Temperature", "dCp"]).sort(
        by=["SampleID", "Temperature"]
    )

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
