"""
Data processing utilities for thermogram analysis.

Includes functions for preprocessing raw data, extracting individual samples
from multi-sample files, and interpolating data onto a defined grid.
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)


def preprocess_thermogram_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a single thermogram sample DataFrame.

    Standardizes column names to 'Temperature' and 'dCp' (assuming they are the
    first two columns if not named correctly). Converts these columns to numeric
    types, coercing errors to NaN. Drops rows with NaN in either column.
    Sorts the data by 'Temperature' and resets the index.

    Args:
        df: Raw DataFrame representing a single thermogram sample.

    Returns:
        A processed DataFrame with standardized 'Temperature' and 'dCp' columns,
        sorted, cleaned, and with a reset index. Returns an empty DataFrame with
        the expected columns if preprocessing fails or results in no valid data.

    Raises:
        ValueError: If the input is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Work with a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    logger.debug(f"Preprocessing DataFrame with initial shape {processed_df.shape}")

    # Standardize column names if necessary
    required_cols = ["Temperature", "dCp"]
    if not all(col in processed_df.columns for col in required_cols):
        if len(processed_df.columns) >= 2:
            logger.warning(
                "'Temperature' or 'dCp' column names not found. "
                "Assuming first column is Temperature, second is dCp and renaming."
            )
            rename_map = {
                processed_df.columns[0]: "Temperature",
                processed_df.columns[1]: "dCp",
            }
            processed_df.rename(columns=rename_map, inplace=True)
        else:
            logger.error(
                "Cannot preprocess: DataFrame has < 2 columns and lacks required names."
            )
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=required_cols)

    # Ensure correct numeric types, coercing errors to NaN
    try:
        processed_df["Temperature"] = pd.to_numeric(
            processed_df["Temperature"], errors="coerce"
        )
        processed_df["dCp"] = pd.to_numeric(processed_df["dCp"], errors="coerce")
    except Exception as e:
        logger.error(f"Error converting columns to numeric: {e}", exc_info=True)
        # Return empty df if conversion fails catastrophically
        return pd.DataFrame(columns=required_cols)

    # Drop rows where Temperature or dCp became NaN after coercion
    initial_rows = len(processed_df)
    processed_df.dropna(subset=["Temperature", "dCp"], inplace=True)
    rows_dropped = initial_rows - len(processed_df)
    if rows_dropped > 0:
        logger.debug(
            f"Dropped {rows_dropped} rows with NaN in Temperature or dCp during preprocessing."
        )

    if processed_df.empty:
        logger.warning("Preprocessing resulted in an empty DataFrame.")
        return processed_df  # Return the empty frame

    # Sort by Temperature (crucial for many downstream processes)
    processed_df.sort_values("Temperature", inplace=True)

    # Reset index after sorting and dropping NaNs
    processed_df.reset_index(drop=True, inplace=True)

    logger.debug(f"Preprocessing complete. Final shape {processed_df.shape}")
    return processed_df


def extract_samples(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extracts and preprocesses individual samples from a multi-column DataFrame.

    Assumes samples are stored in paired columns, typically named 'T[SampleID]'
    for temperature and 'SampleID' for the corresponding dCp values (or similar
    patterns identifiable by regex).

    Args:
        df: The input DataFrame, potentially containing multiple sample pairs.

    Returns:
        A dictionary where keys are the extracted sample IDs (str) and values
        are the fully preprocessed DataFrames (containing 'Temperature' and 'dCp'
        columns, cleaned and sorted) for each sample. Returns an empty dictionary
        if no valid sample pairs are found or processed successfully.
    """
    extracted_samples: Dict[str, pd.DataFrame] = {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("extract_samples: Input DataFrame is empty or invalid.")
        return extracted_samples

    column_names: List[str] = df.columns.tolist()
    logger.info(
        f"Starting sample extraction (paired columns) from DataFrame with shape {df.shape}"
    )
    logger.debug(f"Input columns: {column_names}")

    # Regex to capture SampleID from temperature columns like T[SampleID] or TSampleID
    # Allows for optional brackets around the ID.
    temp_col_pattern = re.compile(r"^T\[?([\w\-. ]+)\]?$")  # More flexible ID chars

    processed_ids = set()

    for temp_col_name in column_names:
        match = temp_col_pattern.match(temp_col_name)
        if match:
            # Extract the SampleID (group 1 of the regex)
            sample_id = match.group(1).strip()
            logger.debug(
                f"Found potential temperature column: '{temp_col_name}' -> ID: '{sample_id}'"
            )

            # Avoid reprocessing the same sample ID
            if sample_id in processed_ids:
                logger.debug(f"Skipping already processed ID: '{sample_id}'")
                continue

            # Corresponding dCp column should match the extracted sample_id exactly
            dcp_col_name = sample_id
            if dcp_col_name in column_names:
                logger.info(
                    f"Found matching pair: Temp='{temp_col_name}', dCp='{dcp_col_name}' for ID='{sample_id}'"
                )

                # Extract and initially clean the data for this sample
                try:
                    # Select the pair of columns and create a copy
                    sample_df_raw = df[[temp_col_name, dcp_col_name]].copy()

                    # Rename columns to standard names 'Temperature' and 'dCp'
                    sample_df_raw.rename(
                        columns={temp_col_name: "Temperature", dcp_col_name: "dCp"},
                        inplace=True,
                    )

                    # Apply full preprocessing (numeric conversion, NaN drop, sort, reset index)
                    processed_sample_df = preprocess_thermogram_data(sample_df_raw)

                    if not processed_sample_df.empty:
                        extracted_samples[sample_id] = processed_sample_df
                        processed_ids.add(sample_id)
                        logger.debug(
                            f"Successfully extracted and preprocessed sample '{sample_id}'. Shape: {processed_sample_df.shape}"
                        )
                    else:
                        logger.warning(
                            f"Sample '{sample_id}' resulted in empty DataFrame after preprocessing. Skipping."
                        )

                except KeyError as ke:
                    logger.error(
                        f"KeyError extracting columns for sample '{sample_id}' (Temp: '{temp_col_name}', dCp: '{dcp_col_name}'): {ke}"
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing pair for sample '{sample_id}': {e}",
                        exc_info=True,
                    )
            else:
                # Log if a potential temp column doesn't have a matching dCp column
                logger.warning(
                    f"Found temperature column '{temp_col_name}' but no matching dCp column named '{dcp_col_name}'"
                )

    # Final log message summarizing results
    if not extracted_samples:
        logger.warning(
            "No valid sample column pairs were successfully extracted and processed."
        )
    else:
        logger.info(
            f"Finished sample extraction. Found {len(extracted_samples)} valid samples: {list(extracted_samples.keys())}"
        )

    return extracted_samples


def interpolate_thermogram(
    df: pd.DataFrame,
    temp_grid: Optional[np.ndarray] = None,
    num_points: int = 500,
) -> Optional[pd.DataFrame]:
    """Interpolates thermogram dCp values onto a specified or uniform temperature grid.

    Uses linear interpolation via numpy.interp.

    Args:
        df: DataFrame containing numeric 'Temperature' and 'dCp' columns.
            Data should ideally be preprocessed and sorted by Temperature.
        temp_grid: Optional 1D NumPy array representing the desired target
                   temperature grid for interpolation. If None, a uniform grid
                   with `num_points` between the min and max temperature of the
                   input `df` will be generated.
        num_points: The number of points to use for the uniform grid generation
                    if `temp_grid` is None. Ignored if `temp_grid` is provided.

    Returns:
        A new DataFrame with 'Temperature' (from the target grid) and interpolated
        'dCp' columns. Returns None if interpolation fails, input is invalid,
        or the temperature range cannot be determined.
    """
    # --- Input Validation --- Start
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("interpolate_thermogram: Input DataFrame is invalid or empty.")
        return None
    if "Temperature" not in df.columns or "dCp" not in df.columns:
        logger.error("interpolate_thermogram: DataFrame missing required columns.")
        return None
    if not is_numeric_dtype(df["Temperature"]) or not is_numeric_dtype(df["dCp"]):
        logger.error("interpolate_thermogram: Columns are not numeric.")
        return None
    if not df["Temperature"].is_monotonic_increasing:
        # Interpolation requires sorted x-values (temperature)
        logger.warning(
            "interpolate_thermogram: Input DataFrame is not sorted by Temperature. Sorting..."
        )
        df = df.sort_values("Temperature").copy()  # Sort and copy
    # --- Input Validation --- End

    # Determine the target temperature grid
    target_temp_grid: np.ndarray
    if temp_grid is None:
        min_temp = df["Temperature"].min()
        max_temp = df["Temperature"].max()
        if pd.isna(min_temp) or pd.isna(max_temp) or min_temp == max_temp:
            logger.warning(
                "interpolate_thermogram: Cannot determine valid temp range for uniform grid generation."
            )
            return None
        target_temp_grid = np.linspace(min_temp, max_temp, num=num_points)
        logger.debug(
            f"Generated uniform temp grid: {num_points} points from {min_temp:.2f} to {max_temp:.2f}"
        )
    else:
        if not isinstance(temp_grid, np.ndarray) or temp_grid.ndim != 1:
            logger.error(
                "interpolate_thermogram: Provided temp_grid must be a 1D NumPy array."
            )
            return None
        # Ensure provided grid is sorted for clarity, although np.interp handles unsorted target x.
        target_temp_grid = np.sort(temp_grid)
        logger.debug(f"Using provided temp grid with {len(target_temp_grid)} points.")

    try:
        # Perform linear interpolation using numpy.interp
        # np.interp requires sorted source x-values (df["Temperature"])
        # It handles the target x-values (target_temp_grid) potentially being outside source range.
        # `left=np.nan, right=np.nan` ensures points outside the original range get NaN.
        interpolated_dcp = np.interp(
            target_temp_grid,
            df["Temperature"].values,  # Source x (must be sorted)
            df["dCp"].values,  # Source y
            left=np.nan,  # Value for target x < min(source x)
            right=np.nan,  # Value for target x > max(source x)
        )

        # Create the resulting DataFrame
        interpolated_df = pd.DataFrame(
            {"Temperature": target_temp_grid, "dCp": interpolated_dcp}
        )

        logger.info(f"Interpolation complete. Result shape: {interpolated_df.shape}")
        return interpolated_df

    except Exception as e:
        logger.error(f"Error during interpolation: {e}", exc_info=True)
        return None
