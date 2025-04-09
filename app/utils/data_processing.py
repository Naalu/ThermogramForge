"""
Data processing utilities for thermogram analysis.

Includes functions for preprocessing raw data, extracting individual samples
from multi-sample files, detecting baseline endpoints, and interpolating data.
"""

import logging
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_thermogram_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw thermogram data.

    Renames columns if necessary (assuming Temperature, dCp are first two),
    converts columns to numeric types, drops NaN values, sorts by temperature,
    and resets the index.

    Args:
        df: Raw DataFrame, potentially with incorrect column names or types.

    Returns:
        Processed DataFrame with 'Temperature' and 'dCp' columns, sorted,
        cleaned, and indexed.
    """
    # Ensure we are working with a copy to avoid SettingWithCopyWarning
    df = df.copy()
    logger.debug(f"Preprocessing DataFrame with initial shape {df.shape}")
    # Check for required columns
    if "Temperature" not in df.columns or "dCp" not in df.columns:
        # Try to guess columns - first column might be temp, second might be dCp
        if len(df.columns) >= 2:
            logger.warning(
                "'Temperature' or 'dCp' not found. Renaming first two columns."
            )
            df = df.rename(columns={df.columns[0]: "Temperature", df.columns[1]: "dCp"})
        else:
            logger.error(
                "Cannot preprocess: DataFrame has less than 2 columns and lacks 'Temperature'/'dCp'."
            )
            # Return empty dataframe with expected columns?
            return pd.DataFrame(columns=["Temperature", "dCp"])

    # Make sure columns are the right type
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["dCp"] = pd.to_numeric(df["dCp"], errors="coerce")

    # Drop NaN values
    initial_rows = len(df)
    df = df.dropna(subset=["Temperature", "dCp"])
    if len(df) < initial_rows:
        logger.debug(
            f"Dropped {initial_rows - len(df)} rows with NaN in Temperature or dCp."
        )

    # Sort by temperature
    df = df.sort_values("Temperature")

    # Reset index
    df = df.reset_index(drop=True)
    logger.debug(f"Preprocessing complete. Final shape {df.shape}")
    return df


def extract_samples(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extracts individual samples based on paired column names (e.g., T[ID] and ID).

    Iterates through columns, identifies pairs like 'T[SampleName]' and 'SampleName',
    extracts the corresponding data, renames columns to 'Temperature' and 'dCp',
    and performs initial cleaning.

    Args:
        df: The input DataFrame, potentially containing multiple sample pairs.

    Returns:
        A dictionary where keys are the extracted sample IDs (as strings) and
        values are DataFrames containing the cleaned numeric data (Temperature, dCp)
        for each corresponding sample. Returns an empty dictionary if no valid sample
        pairs are found.
    """
    samples: Dict[str, pd.DataFrame] = {}
    column_names = df.columns
    logger.info(
        f"Starting sample extraction (paired column method) from DataFrame with shape {df.shape}"
    )
    logger.debug(f"Input columns: {list(column_names)}")

    # Regex to find temperature columns like T[SampleID] or TSampleID
    temp_col_pattern = re.compile(r"^T(.*)$")

    processed_ids = set()

    for col_name in column_names:
        match = temp_col_pattern.match(col_name)
        if match:
            sample_id = match.group(1)  # Extract the SampleID
            logger.debug(
                f"Found potential temperature column: '{col_name}' with ID: '{sample_id}'"
            )

            # Check if we already processed this ID (to avoid duplicates if columns aren't perfectly paired)
            if sample_id in processed_ids:
                continue

            # Look for the corresponding dCp column (exact match for sample_id)
            dcp_col_name = sample_id
            if dcp_col_name in column_names:
                logger.info(
                    f"Found matching pair: Temp='{col_name}', dCp='{dcp_col_name}' for ID='{sample_id}'"
                )

                # Extract the two columns
                try:
                    sample_df_raw = df[[col_name, dcp_col_name]].copy()
                    sample_df_raw.rename(
                        columns={col_name: "Temperature", dcp_col_name: "dCp"},
                        inplace=True,
                    )

                    # Initial Cleaning (Convert to numeric, drop NaNs)
                    sample_df_raw["Temperature"] = pd.to_numeric(
                        sample_df_raw["Temperature"], errors="coerce"
                    )
                    sample_df_raw["dCp"] = pd.to_numeric(
                        sample_df_raw["dCp"], errors="coerce"
                    )
                    initial_rows = len(sample_df_raw)
                    sample_df_cleaned = sample_df_raw.dropna(
                        subset=["Temperature", "dCp"]
                    )

                    if sample_df_cleaned.empty:
                        logger.warning(
                            f"Sample '{sample_id}' resulted in empty DataFrame after initial cleaning (NaNs?). Skipping."
                        )
                    else:
                        rows_dropped = initial_rows - len(sample_df_cleaned)
                        logger.debug(
                            f"Sample '{sample_id}' initial cleaning: Dropped {rows_dropped} rows. Shape: {sample_df_cleaned.shape}"
                        )
                        # Store the cleaned (but not yet fully preprocessed/sorted) DataFrame
                        samples[sample_id] = sample_df_cleaned
                        processed_ids.add(sample_id)

                except KeyError as ke:
                    logger.error(
                        f"KeyError extracting columns for sample '{sample_id}': {ke}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing pair for sample '{sample_id}': {e}",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    f"Found temperature column '{col_name}' but no matching dCp column '{dcp_col_name}'"
                )

    if not samples:
        logger.warning("No valid sample column pairs found in the provided DataFrame.")
    else:
        logger.info(
            f"Finished initial sample extraction. Found {len(samples)} samples: {list(samples.keys())}"
        )

    # --- Post-process extracted samples --- Start
    # Apply full preprocessing (sorting, etc.) to each extracted sample DataFrame
    processed_samples: Dict[str, pd.DataFrame] = {}
    logger.info("Starting post-processing of extracted samples.")
    for sample_id, sample_df in samples.items():
        logger.debug(
            f"Post-processing sample '{sample_id}' (Shape before: {sample_df.shape})"
        )
        try:
            processed_df = preprocess_thermogram_data(sample_df)
            if not processed_df.empty:
                processed_samples[sample_id] = processed_df
                logger.debug(
                    f"Successfully preprocessed extracted sample '{sample_id}'. Final Shape: {processed_df.shape}"
                )
            else:
                logger.warning(
                    f"Preprocessing resulted in an empty DataFrame for sample '{sample_id}'. Skipping."
                )
        except Exception as e:
            logger.error(
                f"Error during final preprocessing of extracted sample '{sample_id}': {e}",
                exc_info=True,
            )
    logger.info("Finished post-processing of extracted samples.")
    # --- Post-process extracted samples --- End

    return processed_samples


def detect_endpoints(
    df: pd.DataFrame,
    window_size: int = 5,
    exclusion_window: Optional[Tuple[float, float]] = (60, 80),
) -> Dict[str, Optional[float]]:
    """Detects potential baseline endpoints using rolling variance analysis.

    Calculates the rolling variance of the 'dCp' values and finds the temperatures
    corresponding to the minimum variance outside a specified exclusion window
    (typically the main transition region).

    Args:
        df: DataFrame with 'Temperature' and 'dCp' columns, sorted by Temperature.
        window_size: The size of the rolling window used for variance calculation.
        exclusion_window: A tuple (min_temp, max_temp) defining the temperature range
                          to exclude from the search for minimum variance.
                          If None, no region is excluded.

    Returns:
        A dictionary with keys 'lower' and 'upper', containing the temperatures
        corresponding to the detected lower and upper baseline endpoints, respectively.
        Values can be None if detection fails.
    """
    if df.empty or "Temperature" not in df.columns or "dCp" not in df.columns:
        logger.warning("detect_endpoints: Input DataFrame is empty or missing columns.")
        return {"lower": None, "upper": None}

    # Ensure data is sorted by Temperature (should be from preprocessing)
    df_sorted = df.sort_values("Temperature").reset_index(drop=True)

    # Calculate rolling variance
    try:
        df_sorted["dCp_variance"] = (
            df_sorted["dCp"].rolling(window=window_size, center=True).var()
        )
    except Exception as e:
        logger.error(f"Error calculating rolling variance: {e}", exc_info=True)
        return {"lower": None, "upper": None}

    # --- Apply Exclusion Window --- Start
    df_analysis = df_sorted.copy()
    if exclusion_window is not None:
        min_exclude, max_exclude = exclusion_window
        if min_exclude is not None and max_exclude is not None:
            # Filter out the exclusion zone for endpoint detection
            df_analysis = df_analysis[
                (df_analysis["Temperature"] < min_exclude)
                | (df_analysis["Temperature"] > max_exclude)
            ]
            logger.debug(f"Applied exclusion window: {min_exclude}°C - {max_exclude}°C")
        else:
            logger.warning(
                "Exclusion window provided but contains None. Ignoring window."
            )
    # --- Apply Exclusion Window --- End

    if df_analysis.empty or df_analysis["dCp_variance"].isnull().all():
        logger.warning(
            "No data points remain after applying exclusion window, or all variances are NaN."
        )
        return {"lower": None, "upper": None}

    # --- Find Minimum Variance Points --- Start
    # Helper function to find min variance index within a given DataFrame slice
    def find_min_var_temp(data_slice: pd.DataFrame) -> Optional[float]:
        if data_slice.empty or data_slice["dCp_variance"].isnull().all():
            return None
        try:
            min_var_idx = data_slice["dCp_variance"].idxmin()
            return data_slice.loc[min_var_idx, "Temperature"]
        except Exception as e:
            logger.error(f"Error finding min variance temp: {e}", exc_info=True)
            return None

    lower_endpoint_temp = None
    upper_endpoint_temp = None

    # Define regions for lower and upper endpoint search
    # Use the midpoint of the *original* data range (or exclusion window if provided) to split
    midpoint_temp = df_sorted["Temperature"].mean()
    if exclusion_window and exclusion_window[0] is not None:
        midpoint_temp = exclusion_window[0]  # Use start of exclusion as rough split

    lower_region = df_analysis[df_analysis["Temperature"] <= midpoint_temp]
    upper_region = df_analysis[df_analysis["Temperature"] > midpoint_temp]

    lower_endpoint_temp = find_min_var_temp(lower_region)
    upper_endpoint_temp = find_min_var_temp(upper_region)

    logger.info(
        f"Detected endpoints: Lower={lower_endpoint_temp}, Upper={upper_endpoint_temp}"
    )
    # --- Find Minimum Variance Points --- End

    return {"lower": lower_endpoint_temp, "upper": upper_endpoint_temp}


def interpolate_thermogram(
    df: pd.DataFrame,
    temp_grid: Optional[np.ndarray] = None,
    num_points: int = 500,  # Default number of points if no grid specified
) -> Optional[pd.DataFrame]:
    """Interpolates thermogram data onto a specified or uniform temperature grid.

    Uses linear interpolation.

    Args:
        df: DataFrame with 'Temperature' and 'dCp' columns, sorted by Temperature.
        temp_grid: Optional NumPy array representing the desired temperature grid for interpolation.
                   If None, a uniform grid with `num_points` between the min and max
                   temperature of the input df will be generated.
        num_points: The number of points to use for the uniform grid if `temp_grid` is None.

    Returns:
        A DataFrame with interpolated 'dCp' values on the target temperature grid,
        or None if interpolation fails or input is invalid.
    """
    if df.empty or "Temperature" not in df.columns or "dCp" not in df.columns:
        logger.warning("interpolate_thermogram: Input DataFrame invalid.")
        return None

    if temp_grid is None:
        min_temp = df["Temperature"].min()
        max_temp = df["Temperature"].max()
        if pd.isna(min_temp) or pd.isna(max_temp):
            logger.warning(
                "interpolate_thermogram: Cannot determine temp range for uniform grid."
            )
            return None
        temp_grid = np.linspace(min_temp, max_temp, num=num_points)
        logger.debug(
            f"Generated uniform temp grid with {num_points} points from {min_temp:.1f} to {max_temp:.1f}"
        )
    else:
        # Ensure provided grid is sorted
        temp_grid = np.sort(temp_grid)

    try:
        # Perform linear interpolation
        interpolated_dcp = np.interp(
            temp_grid, df["Temperature"], df["dCp"], left=np.nan, right=np.nan
        )

        interpolated_df = pd.DataFrame(
            {"Temperature": temp_grid, "dCp": interpolated_dcp}
        )

        # Removed the dropna() call to preserve NaNs and ensure consistent length
        # interpolated_df = interpolated_df.dropna()
        logger.info(f"Interpolation complete. Result shape: {interpolated_df.shape}")
        return interpolated_df

    except Exception as e:
        logger.error(f"Error during interpolation: {e}", exc_info=True)
        return None
