"""Batch processing utilities for thermogram data."""

import concurrent.futures
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import polars as pl

from .baseline import auto_baseline
from .types import BatchProcessingResult, InterpolatedResult, ThermogramData


def process_multiple(
    data: Union[Dict[str, ThermogramData], Dict[str, pl.DataFrame], pl.DataFrame],
    window_size: int = 90,
    exclusion_lower: float = 60,
    exclusion_upper: float = 80,
    grid_temp: Optional[np.ndarray] = None,
    point_selection: Literal["innermost", "outmost", "mid"] = "outmost",
    verbose: bool = True,
    max_workers: Optional[int] = None,
    output_file: Optional[Union[str, Path]] = None,
    detect_signal: bool = False,
) -> BatchProcessingResult:
    """
    Process multiple thermograms in batch mode.

    Args:
        data: Multiple thermograms to process. Can be:
            - Dictionary mapping sample IDs to ThermogramData objects
            - Dictionary mapping sample IDs to DataFrames
            - DataFrame with a column specifying sample IDs
        window_size: Window size for endpoint detection
        exclusion_lower: Lower bound of the exclusion window
        exclusion_upper: Upper bound of the exclusion window
        grid_temp: Temperature grid for interpolation
        point_selection: Method for endpoint selection
        verbose: Whether to print progress information
        max_workers: Maximum number of parallel workers (None for auto)
        output_file: Optional path to save results
        detect_signal: Whether to perform signal detection

    Returns:
        BatchProcessingResult with processed thermograms
    """
    # Convert input to dictionary of ThermogramData if it's a DataFrame
    sample_data = _prepare_input_data(data)

    # Get sample IDs
    sample_ids = list(sample_data.keys())
    num_samples = len(sample_ids)

    if verbose:
        print(f"Processing {num_samples} thermograms in batch mode")

    # Create a default grid if not provided
    if grid_temp is None:
        grid_temp = _create_default_grid()

    # Process thermograms in parallel or sequentially
    results = {}
    processing_stats = {
        "total_samples": num_samples,
        "successful": 0,
        "failed": 0,
        "errors": {},
    }

    if max_workers == 1:
        # Sequential processing
        for i, sample_id in enumerate(sample_ids):
            if verbose:
                print(f"Processing sample {i + 1}/{num_samples}: {sample_id}")

            try:
                result = auto_baseline(
                    data=sample_data[sample_id],
                    window_size=window_size,
                    exclusion_lower=exclusion_lower,
                    exclusion_upper=exclusion_upper,
                    grid_temp=grid_temp,
                    point_selection=point_selection,
                    verbose=False,
                )

                results[sample_id] = result
                processing_stats["successful"] += 1

                if verbose:
                    print(f"  - Success: {sample_id}")

            except Exception as e:
                processing_stats["failed"] += 1
                processing_stats["errors"][sample_id] = str(e)
                if verbose:
                    print(f"  - Failed: {sample_id} - {str(e)}")

    else:
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Prepare arguments for each sample
            futures = {}
            for sample_id in sample_ids:
                future = executor.submit(
                    auto_baseline,
                    data=sample_data[sample_id],
                    window_size=window_size,
                    exclusion_lower=exclusion_lower,
                    exclusion_upper=exclusion_upper,
                    grid_temp=grid_temp,
                    point_selection=point_selection,
                    verbose=False,
                )
                futures[future] = sample_id

            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                sample_id = futures[future]
                if verbose:
                    print(f"Completed {i + 1}/{num_samples}: {sample_id}")

                try:
                    result = future.result()
                    results[sample_id] = result
                    processing_stats["successful"] += 1

                    if verbose:
                        print(f"  - Success: {sample_id}")

                except Exception as e:
                    processing_stats["failed"] += 1
                    processing_stats["errors"][sample_id] = str(e)
                    if verbose:
                        print(f"  - Failed: {sample_id} - {str(e)}")

    # Save results to file if requested
    if output_file is not None:
        _save_results(results, grid_temp, output_file)
        if verbose:
            print(f"Results saved to {output_file}")

    # Create and return result object
    batch_result = BatchProcessingResult(
        results=results, grid_temp=grid_temp, processing_stats=processing_stats
    )

    if verbose:
        print(
            f"Batch processing completed: {processing_stats['successful']} successful, "
            f"{processing_stats['failed']} failed"
        )

    return batch_result


def _prepare_input_data(
    data: Union[Dict[str, ThermogramData], Dict[str, pl.DataFrame], pl.DataFrame],
) -> Dict[str, ThermogramData]:
    """
    Convert input data to a dictionary of ThermogramData objects.

    Args:
        data: Input data in various formats

    Returns:
        Dictionary mapping sample IDs to ThermogramData objects
    """
    if isinstance(data, pl.DataFrame):
        # DataFrame with multiple samples
        sample_col = [
            col
            for col in data.columns
            if col.lower() in ["sample", "sampleid", "sample_id", "id"]
        ][0]

        # Group by sample ID
        grouped_data = {}
        for sample_id in data[sample_col].unique():
            sample_df = data.filter(pl.col(sample_col) == sample_id)
            # Assuming the DataFrame has Temperature and dCp columns
            grouped_data[str(sample_id)] = ThermogramData.from_dataframe(sample_df)

        return grouped_data

    elif isinstance(data, dict):
        # Dictionary of samples
        result = {}
        for sample_id, sample_data in data.items():
            if isinstance(sample_data, ThermogramData):
                result[str(sample_id)] = sample_data
            elif isinstance(sample_data, pl.DataFrame):
                result[str(sample_id)] = ThermogramData.from_dataframe(sample_data)
            else:
                raise ValueError(
                    f"Unsupported data type for sample {sample_id}: {type(sample_data)}"
                )

        return result

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def _create_default_grid(
    min_temp: float = 45.0,
    max_temp: float = 90.0,
    step: float = 0.1,
) -> np.ndarray:
    """
    Create a default temperature grid for interpolation.

    Args:
        min_temp: Minimum temperature
        max_temp: Maximum temperature
        step: Temperature step size

    Returns:
        NumPy array containing the temperature grid
    """
    return np.arange(min_temp, max_temp + step / 2, step)


def _save_results(
    results: Dict[str, InterpolatedResult],
    grid_temp: np.ndarray,
    path: Union[str, Path],
) -> None:
    """
    Save batch processing results to a file.

    Args:
        results: Dictionary of processed results
        grid_temp: Temperature grid
        path: File path to save to
    """
    # Convert to string path if it's a Path object
    path = str(path)

    # Create DataFrame from results
    if path.endswith(".csv"):
        # Wide format: each sample is a column
        data = {"Temperature": grid_temp}

        for sample_id, result in results.items():
            data[sample_id] = result.data.dcp

        df = pl.DataFrame(data)
        df.write_csv(path)

    elif path.endswith(".parquet"):
        # Save as parquet with more information
        data = []

        for sample_id, result in results.items():
            for i, temp in enumerate(grid_temp):
                data.append(
                    {
                        "SampleID": sample_id,
                        "Temperature": temp,
                        "dCp": result.data.dcp[i],
                    }
                )

        df = pl.DataFrame(data)
        df.write_parquet(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Use .csv or .parquet")


def combine_results(
    results: Dict[str, InterpolatedResult],
    grid_temp: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """
    Combine multiple processed thermograms into a single DataFrame.

    Args:
        results: Dictionary mapping sample IDs to processed results
        grid_temp: Common temperature grid (if None, use the grid from first result)

    Returns:
        DataFrame containing all samples in long format
    """
    if not results:
        return pl.DataFrame()

    # Use the grid from the first result if not provided
    if grid_temp is None:
        grid_temp = next(iter(results.values())).grid_temp

    # Combine data
    data = []

    for sample_id, result in results.items():
        for i, temp in enumerate(grid_temp):
            data.append(
                {"SampleID": sample_id, "Temperature": temp, "dCp": result.data.dcp[i]}
            )

    return pl.DataFrame(data)
