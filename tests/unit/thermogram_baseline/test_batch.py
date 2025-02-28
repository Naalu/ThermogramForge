"""Tests for the batch processing module."""

import tempfile
from pathlib import Path

import numpy as np

from packages.thermogram_baseline.thermogram_baseline.batch import (
    combine_results,
    process_multiple,
)
from packages.thermogram_baseline.thermogram_baseline.types import (
    BatchProcessingResult,
    ThermogramData,
)
from tests.test_data_utils import generate_simple_thermogram, thermogram_to_dataframe


def test_process_multiple_with_dict_input():
    """Test batch processing with dictionary input."""
    # Generate a few test thermograms
    thermograms = {
        "sample1": generate_simple_thermogram(peak_center=65.0),
        "sample2": generate_simple_thermogram(peak_center=70.0),
        "sample3": generate_simple_thermogram(peak_center=75.0),
    }

    # Process the thermograms
    result = process_multiple(
        thermograms,
        window_size=50,
        exclusion_lower=60.0,
        exclusion_upper=80.0,
        verbose=True,
        max_workers=1,  # Use sequential processing for testing
    )

    # Check result structure
    assert isinstance(result, BatchProcessingResult)
    assert len(result.results) == 3
    assert "sample1" in result.results
    assert "sample2" in result.results
    assert "sample3" in result.results

    # Check processing stats
    assert result.processing_stats["total_samples"] == 3
    assert result.processing_stats["successful"] == 3
    assert result.processing_stats["failed"] == 0

    # Check grid temperature
    assert len(result.grid_temp) == 451  # Default grid


def test_process_multiple_with_dataframe_dict_input():
    """Test batch processing with dictionary of DataFrames input."""
    # Generate test thermograms as DataFrames
    thermograms = {
        "sample1": thermogram_to_dataframe(
            generate_simple_thermogram(peak_center=65.0)
        ),
        "sample2": thermogram_to_dataframe(
            generate_simple_thermogram(peak_center=70.0)
        ),
    }

    # Process the thermograms
    result = process_multiple(
        thermograms,
        window_size=50,
        exclusion_lower=60.0,
        exclusion_upper=80.0,
        verbose=False,
        max_workers=1,
    )

    # Check result structure
    assert len(result.results) == 2
    assert "sample1" in result.results
    assert "sample2" in result.results


def test_process_multiple_with_custom_grid():
    """Test batch processing with a custom temperature grid."""
    # Generate test thermograms
    thermograms = {
        "sample1": generate_simple_thermogram(),
        "sample2": generate_simple_thermogram(),
    }

    # Define custom grid
    custom_grid = np.linspace(50, 85, 100)

    # Process the thermograms
    result = process_multiple(
        thermograms, grid_temp=custom_grid, verbose=False, max_workers=1
    )

    # Check grid properties
    assert len(result.grid_temp) == 100
    assert result.grid_temp[0] == 50.0
    assert np.isclose(result.grid_temp[-1], 85.0)

    # Check that all results use the custom grid
    for sample_id, sample_result in result.results.items():
        assert len(sample_result.data.temperature) == 100
        assert sample_result.data.temperature[0] == 50.0
        assert np.isclose(sample_result.data.temperature[-1], 85.0)


def test_process_multiple_with_output_file():
    """Test batch processing with output file."""
    # Generate test thermograms
    thermograms = {
        "sample1": generate_simple_thermogram(),
        "sample2": generate_simple_thermogram(),
    }

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        output_path = tmp.name

        # Process the thermograms
        result = process_multiple(
            thermograms, verbose=False, max_workers=1, output_file=output_path
        )

        # Check that the file was created
        assert Path(output_path).exists()

        # Check file content (just basic check)
        with open(output_path, "r") as f:
            content = f.read()
            assert "Temperature" in content
            assert "sample1" in content
            assert "sample2" in content

        # Check result structure
        assert isinstance(result, BatchProcessingResult)


def test_combine_results():
    """Test combining multiple results into a DataFrame."""
    # Generate test thermograms
    thermograms = {
        "sample1": generate_simple_thermogram(),
        "sample2": generate_simple_thermogram(),
    }

    # Process the thermograms
    result = process_multiple(thermograms, verbose=False, max_workers=1)

    # Combine results
    combined_df = combine_results(result.results)

    # Check DataFrame structure
    assert "SampleID" in combined_df.columns
    assert "Temperature" in combined_df.columns
    assert "dCp" in combined_df.columns

    # Check that all samples are included
    sample_ids = combined_df["SampleID"].unique().to_list()
    assert "sample1" in sample_ids
    assert "sample2" in sample_ids

    # Check row count
    expected_rows = len(result.grid_temp) * len(thermograms)
    assert len(combined_df) == expected_rows


def test_process_multiple_with_errors():
    """Test batch processing with some errors."""
    # Generate valid thermograms
    valid_thermograms = {
        "sample1": generate_simple_thermogram(),
        "sample2": generate_simple_thermogram(),
    }

    # Create an invalid thermogram
    invalid_thermogram = ThermogramData(
        temperature=np.array([]),  # Empty array will cause an error
        dcp=np.array([]),
    )

    # Add to dictionary
    thermograms = {**valid_thermograms, "invalid": invalid_thermogram}

    # Process the thermograms
    result = process_multiple(thermograms, verbose=False, max_workers=1)

    # Check processing stats
    assert result.processing_stats["total_samples"] == 3
    assert result.processing_stats["successful"] == 2
    assert result.processing_stats["failed"] == 1
    assert "invalid" in result.processing_stats["errors"]

    # Check that valid samples were processed
    assert "sample1" in result.results
    assert "sample2" in result.results
    assert "invalid" not in result.results
