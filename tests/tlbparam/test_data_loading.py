"""
Tests for the thermogram data loading functionality.

This module contains tests for the functions that load and convert thermogram data
in various formats, verifying their correctness and error handling.
"""

import importlib.util
import os
import tempfile

import numpy as np
import polars as pl
import pytest

from tlbparam.utils import (
    convert_long_to_wide,
    convert_wide_to_long,
    load_raw_thermograms,
    load_thermograms,
)


def create_test_wide_data() -> pl.DataFrame:
    """Create test data in wide format for testing."""
    # Create sample data with SampleCode and temperature columns
    data = {
        "SampleCode": ["Control 1", "Control 2", "Control 3"],
    }

    # Add temperature columns
    temps = np.arange(45.0, 47.1, 0.5)
    for temp in temps:
        temp_col = f"T{temp:.1f}"
        # Create synthetic dCp values
        data[temp_col] = [
            0.1 * np.sin(temp / 10) + 0.02 * i for i, _ in enumerate(data["SampleCode"])
        ]

    return pl.DataFrame(data)


def create_test_raw_data() -> str:
    """Create test raw data content (CSV format)."""
    # Create CSV content with paired columns
    csv_content = "T41a,41a,T41b,41b\n"

    # Add some rows with synthetic data
    for i in range(10):
        temp_a = 20 + i * 0.2
        value_a = np.sin(i / 5) * 10
        temp_b = 20 + i * 0.15
        value_b = np.cos(i / 4) * 8
        csv_content += f"{temp_a:.5f},{value_a:.5f},{temp_b:.5f},{value_b:.5f}\n"

    return csv_content


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    data = create_test_wide_data()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        data.write_csv(temp_file.name)
        file_path = temp_file.name
    yield file_path, data
    if os.path.exists(file_path):
        os.remove(file_path)


def test_load_thermograms_csv(temp_csv_file):
    """Test loading processed thermogram data from CSV file."""
    file_path, data = temp_csv_file

    # Test loading the CSV file
    temps = [f"T{temp:.1f}" for temp in np.arange(45.0, 47.1, 0.5)]
    result = load_thermograms(file_path, temps=temps)

    # Check the result
    assert isinstance(result, pl.DataFrame), "Result should be a polars DataFrame"

    # Check that we have the right columns
    assert "SampleCode" in result.columns, "Result should have a SampleCode column"
    for temp in temps:
        assert temp in result.columns, f"Result should have column {temp}"

    # Check that we have the right number of rows (samples)
    assert (
        result.height == 3
    ), f"Result should have 3 rows (samples), got {result.height}"

    # Check that the data values are numeric
    for temp in temps:
        assert (
            result[temp].dtype == pl.Float64
        ), f"Column {temp} should be Float64, but is {result[temp].dtype}"


def test_load_thermograms_excel():
    """Test loading processed thermogram data from Excel file.

    Note: This test requires the 'openpyxl' package to be installed.
    """

    # Skip if openpyxl is not available
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl package not available")

    # Create test data
    data = create_test_wide_data()

    # Create a temporary Excel file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        data.write_excel(temp_file.name)

    try:
        # Test loading the Excel file
        temps = [f"T{temp:.1f}" for temp in np.arange(45.0, 47.1, 0.5)]
        result = load_thermograms(temp_file.name, temps=temps, blank_row=0)

        # Check the result
        assert isinstance(result, pl.DataFrame), "Result should be a polars DataFrame"

        # Check that we have the right columns
        assert "SampleCode" in result.columns, "Result should have a SampleCode column"
        for temp in temps:
            assert temp in result.columns, f"Result should have column {temp}"

        # Check that we have the right number of rows (samples)
        assert result.height == 3, "Result should have 3 rows (samples)"

    finally:
        # Remove the temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def test_load_raw_thermograms():
    """Test loading raw thermogram data from paired columns."""
    # Create CSV content with paired columns
    csv_content = create_test_raw_data()

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        temp_file.write(csv_content.encode("utf-8"))

    try:
        # Test loading the raw CSV file
        result = load_raw_thermograms(temp_file.name)

        # Check the result
        assert isinstance(result, pl.DataFrame), "Result should be a polars DataFrame"

        # Check that we have the right columns
        expected_columns = ["SampleID", "Temperature", "dCp"]
        assert all(
            col in result.columns for col in expected_columns
        ), f"Result should have columns {expected_columns}"

        # Check that we have rows for both samples
        sample_ids = result.select("SampleID").unique().to_series().to_list()
        assert "41a" in sample_ids, "Result should contain sample 41a"
        assert "41b" in sample_ids, "Result should contain sample 41b"

        # Check that we have the right data types
        assert result["SampleID"].dtype == pl.Utf8, "SampleID should be string"
        assert (
            result["Temperature"].dtype == pl.Float64
        ), "Temperature should be Float64"
        assert result["dCp"].dtype == pl.Float64, "dCp should be Float64"

        # Check that the data is sorted
        sorted_df = result.sort(by=["SampleID", "Temperature"])
        assert result.equals(
            sorted_df
        ), "Result should be sorted by SampleID and Temperature"

    finally:
        # Remove the temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def test_load_raw_thermograms_large():
    """Test loading large raw thermogram data efficiently."""
    # Create a larger dataset with more samples and points
    # This test is designed to verify that our implementation can handle large files
    csv_content = "T41a,41a,T41b,41b,T41c,41c,T41d,41d\n"

    # Add more rows (1000 points per sample)
    for i in range(1000):
        line_parts = []
        for j in range(4):  # 4 samples
            temp = 20 + i * 0.1 + j * 0.01
            value = np.sin(i / (5 + j)) * 10
            line_parts.extend([f"{temp:.5f}", f"{value:.5f}"])
        csv_content += ",".join(line_parts) + "\n"

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        temp_file.write(csv_content.encode("utf-8"))

    try:
        # Load the data - this should handle the large file efficiently
        result = load_raw_thermograms(temp_file.name)

        # Basic validation
        assert isinstance(result, pl.DataFrame), "Result should be a polars DataFrame"
        assert result.height > 0, "Result should have rows"

        # Check we have all 4 samples
        sample_ids = result.select("SampleID").unique().to_series().to_list()
        assert len(sample_ids) == 4, "Should have 4 unique sample IDs"

        # Verify row count is close to expected (allowing for some NaN filtering)
        # We expect approximately 1000 points per sample
        assert result.height >= 3800, "Should have approximately 4000 data points"

    finally:
        # Remove the temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def test_convert_long_to_wide():
    """Test converting from long to wide format."""
    # Create test data in long format
    long_data = {
        "SampleID": ["Sample1", "Sample1", "Sample1", "Sample2", "Sample2", "Sample2"],
        "Temperature": [45.0, 46.0, 47.0, 45.0, 46.0, 47.0],
        "dCp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }
    df_long = pl.DataFrame(long_data)

    # Convert to wide format
    df_wide = convert_long_to_wide(df_long)

    # Check the result
    assert isinstance(df_wide, pl.DataFrame), "Result should be a polars DataFrame"

    # Check that we have the right columns
    expected_columns = ["SampleID", "T45.0", "T46.0", "T47.0"]
    assert all(
        col in df_wide.columns for col in expected_columns
    ), f"Result should have columns {expected_columns}"

    # Check that we have the right number of rows
    assert df_wide.height == 2, "Result should have 2 rows (samples)"

    # Check values
    sample1_row = df_wide.filter(pl.col("SampleID") == "Sample1").to_dicts()[0]
    assert sample1_row["T45.0"] == 0.1, "Sample1 should have T45.0 = 0.1"
    assert sample1_row["T46.0"] == 0.2, "Sample1 should have T46.0 = 0.2"
    assert sample1_row["T47.0"] == 0.3, "Sample1 should have T47.0 = 0.3"

    sample2_row = df_wide.filter(pl.col("SampleID") == "Sample2").to_dicts()[0]
    assert sample2_row["T45.0"] == 0.4, "Sample2 should have T45.0 = 0.4"
    assert sample2_row["T46.0"] == 0.5, "Sample2 should have T46.0 = 0.5"
    assert sample2_row["T47.0"] == 0.6, "Sample2 should have T47.0 = 0.6"


def test_convert_long_to_wide_with_grid():
    """Test converting from long to wide format with specified temperature grid."""
    # Create test data in long format with irregular temperature points
    long_data = {
        "SampleID": [
            "Sample1",
            "Sample1",
            "Sample1",
            "Sample1",
            "Sample2",
            "Sample2",
            "Sample2",
            "Sample2",
        ],
        "Temperature": [
            45.1,
            46.3,
            47.2,
            48.7,  # Irregular temperatures for Sample1
            45.3,
            46.7,
            47.5,
            48.2,  # Irregular temperatures for Sample2
        ],
        "dCp": [
            0.11,
            0.23,
            0.32,
            0.47,  # Values for Sample1
            0.13,
            0.27,
            0.35,
            0.42,  # Values for Sample2
        ],
    }
    df_long = pl.DataFrame(long_data)

    # Define a regular temperature grid
    temp_grid = np.array([45.0, 46.0, 47.0, 48.0, 49.0])

    # Convert to wide format with specified grid
    df_wide = convert_long_to_wide(df_long, temp_grid=temp_grid)

    # Check the result
    assert isinstance(df_wide, pl.DataFrame), "Result should be a polars DataFrame"

    # Check that we have columns for all temperatures in the grid
    expected_columns = ["SampleID"] + [f"T{temp:.1f}" for temp in temp_grid]
    assert all(
        col in df_wide.columns for col in expected_columns
    ), f"Result should have columns {expected_columns}"

    # Check that we have both samples
    assert df_wide.height == 2, "Result should have 2 rows (samples)"

    # Verify some mapped values - they should be mapped to closest grid points
    sample1 = df_wide.filter(pl.col("SampleID") == "Sample1").to_dicts()[0]
    sample2 = df_wide.filter(pl.col("SampleID") == "Sample2").to_dicts()[0]

    # Sample1's 45.1 should map to T45.0
    assert sample1["T45.0"] == 0.11, "Sample1 value at 45.1 should map to T45.0"
    # Sample1's 46.3 should map to T46.0
    assert sample1["T46.0"] == 0.23, "Sample1 value at 46.3 should map to T46.0"

    # Sample2's 45.3 should map to T45.0
    assert sample2["T45.0"] == 0.13, "Sample2 value at 45.3 should map to T45.0"
    # Sample2's 48.2 should map to T48.0
    assert sample2["T48.0"] == 0.42, "Sample2 value at 48.2 should map to T48.0"


def test_convert_wide_to_long():
    """Test converting from wide to long format."""
    # Create test data in wide format
    wide_data = {
        "SampleCode": ["Sample1", "Sample2"],
        "T45.0": [0.1, 0.4],
        "T46.0": [0.2, 0.5],
        "T47.0": [0.3, 0.6],
    }
    df_wide = pl.DataFrame(wide_data)

    # Convert to long format
    df_long = convert_wide_to_long(df_wide)

    # Check the result
    assert isinstance(df_long, pl.DataFrame), "Result should be a polars DataFrame"

    # Check that we have the right columns
    expected_columns = ["SampleID", "Temperature", "dCp"]
    assert all(
        col in df_long.columns for col in expected_columns
    ), f"Result should have columns {expected_columns}"

    # Check that we have all data points
    assert df_long.height == 6, "Result should have 6 rows (2 samples × 3 temperatures)"

    # Check that the data is sorted
    sorted_df = df_long.sort(by=["SampleID", "Temperature"])
    assert df_long.equals(
        sorted_df
    ), "Result should be sorted by SampleID and Temperature"

    # Check values for one sample
    sample1_points = (
        df_long.filter(pl.col("SampleID") == "Sample1")
        .sort(by="Temperature")
        .to_dicts()
    )
    assert len(sample1_points) == 3, "Sample1 should have 3 data points"
    assert sample1_points[0]["Temperature"] == 45.0 and sample1_points[0]["dCp"] == 0.1
    assert sample1_points[1]["Temperature"] == 46.0 and sample1_points[1]["dCp"] == 0.2
    assert sample1_points[2]["Temperature"] == 47.0 and sample1_points[2]["dCp"] == 0.3


def test_error_handling():
    """Test error handling in data loading functions."""
    # Test file not found
    with pytest.raises(FileNotFoundError):
        # Attempting to load a nonexistent file should raise a FileNotFoundError
        load_thermograms("nonexistent_file.csv")

    with pytest.raises(FileNotFoundError):
        load_raw_thermograms("nonexistent_file.csv")

    # Test invalid file format
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"This is not a CSV or Excel file")

    try:
        with pytest.raises(ValueError):
            load_thermograms(temp_file.name)

        with pytest.raises(ValueError):
            load_raw_thermograms(temp_file.name)

    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    # Test invalid data format for conversion
    # Missing required columns
    with pytest.raises(ValueError):
        convert_long_to_wide(pl.DataFrame({"WrongColumn": [1, 2, 3]}))

    with pytest.raises(ValueError):
        convert_wide_to_long(pl.DataFrame({"WrongColumn": [1, 2, 3]}))


def main():
    """Run all tests with proper setup and detailed output for debugging."""
    import traceback
    from contextlib import contextmanager

    @contextmanager
    def test_case(name):
        """Context manager to handle test case execution with better error reporting."""
        print(f"\n{'=' * 80}\nRunning test: {name}\n{'-' * 80}")
        try:
            yield
            print(f"\n✅ {name} PASSED")
        except Exception as e:
            print(f"\n❌ {name} FAILED: {type(e).__name__}: {e}")
            print("\nTraceback:")
            traceback.print_exc()
            print("\n")

    def setup_csv_fixture():
        """Create and return a temporary CSV fixture."""
        data = create_test_wide_data()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            data.write_csv(temp_file.name)
        print(f"Created temporary CSV file: {temp_file.name}")
        print(f"Data preview:\n{data.head(3)}")
        return temp_file.name, data

    # Track files to clean up
    temp_files = []

    try:
        # Run tests with CSV fixture
        with test_case("test_load_thermograms_csv"):
            csv_path, csv_data = setup_csv_fixture()
            temp_files.append(csv_path)
            test_load_thermograms_csv((csv_path, csv_data))

        # Run test with Excel fixture if possible
        with test_case("test_load_thermograms_excel"):
            if importlib.util.find_spec("openpyxl") is not None:
                data = create_test_wide_data()
                with tempfile.NamedTemporaryFile(
                    suffix=".xlsx", delete=False
                ) as temp_file:
                    data.write_excel(temp_file.name)
                    temp_files.append(temp_file.name)
                    print(f"Created temporary Excel file: {temp_file.name}")
                test_load_thermograms_excel()
            else:
                print("Skipping test_load_thermograms_excel: openpyxl not available")

        # Run raw thermogram test
        with test_case("test_load_raw_thermograms"):
            csv_content = create_test_raw_data()
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                temp_file.write(csv_content.encode("utf-8"))
                temp_files.append(temp_file.name)
                print(f"Created temporary raw thermogram file: {temp_file.name}")
            test_load_raw_thermograms()

        # Run large raw thermogram test with more verbose output
        with test_case("test_load_raw_thermograms_large"):
            print("Creating large test dataset (this might take a moment)...")
            csv_content = "T41a,41a,T41b,41b,T41c,41c,T41d,41d\n"
            # Add sample rows to the content
            for i in range(5):  # Just add 5 rows for the preview
                line_parts = []
                for j in range(4):
                    temp = 20 + i * 0.1 + j * 0.01
                    value = np.sin(i / (5 + j)) * 10
                    line_parts.extend([f"{temp:.5f}", f"{value:.5f}"])
                csv_content += ",".join(line_parts) + "\n"
            print(f"Sample content:\n{csv_content}")

            # Now create the actual test file
            print("Writing full test file with 1000 data points per sample...")
            csv_content = "T41a,41a,T41b,41b,T41c,41c,T41d,41d\n"
            for i in range(1000):
                line_parts = []
                for j in range(4):
                    temp = 20 + i * 0.1 + j * 0.01
                    value = np.sin(i / (5 + j)) * 10
                    line_parts.extend([f"{temp:.5f}", f"{value:.5f}"])
                csv_content += ",".join(line_parts) + "\n"

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                temp_file.write(csv_content.encode("utf-8"))
                temp_files.append(temp_file.name)
                print(f"Created temporary large raw thermogram file: {temp_file.name}")
            test_load_raw_thermograms_large()

        # Run conversion tests
        with test_case("test_convert_long_to_wide"):
            test_convert_long_to_wide()

        with test_case("test_convert_long_to_wide_with_grid"):
            test_convert_long_to_wide_with_grid()

        with test_case("test_convert_wide_to_long"):
            test_convert_wide_to_long()

        # Run error handling tests
        with test_case("test_error_handling"):
            test_error_handling()

        print("\n" + "=" * 80)
        print("All tests completed!")

    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")


if __name__ == "__main__":
    main()
