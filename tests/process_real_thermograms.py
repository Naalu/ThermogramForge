"""
Process raw and processed thermogram data into consistent format.
"""

import os
from pathlib import Path
from typing import Optional

import polars as pl


def process_raw_thermogram_csv(file_path, output_dir, max_samples=5):
    """
    Process raw thermogram CSV data into consistent format.

    Args:
        file_path: Path to the raw CSV file
        output_dir: Directory to save processed files
        max_samples: Maximum number of samples to process (for small test sets)
    """
    print(f"Processing {file_path}...")

    # Read the CSV file
    df = pl.read_csv(file_path)

    # Get column names
    columns = df.columns

    # Identify sample pairs (Temperature and Value columns)
    sample_pairs = []
    for col in columns:
        if col.startswith("T") and col[1:] in columns:
            # This is a temperature column with matching value column
            temp_col = col
            value_col = col[1:]
            sample_pairs.append((temp_col, value_col))

    print(f"Found {len(sample_pairs)} sample pairs")

    # Process only a subset if requested
    if max_samples is not None and max_samples < len(sample_pairs):
        print(f"Limiting to {max_samples} samples")
        sample_pairs = sample_pairs[:max_samples]

    # Process each sample pair
    for i, (temp_col, value_col) in enumerate(sample_pairs):
        sample_id = value_col

        # Extract the data for this sample
        sample_data = df.select([temp_col, value_col])

        # Drop rows where either temperature or value is missing
        sample_data = sample_data.drop_nulls()

        # Rename columns to standard format
        sample_data = sample_data.rename({temp_col: "Temperature", value_col: "dCp"})

        # Add sample ID column
        sample_data = sample_data.with_columns(pl.lit(sample_id).alias("SampleID"))

        # Save as CSV in long format
        output_file = os.path.join(output_dir, f"{sample_id}_raw.csv")
        sample_data.write_csv(output_file)

        print(f"  - Saved sample {sample_id} with {len(sample_data)} rows")

    print("Processing complete")


def process_processed_excel(
    file_path: str,
    output_dir: str,
    sheet_name: Optional[str] = None,
    max_samples: int = 5,
) -> None:
    """
    Process processed Excel data into consistent format.

    Args:
        file_path: Path to the processed Excel file
        output_dir: Directory to save processed files
        sheet_name: Name of the sheet to read (None for first/default sheet)
        max_samples: Maximum number of samples to process (for small test sets)
    """
    print(f"Processing {file_path}...")

    # Display sheet info if sheet name is provided
    sheet_info = f" (sheet: {sheet_name})" if sheet_name else ""
    print(f"Reading from Excel file{sheet_info}...")

    # Read the Excel file with specified sheet
    try:
        df = pl.read_excel(
            file_path,
            sheet_name=sheet_name,
            engine="calamine",  # Use Calamine for better compatibility
            has_header=False,  # Assume no header in data
        )
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        if sheet_name:
            print(f"Available sheets: {pl.read_excel(file_path, sheet_id=None)}")
        return

    # First column should be sample IDs (skip first row which contains headers)
    sample_ids = df.slice(1).get_column(df.columns[0]).to_list()

    # First row should be temperatures (skip the first cell which is a header/label)
    temperatures = df.slice(0, 1).select(pl.exclude(df.columns[0])).row(0)

    print(f"Found {len(sample_ids)} samples and {len(temperatures)} temperature points")

    # Process only a subset if requested
    if max_samples is not None and max_samples < len(sample_ids):
        print(f"Limiting to {max_samples} samples")
        sample_ids = sample_ids[:max_samples]

    # Process each sample
    combined_data = []
    for i, sample_id in enumerate(sample_ids):
        # Extract the data for this sample
        values = df.slice(i + 1, 1).select(pl.exclude(df.columns[0])).row(0)

        # Create a DataFrame with this sample's data
        sample_data = pl.DataFrame(
            {
                "Temperature": temperatures,
                "dCp": values,
                "SampleID": [sample_id] * len(temperatures),
            }
        )

        # Drop any rows with missing values
        sample_data = sample_data.drop_nulls()

        # Save as CSV in long format
        # Use sheet name in the output filename if provided
        sheet_suffix = f"_{sheet_name}" if sheet_name else ""
        output_file = os.path.join(
            output_dir, f"{sample_id}{sheet_suffix}_processed.csv"
        )
        sample_data.write_csv(output_file)

        combined_data.append(sample_data)

        print(f"  - Saved sample {sample_id} with {len(sample_data)} rows")

    # Combine all samples
    if combined_data:
        all_samples = pl.concat(combined_data)

        # Save combined dataset
        sheet_suffix = f"_{sheet_name}" if sheet_name else ""
        output_file = os.path.join(output_dir, f"combined{sheet_suffix}_processed.csv")
        all_samples.write_csv(output_file)
        print(f"  - Saved combined dataset with {len(all_samples)} rows")
    else:
        print("  - No valid data to combine")

    print("Processing complete")


def main():
    # Set up paths
    base_dir = Path("tests/data/real_thermograms")
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    # Create the directories if they don't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process raw data file
    # path to get data from (for now): data/reference/r_raw.csv
    raw_file = Path("data/reference/r_raw.csv")
    if os.path.exists(raw_file):
        process_raw_thermogram_csv(raw_file, raw_dir)

    # Process processed data file with specific sheet
    # path to get data from (for now):
    # data/reference/Pub_Processed.xlsx, sheet_name="DSC"
    processed_file = Path("data/reference/Pub_Processed.xlsx")
    if os.path.exists(processed_file):
        # Process sheet
        process_processed_excel(processed_file, processed_dir, sheet_name="DSC")
    print("All processing complete")


if __name__ == "__main__":
    main()
