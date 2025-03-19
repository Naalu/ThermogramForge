# tests/inspect_real_thermograms.py

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl


def inspect_real_thermograms(directory, output_file=None):
    """
    Inspect real thermogram data files and generate a report.

    Args:
        directory: Directory containing thermogram CSV files
        output_file: Optional file to save the report to
    """
    # Get all CSV files in the directory
    files = list(Path(directory).glob("*.csv"))

    print(f"Found {len(files)} thermogram files in {directory}")

    results = []

    for file_path in files:
        # Read the CSV file
        df = pl.read_csv(file_path)

        # Get basic statistics
        num_rows = len(df)

        # Get statistics if columns exist
        temp_stats = (
            df.select(pl.min("Temperature"), pl.max("Temperature")).row(0)
            if "Temperature" in df.columns
            else (None, None)
        )
        dcp_stats = (
            df.select(pl.min("dCp"), pl.max("dCp")).row(0)
            if "dCp" in df.columns
            else (None, None)
        )

        temp_min, temp_max = temp_stats if "Temperature" in df.columns else (None, None)
        dcp_min, dcp_max = dcp_stats if "dCp" in df.columns else (None, None)

        # Store the results
        results.append(
            {
                "File": file_path.name,
                "Rows": num_rows,
                "Temp_Min": temp_min,
                "Temp_Max": temp_max,
                "dCp_Min": dcp_min,
                "dCp_Max": dcp_max,
            }
        )

    # Create a summary DataFrame
    summary = pl.DataFrame(results)

    # Print summary
    print("\nSummary Statistics:")
    print(summary)

    # Save to file if requested
    if output_file:
        summary.write_csv(output_file)
        print(f"Summary saved to {output_file}")

    # Generate a plot of a few samples
    if len(files) > 0:
        fig = go.Figure()

        # Plot up to 5 thermograms
        for i, file_path in enumerate(files[:5]):
            df = pl.read_csv(file_path)
            if "Temperature" in df.columns and "dCp" in df.columns:
                # Convert to pandas for easy plotting with Plotly
                temp_values = df.get_column("Temperature").to_list()
                dcp_values = df.get_column("dCp").to_list()
                fig.add_trace(
                    go.Scatter(
                        x=temp_values, y=dcp_values, mode="lines", name=file_path.stem
                    )
                )

        fig.update_layout(
            title="Sample Thermograms",
            xaxis_title="Temperature (°C)",
            yaxis_title="dCp",
            grid=dict(visible=True),
        )

        plot_file = Path(directory) / "sample_plot.png"
        pio.write_image(fig, str(plot_file))
        print(f"Sample plot saved to {plot_file}")


def main():
    # Inspect raw thermograms
    raw_dir = Path("tests/data/real_thermograms/raw")
    if raw_dir.exists():
        inspect_real_thermograms(raw_dir, raw_dir / "summary.csv")

    # Inspect processed thermograms
    processed_dir = Path("tests/data/real_thermograms/processed")
    if processed_dir.exists():
        inspect_real_thermograms(processed_dir, processed_dir / "summary.csv")


if __name__ == "__main__":
    main()
