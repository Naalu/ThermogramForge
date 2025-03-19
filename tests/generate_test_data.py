#!/usr/bin/env python3
"""
Generate a set of test data files.
"""

from tests.data_generators import (
    create_basic_thermogram,
    create_edge_case_thermogram,
    save_test_data,
)


def main():
    """Generate and save test data files."""
    # Create and save basic thermogram
    basic = create_basic_thermogram(random_seed=42)
    save_test_data(basic, "basic_thermogram.csv")

    # Create and save edge case thermograms
    for case in ["sparse", "noisy", "flat", "single_peak"]:
        data = create_edge_case_thermogram(case, random_seed=42)
        save_test_data(data, f"{case}_thermogram.csv")

    print("Test data generation complete")


if __name__ == "__main__":
    main()
