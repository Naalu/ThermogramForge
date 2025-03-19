Test Data
=========

ThermogramForge includes a set of test data files for development, testing, and validation purposes.

Basic Test Data
--------------

These files are generated using the scripts in the ``tests`` directory:

**Basic Thermogram Data**

- **basic_thermogram.csv**: Standard thermogram with multiple peaks and moderate noise
  
  - Contains three peaks at approximately 63°C, 70°C, and 77°C
  - Has a linear baseline
  - Includes moderate Gaussian noise (0.01 standard deviation)

**Edge Case Thermograms**

- **sparse_thermogram.csv**: Thermogram with sparse data points
  
  - Contains only 20% of the normal number of points
  - Still maintains a recognizable peak shape

- **noisy_thermogram.csv**: Thermogram with high noise levels
  
  - Contains standard peaks but with high noise (0.2 standard deviation)
  - Tests the robustness of peak detection algorithms

- **flat_thermogram.csv**: Nearly flat thermogram (baseline only)
  
  - Contains only a linear baseline and minimal noise
  - Used to test behavior when no significant peaks are present

- **single_peak_thermogram.csv**: Thermogram with a single sharp peak
  
  - Contains one narrow peak at 65°C
  - Used to test simple peak detection

# Addition to docs/source/test_data.rst - add this section

Real Thermogram Data
-------------------

The ``tests/data/real_thermograms/`` directory contains real thermogram data from laboratory experiments:

**Raw Data**

- Original thermogram measurements from DSC instruments
- Data is in long format with Temperature, dCp, and SampleID columns
- Each file represents a single sample
- Contains unprocessed measurements direct from the instrument

**Processed Data**

- Thermogram data that has been baseline-subtracted and interpolated
- Contains data interpolated to a consistent temperature grid
- Provides a reference for validation of processing algorithms
- Combined dataset with multiple samples available for batch processing tests

This real-world data is particularly valuable for validating the software against actual laboratory conditions and ensuring it performs correctly with data from real experiments.

Thermogram Reference Data
-----------------------

The ``tests/data/thermogram_reference/`` directory contains more sophisticated thermogram 
data designed for comprehensive testing and validation:

- **standard_thermogram.csv**: Reference thermogram with well-defined peaks
  
  - Contains peaks at 55°C, 63°C, 70°C, and 77°C
  - Used as the primary reference for validation

- **shifted_peaks_thermogram.csv**: Same peaks shifted to different temperatures
  
  - Tests algorithm robustness to peak position changes

- **varied_heights_thermogram.csv**: Same peaks with different heights
  
  - Tests sensitivity to variations in peak amplitude

- **steep_baseline_thermogram.csv**: Thermogram with a steeper baseline
  
  - Tests baseline subtraction with more significant slopes

- **nonlinear_baseline_thermogram.csv**: Thermogram with quadratic baseline
  
  - Tests baseline subtraction with nonlinear baselines

- **noise_*_thermogram.csv**: Thermograms with different noise levels
  
  - Tests algorithm performance under various noise conditions

R Reference Data
--------------

For validating the spline fitting implementation against R, reference data in 
``tests/data/r_reference/`` provides direct comparisons between R's ``smooth.spline`` 
and Python implementations:

- **sine_input.csv**, **sine_fitted.csv**, **sine_params.csv**: Sine wave test pattern
  
  - Tests spline fitting with oscillatory data

- **exp_input.csv**, **exp_fitted.csv**, **exp_params.csv**: Exponential test pattern
  
  - Tests spline fitting with monotonic, nonlinear data

- **peaks_input.csv**, **peaks_fitted.csv**, **peaks_params.csv**: Multi-peak pattern
  
  - Tests spline fitting with thermogram-like data

Each input file contains the x and y values, while fitted files contain the spline fit
values from R. The params files contain the optimized smoothing parameters (spar, df, lambda)
from R's implementation.

Generating Test Data
------------------

To regenerate test data, run:

.. code-block:: bash

    # Basic test data
    python tests/generate_test_data.py
    
    # R reference data (requires R and rpy2)
    python tests/generate_r_reference.py
    
    # Thermogram reference data
    python tests/generate_thermogram_reference.py

Custom Test Data Generation
-------------------------

These scripts use utility functions that can also be imported directly for custom test data generation:

.. code-block:: python

    from tests.data_generators import create_basic_thermogram, create_edge_case_thermogram
    
    # Create custom thermogram data
    my_data = create_basic_thermogram(n_points=200, noise_level=0.05)
    
    # Create specific edge case data
    noisy_data = create_edge_case_thermogram('noisy', n_points=150)
    
    # Create realistic thermogram with custom parameters
    from tests.generate_thermogram_reference import generate_realistic_thermogram
    
    custom_thermogram = generate_realistic_thermogram(
        peak_centers=[58, 65, 72, 80],
        peak_heights=[0.15, 0.25, 0.3, 0.1],
        baseline_slope=0.03
    )

File Format
----------

All test data files are in CSV format with the following structure:

For thermogram data:

.. code-block:: text

    Temperature,dCp
    45.0,0.123
    45.5,0.128
    ...

For reference data:

.. code-block:: text

    x,y,fitted
    45.0,0.123,0.125
    45.5,0.128,0.130
    ...