# ThermogramForge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Made with: Dash](https://img.shields.io/badge/Made%20with-Dash-success.svg)](https://dash.plotly.com/)

**ThermogramForge** is a comprehensive web application for the analysis and visualization of Thermal Liquid Biopsy (TLB) thermogram data. Born out of the need to make thermogram analysis more accessible and efficient, this project reimagines and reinvents traditional R-based analytical tools into an intuitive Python ecosystem with a visual interface, enabling researchers to process, visualize, and extract meaningful insights from TLB profiles without specialized programming knowledge.

![ThermogramForge Interface](app/assets/images/ThermogramForge-Interface.png)

## Why ThermogramForge?

Thermal Liquid Biopsy (TLB) captures calorimetric signatures of blood plasma proteomes, providing valuable diagnostic information. However, traditional analysis methods are often slow, cumbersome, and require specialized expertise. ThermogramForge addresses these challenges by:

- Reimagining the analytical framework with a focus on user experience
- Translating complex R-based algorithms into an intuitive Python ecosystem
- Significantly reducing analysis time while maintaining methodological rigor
- Democratizing access to TLB analysis for a broader range of researchers and clinicians

## Key Features

- **Intelligent Baseline Analysis**
  - Automatic endpoint detection using rolling variance on spline-smoothed data
  - Interactive visual adjustment of baseline endpoints
  - Multiple endpoint selection methods (innermost, outermost, middle)
  - Real-time visualization of baseline subtraction effects

- **Powerful Data Processing**
  - Multi-sample batch processing with parallel computation
  - Temperature range filtering and normalization options
  - Interpolation to uniform temperature grids
  - Automated peak and valley detection algorithms

- **Comprehensive Metric Calculation**
  - Peak heights and temperatures for multiple regions (TPeak_F, TPeak_1, TPeak_2, TPeak_3)
  - Peak ratios and valley metrics
  - Global thermogram characteristics (Area, TFM, FWHM)
  - Customizable metric selection for reporting

- **Flexible Data Import & Export**
  - Support for various CSV and Excel file formats (single or multi-sample)
  - Column pair detection for multi-sample files (T[ID]/ID pattern)
  - Report generation in CSV or Excel formats
  - Exportable processed data for downstream analysis

- **Modern, Responsive Interface**
  - Interactive Plotly visualizations
  - AG Grid for efficient sample management and review
  - Clear workflow-oriented tab design
  - Bootstrap-styled responsive layout

## Prerequisites

- **Python 3.10+** (Required for type hinting features used in the codebase)
- **Modern web browser** (Chrome, Firefox, Edge, or Safari)
- **Basic understanding of thermogram data** (terminology, expected curve shapes)
- **~50MB disk space** for installation and dependencies
- **Git** (for cloning the repository)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/Naalu/ThermogramForge.git
cd ThermogramForge

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the package and dependencies
pip install .

# Run the application
python main.py
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/Naalu/ThermogramForge.git
cd ThermogramForge

# Run the development setup script (creates venv-v1 environment)
./setup_dev.sh  # On Windows, you may need to set up manually

# Or manually install with development dependencies
python -m venv venv-dev
source venv-dev/bin/activate  # On Windows use: venv-dev\Scripts\activate
pip install -e ".[dev]"
```

## Detailed Workflow

1. **Upload Raw Data**
   - Navigate to the "Data Overview" tab
   - Click "Upload New Raw Thermogram Data"
   - Select a CSV or Excel file containing thermogram data
   - Configure temperature range and advanced parameters
   - Submit to process the raw data

2. **Review Baseline Endpoints**
   - Navigate to the "Review Endpoints" tab
   - Select your uploaded dataset from the dropdown
   - Review samples in the grid, clicking each to examine
   - Evaluate automatic baseline endpoints in the plot area
   - Use "Manually Adjust" buttons to refine endpoints if needed
   - Mark samples as "Reviewed" (or "Exclude" if problematic)
   - Click "Save Processed Data" when finished with review

3. **Generate Reports**
   - Navigate to the "Report Builder" tab
   - Select a processed dataset from the dropdown
   - Configure report name and format (CSV or Excel)
   - Select desired metrics using the checkboxes
   - Preview the report data in the table
   - Click "Generate Report" to create and download the file

## Project Structure

```
thermogram_project/
├── app/                    # Web application code
│   ├── assets/             # Static assets (CSS, JS)
│   ├── callbacks/          # Dash callback definitions 
│   ├── components/         # Reusable UI components
│   ├── layouts/            # Page layouts and structure
│   └── utils/              # App utility functions
│
├── core/                   # Core data processing logic
│   ├── baseline/           # Baseline detection and subtraction
│   ├── metrics/            # Metric calculation algorithms
│   └── peaks/              # Peak detection algorithms
│
├── docs/                   # Documentation
│   ├── api/                # API reference
│   ├── contributing/       # Contributor guides
│   └── user_guide/         # User manual
│
├── tests/                  # Test suite (planned)
├── main.py                 # Application entry point
└── pyproject.toml          # Project configuration
```

## Use Cases

ThermogramForge is particularly valuable for:

- **Clinical Researchers** analyzing patient thermogram profiles for biomarker discovery
- **Pharmaceutical Scientists** evaluating protein stability and interactions
- **Laboratory Technicians** processing large batches of thermogram data
- **Academic Researchers** exploring thermal properties of biological samples
- **Quality Control** in biotechnology and diagnostics

## Documentation

For comprehensive documentation:

1. **Build locally:**
   ```bash
   pip install -e ".[docs]"
   cd docs
   sphinx-build -b html . _build/html
   ```
2. Open `docs/_build/html/index.html` in your browser

3. **Key sections:**
   - User Guide: Installation, interface overview, data formats
   - API Reference: Details on core functions and classes
   - Contributor Guide: Development setup, coding standards

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development environment setup
- Coding conventions and standards
- Pull request workflow
- Testing guidelines

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dr. Robert Buscaglia for project mentorship
- Northern Arizona University for research support
- The Python scientific computing community for the robust ecosystem that makes this project possible
