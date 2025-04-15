# ThermogramForge

<!-- Add Shields/Badges here later: e.g., build status, license -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![Documentation Status](https://readthedocs.org/projects/thermogramforge/badge/?version=latest)](https://thermogramforge.readthedocs.io/en/latest/?badge=latest) -->

**ThermogramForge** is a Dash web application for interactive analysis and visualization of Differential Scanning Fluorimetry (DSF) thermogram data.

It allows researchers to upload raw thermogram data, review automatically determined baseline endpoints, manually adjust endpoints, exclude problematic samples, and generate reports with calculated metrics (e.g., Tm, Onset).

<!-- Add Screenshot/GIF here later -->

## Key Features

* Interactive baseline endpoint selection and adjustment.
* Visualization of raw and baseline-subtracted thermograms.
* AG Grid for efficient sample overview and editing.
* Calculation of common DSF metrics (Tm, Onset, etc.).
* Report generation in CSV or Excel format.
* Support for multi-sample files (CSV/Excel).

## Quick Start

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Naalu/ThermogramForge.git
    cd ThermogramForge
    ```

2. **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**

    ```bash
    python main.py
    ```

5. Open your web browser and navigate to `http://127.0.0.1:8050/`.

## Documentation

or detailed usage instructions, contribution guidelines, and API reference, please see the **[Full Documentation](docs/_build/html/index.html)** 

## Contributing

Contributions are welcome! Please see the [Contribution Guidelines](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
