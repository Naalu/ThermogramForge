.. _contributing_project_structure:

Project Structure
=================

This document provides an overview of the ThermogramForge project directory structure. Understanding the layout helps in locating relevant code and contributing effectively.

Main Directories
----------------

*   **/app/**: Contains all the code related to the Dash web application frontend and its interactions.
    *   ``app/assets/``: Static files (CSS, JavaScript) served directly by Dash to the browser (e.g., ``ag_grid_renderers.js``, ``custom.css``).
    *   ``app/callbacks/``: Python modules defining the application's callbacks, which handle user interactions and update the UI. Organized by feature area (e.g., ``upload_callbacks.py``, ``visualization_callbacks.py``).
    *   ``app/components/``: Reusable custom Dash components built using ``dash-bootstrap-components`` or standard ``dcc``/``html`` components (e.g., ``upload_processed_modal.py``).
    *   ``app/layouts/``: Defines the structure and layout of the Dash application UI. ``main_layout.py`` contains the primary layout definition using ``dash-bootstrap-components``.
    *   ``app/utils/``: Utility functions specifically supporting the web application, such as data processing for display grids or generating Plotly figures for visualization (``visualization.py``, ``data_processing.py``).
    *   ``app.py``: Initializes the core Dash app instance (``app = Dash(...)``).
    *   ``__init__.py``: Makes the ``app`` directory a Python package and may contain app-level configurations or initializations.

*   **/core/**: Contains the backend logic for thermogram analysis, designed to be independent of the web interface.
    *   ``core/baseline/``: Modules related to baseline detection and subtraction algorithms (e.g., ``simple.py``, ``advanced.py``).
    *   ``core/metrics/``: Modules for calculating various scientific metrics from thermogram data (``metric_calculation.py``).
    *   ``core/peaks/``: Modules related to peak detection algorithms (``peak_detection.py``).
    *   ``__init__.py``: Makes the ``core`` directory a Python package.

*   **/docs/**: Contains the source files for generating this documentation using Sphinx.
    *   ``conf.py``: Sphinx configuration file (themes, extensions, etc.).
    *   ``index.rst``: The main landing page (toctree) for the documentation.
    *   ``api/``, ``contributing/``, ``user_guide/``: Subdirectories containing the ``.rst`` source files for different sections of the documentation.
    *   ``_static/``: Static files (CSS, images) used specifically by the documentation theme.
    *   ``_build/``: (Typically gitignored) Default output directory where Sphinx generates the documentation (e.g., HTML files).

*   **/tests/**: *(Directory currently missing)* Intended location for automated tests (unit, integration). The ``pyproject.toml`` file is configured to look for tests in this directory using ``pytest``.

Configuration Files
-------------------

*   ``pyproject.toml``: The primary configuration file for the project. Defines:
    *   Project metadata (name, version, author, license).
    *   Runtime dependencies (``[project.dependencies]``).
    *   Optional dependencies for development, documentation, and production (``[project.optional-dependencies]``).
    *   Build system configuration (``[build-system]``).
    *   Tool configurations (e.g., ``ruff`` for linting, ``black`` for formatting, ``pytest`` for testing).
*   ``.gitignore``: Specifies intentionally untracked files that Git should ignore (e.g., ``venv-v1/``, ``__pycache__/``, ``_build/``).
*   ``main.py``: The main executable script to launch the Dash development server. Imports necessary components from ``app`` and ``core``.
*   ``check_app.py``: A utility script to perform basic checks on the application's layout, such as finding duplicate component IDs.
*   ``setup_dev.sh``: A convenience shell script to automate the setup of the development environment (creates venv, installs dependencies with dev extras).
*   ``README.md``, ``LICENSE``, ``CONTRIBUTING.md``, ``CODE_OF_CONDUCT.md``: Standard project documentation files. 