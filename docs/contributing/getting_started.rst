.. _contributing_getting_started:

Getting Started
===============

Interested in contributing to ThermogramForge? Great! This guide will help you set up your development environment.

Prerequisites
-------------

*   You should have already followed the user :doc:`../user_guide/installation` guide to clone the repository and ensure you have Python (>=3.8) and Git installed.
*   Familiarity with `git` for version control is essential.

Development Setup
-----------------

Instead of just installing the base requirements, the development setup installs the package in editable mode along with additional development dependencies (like testing tools and linters) specified in ``pyproject.toml``.

1.  **Navigate to Project Root:**
    Ensure you are in the main ``ThermogramForge`` directory.

2.  **Run Development Setup Script:**
    The easiest way to set up the development environment is to run the provided script. This will create the virtual environment (if it doesn't exist), activate it, and install all necessary dependencies.

    .. code-block:: bash

        ./setup_dev.sh

    *(Note: You might need to make the script executable first: ``chmod +x setup_dev.sh``)*

    This script performs the following steps:
    *   Creates a virtual environment named ``venv-v1`` if it's not already present.
    *   Activates the ``venv-v1`` environment.
    *   Installs the package in editable mode (``-e``) and includes the development dependencies (``[dev]``).

3.  **Manual Setup (Alternative):**
    If you prefer not to use the script, you can perform the steps manually:

    .. code-block:: bash

        # Create/activate virtual environment (if not done)
        python -m venv venv-v1
        source venv-v1/bin/activate  # Or venv-v1\Scripts\activate on Windows

        # Install in editable mode with dev extras
        pip install -e ".[dev]"

4.  **Set Up Pre-commit Hooks (Optional but Recommended):**
    *(Note: Check ``pyproject.toml`` for pre-commit configuration details if you want to enable this).*
    While not automatically installed by ``setup_dev.sh``, pre-commit hooks might be configured. If you wish to use them:

    .. code-block:: bash

        # Ensure pre-commit is installed (it may be in [dev] extras)
        # pip install pre-commit
        pre-commit install

Next Steps
----------

Once your development environment is set up, you should familiarize yourself with the:

*   :doc:`project_structure`
*   :doc:`coding_conventions`
*   :doc:`workflow` for contributing changes.
*   :doc:`testing` procedures.
*   How to build the documentation (:doc:`building_docs`). 