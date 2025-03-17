Contributing
===========

We welcome contributions to the ThermogramForge project!

Development Setup
---------------

.. code-block:: bash

    # Create a development environment
    uv venv

    # Install development dependencies
    uv add --dev

    # Run tests
    uv run pytest

    # Check code formatting
    uv run black . --check
    uv run ruff check .
