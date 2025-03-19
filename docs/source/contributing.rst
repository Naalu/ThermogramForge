Contributing to ThermogramForge
=============================

Thank you for considering contributing to ThermogramForge! This document outlines the process for contributing to the project.

Development Setup
---------------

Prerequisites
~~~~~~~~~~~~

- Python 3.10+
- Git
- (Optional) R 4.0+ for R integration

Setup Steps
~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally

   .. code-block:: bash

       git clone https://github.com/YourUsername/ThermogramForge.git
       cd ThermogramForge

3. Create a virtual environment and install dependencies

   .. code-block:: bash

       # Using uv (recommended)
       uv venv
       uv pip install -e ".[dev]"

       # Or using standard tools
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       pip install -e ".[dev]"

4. Install pre-commit hooks

   .. code-block:: bash

       pre-commit install

Development Workflow
------------------

1. Create a branch for your feature

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. Make your changes, following the code style guidelines

3. Run tests

   .. code-block:: bash

       # Run all tests
       pytest

       # Run tests with coverage
       pytest --cov=thermogram_baseline --cov=tlbparam --cov-report=term-missing

4. Check code quality

   .. code-block:: bash

       # Run linting and type checking
       python -m ruff check .
       python -m black --check .
       python -m mypy .

       # Fix formatting issues automatically
       python -m black .

5. Commit your changes with a descriptive message

   .. code-block:: bash

       git commit -m "Add feature: your feature description"

6. Push to your fork

   .. code-block:: bash

       git push origin feature/your-feature-name

7. Create a Pull Request on GitHub

Code Style Guidelines
-------------------

This project follows these style guidelines:

- `Black <https://black.readthedocs.io/>`_ for code formatting
- `Ruff <https://beta.ruff.rs/docs/>`_ for linting
- `MyPy <https://mypy.readthedocs.io/>`_ for type checking
- `Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ for documentation

Key Style Points
~~~~~~~~~~~~~~

1. All functions, methods, and classes should have docstrings
2. All function parameters should have type hints
3. Use descriptive variable names
4. Keep functions focused on a single responsibility
5. Maximum line length is 88 characters (Black default)
6. Use f-strings for string formatting

Testing Guidelines
---------------

- All new features should include tests
- Aim for at least 80% test coverage
- Tests should be in the ``tests/`` directory, following the same structure as the code
- Use pytest fixtures where appropriate
- Test edge cases and error conditions

Documentation
-----------

- Update the README.md file with any new features or changes to usage
- Add examples to the documentation for new functionality
- Update docstrings for any modified functions or classes
- Build and check the documentation locally:

  .. code-block:: bash

      cd docs
      sphinx-build -b html source build/html
      # View the docs at build/html/index.html

Pull Request Process
-----------------

1. Ensure all tests pass and code quality checks succeed
2. Update documentation as needed
3. Make sure your PR description clearly describes the changes and their purpose
4. Request review from maintainers
5. Address any feedback from reviewers

Release Process
------------

Releases are managed by the core maintainers. The general process is:

1. Update version number in relevant files
2. Update CHANGELOG.md with notable changes
3. Create a tagged release on GitHub
4. Build and publish package to PyPI

Questions?
--------

If you have questions about contributing, please open an issue on GitHub.
