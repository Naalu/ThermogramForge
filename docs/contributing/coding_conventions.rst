.. _contributing_coding_conventions:

Coding Conventions
==================

To maintain code quality, consistency, and readability across the ThermogramForge project, please adhere to the following conventions when contributing:

Code Style and Formatting
-------------------------

*   **PEP 8**: Follow the `PEP 8 Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`__.
*   **Black**: Code formatting is enforced using `Black <https://black.readthedocs.io/en/stable/>`__. Ensure your code is formatted with Black before committing. The project configuration in `pyproject.toml` specifies Black settings. Running the pre-commit hooks (if set up) or manually running ``black .`` in the project root will format the code automatically.
*   **Line Length**: Adhere to the line length limit configured for Black (check `pyproject.toml`, typically 88 or 100 characters).

Linting
-------

*   **Ruff**: Code linting is performed using `Ruff <https://beta.ruff.rs/docs/>`__. Ruff checks for a wide range of potential errors, style issues, and complexity problems. The specific rules enabled are configured in `pyproject.toml` under `[tool.ruff]`. Run ``ruff check .`` in the project root to check for linting errors. Pre-commit hooks (if set up) will also run Ruff. Address any reported linting issues before submitting contributions.

Docstrings
----------

*   **Style**: Use `Google Style <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`__ docstrings for all modules, classes, functions, and methods.
*   **Completeness**: Provide clear and concise docstrings that explain the purpose, arguments, return values, and any raised exceptions for public APIs. Docstrings are used to generate the API documentation.
*   **Sphinx Compatibility**: Ensure docstrings are compatible with Sphinx for automatic documentation generation. Use reStructuredText (RST) syntax within docstrings where needed (e.g., for references or code examples).

Type Hinting
------------

*   **PEP 484**: Use `type hints <https://www.python.org/dev/peps/pep-0484/>`__ for function signatures (arguments and return types) and variable annotations where appropriate. This improves code clarity and allows for static analysis.
*   **Clarity**: Aim for clear and accurate type hints. Use types from the `typing` module as needed (e.g., `List`, `Dict`, `Optional`, `Tuple`, `Callable`).

Imports
-------

*   **Organization**: Group imports in the following order:
    1.  Standard library imports (e.g., `os`, `sys`).
    2.  Third-party library imports (e.g., `pandas`, `plotly`, `dash`).
    3.  Local application imports (e.g., `from core.baseline import simple`, `from app.components import ...`).
*   **Sorting**: Within each group, imports should be sorted alphabetically. Ruff can help enforce this (`I` rules).
*   **Absolute vs Relative**: Prefer absolute imports (`from app.utils import ...`) over relative imports (`from ..utils import ...`) for better clarity, especially in deeper package structures.

Naming Conventions
------------------

*   **Modules**: `lowercase_with_underscores.py`
*   **Packages**: `lowercase_with_underscores`
*   **Classes**: `CapWords`
*   **Functions**: `lowercase_with_underscores()`
*   **Methods**: `lowercase_with_underscores()`
*   **Variables**: `lowercase_with_underscores`
*   **Constants**: `UPPERCASE_WITH_UNDERSCORES`

Logging
-------

*   Use the standard Python `logging` module for application logging.
*   Obtain logger instances via `logging.getLogger(__name__)`.
*   Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   Provide informative log messages.

Testing
-------

*   While tests are not yet fully implemented (see :ref:`contributing_testing`), aim to write code that is testable.
*   Separate concerns: Keep core logic separate from UI logic where possible.
*   Use dependency injection where it simplifies testing.

By following these conventions, we can ensure the ThermogramForge codebase remains maintainable and accessible to all contributors. 