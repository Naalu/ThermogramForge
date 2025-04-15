.. _contributing_testing:

Running Checks & Tests
======================

Before submitting contributions, please ensure your changes pass the project's quality checks. While automated testing is planned, the current checks focus on code style, formatting, and linting.

Formatting and Linting Checks
-----------------------------

These checks ensure code consistency and help catch potential errors early. They are configured in ``pyproject.toml`` and enforced by `Black`_ and `Ruff`_.

*   **Run Black:** To automatically format your code according to the project style:

    .. code-block:: bash

       black .

*   **Run Ruff:** To check for linting errors and style issues:

    .. code-block:: bash

       ruff check .

It's recommended to run these checks locally before committing. If pre-commit hooks are set up (see :ref:`contributing_getting_started`), these checks might run automatically.

Automated Testing (Planned)
---------------------------

The project is configured to use `pytest`_ for automated testing, along with `pytest-cov`_ for measuring test coverage. However, the test suite is not yet implemented.

*   **Test Location:** Tests should be placed in the ``/tests`` directory in the project root. This directory needs to be created.
*   **Running Tests (Future):** Once tests are added, they can be run using the following command from the project root:

    .. code-block:: bash

       pytest

    To run tests with coverage reporting:

    .. code-block:: bash

       pytest --cov=app --cov=core --cov-report=term-missing

Contributions that include adding tests for existing or new functionality are highly welcome! Please follow standard `pytest` conventions when writing tests.

.. _Black: https://black.readthedocs.io/en/stable/
.. _Ruff: https://beta.ruff.rs/docs/
.. _pytest: https://docs.pytest.org/
.. _pytest-cov: https://pytest-cov.readthedocs.io/ 