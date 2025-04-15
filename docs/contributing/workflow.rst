.. _contributing_workflow:

Contribution Workflow
=====================

We welcome contributions to ThermogramForge! Please follow this standard workflow based on GitHub Pull Requests (PRs).

1. Fork the Repository
----------------------

*   Navigate to the `ThermogramForge repository <https://github.com/Naalu/ThermogramForge>`__ on GitHub.
*   Click the "Fork" button in the upper-right corner to create your own copy of the repository under your GitHub account.

2. Clone Your Fork
------------------

*   Clone your forked repository to your local machine:

    .. code-block:: bash

        git clone https://github.com/<Your-GitHub-Username>/ThermogramForge.git
        cd ThermogramForge

3. Set Up Development Environment
-------------------------------

*   Ensure you have followed the :doc:`getting_started` guide to set up your local development environment, including installing dependencies with the `[dev]` extras.

4. Create a New Branch
----------------------

*   Before starting work on a specific feature or bug fix, create a new branch from the main development branch (typically `main` or `develop` - check the repository):

    .. code-block:: bash

        git checkout main  # Or the primary development branch
        git pull origin main # Ensure your main branch is up-to-date
        git checkout -b <your-branch-name> # e.g., feature/add-new-metric or fix/plot-rendering-bug

    *Use descriptive branch names.*

5. Make Your Changes
--------------------

*   Write your code, add tests (if applicable), and update documentation as needed.
*   Ensure your changes adhere to the :doc:`coding_conventions`.
*   Run local checks (:doc:`testing`) like formatting (``black .``) and linting (``ruff check .``) frequently.

6. Commit Your Changes
----------------------

*   Commit your changes with clear and concise commit messages. Follow conventional commit message formats if possible (e.g., ``feat: Add Tm calculation``, ``fix: Correct baseline subtraction``).

    .. code-block:: bash

        git add <files-you-changed>
        git commit -m "Your descriptive commit message"

7. Push to Your Fork
--------------------

*   Push your local branch to your forked repository on GitHub:

    .. code-block:: bash

        git push origin <your-branch-name>

8. Create a Pull Request (PR)
-----------------------------

*   Go to your forked repository on GitHub (``https://github.com/<Your-GitHub-Username>/ThermogramForge``).
*   GitHub should automatically detect the pushed branch and suggest creating a Pull Request. If not, navigate to the "Pull requests" tab and click "New pull request".
*   Ensure the base repository is ``Naalu/ThermogramForge`` and the base branch is the main development branch (e.g., ``main``).
*   Ensure the head repository is your fork and the compare branch is ``<your-branch-name>``.
*   Provide a clear title and detailed description for your PR, explaining the changes made and referencing any related issues (e.g., "Closes #123").
*   Submit the Pull Request.

9. Code Review and Merge
------------------------

*   Project maintainers will review your PR. Address any feedback or requested changes by pushing additional commits to your branch.
*   Once approved, a maintainer will merge your PR into the main project repository.

Thank you for contributing! 