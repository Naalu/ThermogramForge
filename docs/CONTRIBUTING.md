# Contributing to ThermogramForge

Thank you for considering contributing to ThermogramForge! This document outlines the process for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- (Optional) R 4.0+ for R integration

### Setup Steps

1. Fork the repository on GitHub
2. Clone your fork locally

   ```bash
   git clone https://github.com/Naalu/ThermogramForge.git
   cd ThermogramForge
   ```

3. Create a virtual environment and install dependencies

   ```bash
   # Using uv (recommended)
   uv venv
   uv sync

   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,test]"
   ```

4. Install pre-commit hooks

   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a branch for your feature

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines

3. Run tests

   ```bash
   # Run all tests
   pytest
   
   # Run tests with coverage
   pytest --cov=thermogram_baseline --cov=tlbparam --cov-report=term-missing
   ```

4. Check code quality

   ```bash
   # Run linting and type checking
   python lint.py
   
   # Fix formatting issues automatically
   python lint.py --fix
   ```

5. Commit your changes with a descriptive message

   ```bash
   git commit -m "Add feature: your feature description"
   ```

6. Push to your fork

   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request on GitHub

## Code Style Guidelines

This project follows these style guidelines:

- [Black](https://black.readthedocs.io/) for code formatting
- [Ruff](https://beta.ruff.rs/docs/) for linting
- [MyPy](https://mypy.readthedocs.io/) for type checking
- [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for documentation

### Key Style Points

1. All functions, methods, and classes should have docstrings
2. All function parameters should have type hints
3. Use descriptive variable names
4. Keep functions focused on a single responsibility
5. Maximum line length is 88 characters (Black default)
6. Use f-strings for string formatting

## Testing Guidelines

- All new features should include tests
- Aim for at least 80% test coverage
- Tests should be in the `tests/` directory, following the same structure as the code
- Use pytest fixtures where appropriate
- Test edge cases and error conditions

## Documentation

- Update the README.md file with any new features or changes to usage
- Add examples to the documentation for new functionality
- Update docstrings for any modified functions or classes

## Pull Request Process

1. Ensure all tests pass and code quality checks succeed
2. Update documentation as needed
3. Make sure your PR description clearly describes the changes and their purpose
4. Request review from maintainers
5. Address any feedback from reviewers

## Questions?

If you have questions about contributing, please open an issue on GitHub.
