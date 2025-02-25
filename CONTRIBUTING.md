# Contributing to ThermogramForge

## Branching Strategy

We follow a simplified GitFlow workflow:

### Main Branches

- `main`: Production-ready code. Protected branch, only accepts PRs from `develop`.
- `develop`: Integration branch for features. Protected branch, only accepts PRs from feature branches.

### Supporting Branches

- `feature/<name>`: New features or enhancements. Branch from and merge to `develop`.
- `bugfix/<name>`: Bug fixes. Branch from and merge to `develop`.
- `release/<version>`: Release preparation. Branch from `develop` and merge to `main` and back to `develop`.
- `hotfix/<name>`: Urgent fixes for production. Branch from `main` and merge to both `main` and `develop`.

## Workflow

1. For new features:

```bash
git checkout develop
git pull
git checkout -b feature/your-feature-name

# Make changes
git commit -m "Add feature X"

# Push and Create PR to develop
git push origin feature/your-feature-name
```

2. For bug fixes:

```bash
git checkout develop
git pull
git checkout -b bugfix/issue-description

# Make changes
git commit -m "Fix issue Y"

# Push and Create PR to develop
git push origin bugfix/issue-description
```

3. For releases:

```bash
git checkout develop
git pull
git checkout -b release/vX.Y.Z

# Update version, changelog, etc.
git commit -m "Prepare release vX.Y.Z"

# Push and Create PR to main
git push origin release/vX.Y.Z
```

4. For hotfixes:

```bash
git checkout main
git pull
git checkout -b hotfix/critical-issue-description

# Make changes 
git commit -m "Fix critical issue Z"

# Push and Create PR to main and develop
git push origin hotfix/critical-issue-description
```

### Commit Messages

Follow the conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code changes that neither fix bugs nor add features
- `perf:` Performance improvements
- `test:` Adding or fixing tests
- `chore:` Maintenance tasks

Example: `feat: add endpoint detection algorithm`

## Code Style

### Documentation

We use Google-style docstrings for all functions and classes.

```python
def my_function(param1: int, param2: str) -> None:
    """Short description of the function.

    Longer description of the function, including parameters and return value.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.

    Returns:
        Description of the return value.
    """
    pass
```

When adding new features, ensure to update the documentation accordingly.

### Code Formatting

When in doubt, follow the [PEP 8](https://pep8.org/) style guide.

We use `Ruff` for code formatting. Run `ruff format` before committing your changes.

### Type Checking

We use type hints for all functions and classes.

```python
def my_function(param1: int, param2: str) -> None:
    pass
```

We also use `mypy` for type checking. Ensure your code passes type checking before creating a PR.

### Testing

Where possible, add tests to cover new features and bug fixes.

We use `pytest` for testing. Run `pytest` before committing your changes.
