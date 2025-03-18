#!/usr/bin/env python3
"""Generate requirements.txt files from pyproject.toml."""

import tomli


def main():
    """Generate requirements files from pyproject.toml."""
    # Read pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    # Extract dependencies
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})

    # Write requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("# Generated from pyproject.toml - DO NOT EDIT\n\n")
        for dep in dependencies:
            f.write(f"{dep}\n")

    # Write requirements-dev.txt
    with open("requirements-dev.txt", "w") as f:
        f.write("# Generated from pyproject.toml - DO NOT EDIT\n\n")
        f.write("-r requirements.txt\n\n")
        for dep in optional_deps.get("dev", []):
            f.write(f"{dep}\n")

    print("Generated requirements.txt and requirements-dev.txt")


if __name__ == "__main__":
    main()
