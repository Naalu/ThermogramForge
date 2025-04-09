# core/baseline/__init__.py

# Import functions to expose them at the package level
from .advanced import (EndpointSelectionMethod, find_spline_endpoints,
                       subtract_spline_baseline)
from .simple import \
    simple_baseline_subtraction  # Keep simple for now if used elsewhere

__all__ = [
    "EndpointSelectionMethod",
    "find_spline_endpoints",
    "subtract_spline_baseline",
    "simple_baseline_subtraction",
]
