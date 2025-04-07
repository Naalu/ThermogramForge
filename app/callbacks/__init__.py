"""
Initialize callbacks package and register all callbacks.
"""

# Import all callback modules to register them
from . import baseline_callbacks, upload_callbacks

__all__ = [
    "upload_callbacks",
    "baseline_callbacks",
]
