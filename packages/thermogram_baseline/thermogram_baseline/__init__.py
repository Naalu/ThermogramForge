"""Thermogram baseline detection and subtraction package."""

__version__ = "0.1.0"

from .detection import detect_endpoints
from .interpolation import interpolate_sample
from .subtraction import subtract_baseline
from .types import (
    BaselineResult,
    BatchProcessingResult,
    Endpoints,
    InterpolatedResult,
    SignalDetectionResult,
    ThermogramData,
)
