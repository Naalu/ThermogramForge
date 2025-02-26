"""Thermogram baseline detection and subtraction package."""

__version__ = "0.1.0"

from .baseline import auto_baseline
from .batch import combine_results, process_multiple
from .detection import detect_endpoints
from .interpolation import interpolate_sample
from .signal import detect_signal
from .subtraction import subtract_baseline
from .types import (
    BaselineResult,
    BatchProcessingResult,
    Endpoints,
    InterpolatedResult,
    SignalDetectionResult,
    ThermogramData,
)
