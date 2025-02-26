"""Thermogram baseline detection and subtraction package."""

__version__ = "0.1.0"

from .baseline import auto_baseline as auto_baseline
from .batch import combine_results as combine_results
from .batch import process_multiple as process_multiple
from .detection import detect_endpoints as detect_endpoints
from .interpolation import interpolate_sample as interpolate_sample
from .signal import detect_signal as detect_signal
from .subtraction import subtract_baseline as subtract_baseline
from .types import BaselineResult as BaselineResult
from .types import BatchProcessingResult as BatchProcessingResult
from .types import Endpoints as Endpoints
from .types import InterpolatedResult as InterpolatedResult
from .types import SignalDetectionResult as SignalDetectionResult
from .types import ThermogramData as ThermogramData
from .visualization import create_heatmap as create_heatmap
from .visualization import plot_baseline_result as plot_baseline_result
from .visualization import plot_interpolated_result as plot_interpolated_result
from .visualization import plot_multiple_thermograms as plot_multiple_thermograms
from .visualization import plot_thermogram as plot_thermogram

__all__ = [
    "auto_baseline",
    "combine_results",
    "process_multiple",
    "detect_endpoints",
    "interpolate_sample",
    "detect_signal",
    "subtract_baseline",
    "BaselineResult",
    "BatchProcessingResult",
    "Endpoints",
    "InterpolatedResult",
    "SignalDetectionResult",
    "ThermogramData",
    "plot_thermogram",
    "plot_baseline_result",
    "plot_interpolated_result",
    "plot_multiple_thermograms",
    "create_heatmap",
]
