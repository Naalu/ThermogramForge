"""Core type definitions for thermogram data processing."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl


@dataclass
class ThermogramData:
    """Represents thermogram data with temperature and dCp values.

    Attributes:
        temperature: Array of temperature values in degrees Celsius
        dcp: Array of dCp (excess heat capacity) values
    """

    temperature: np.ndarray
    dcp: np.ndarray

    @classmethod
    def from_dataframe(
        cls, df: pl.DataFrame, temp_col: str = "Temperature", dcp_col: str = "dCp"
    ) -> "ThermogramData":
        """Create a ThermogramData instance from a Polars DataFrame.

        Args:
            df: DataFrame containing thermogram data
            temp_col: Name of the temperature column
            dcp_col: Name of the dCp column

        Returns:
            ThermogramData instance
        """
        return cls(
            temperature=df[temp_col].to_numpy(),
            dcp=df[dcp_col].to_numpy(),
        )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to a Polars DataFrame.

        Returns:
            DataFrame with Temperature and dCp columns
        """
        return pl.DataFrame(
            {
                "Temperature": self.temperature,
                "dCp": self.dcp,
            }
        )

    def __len__(self) -> int:
        """Return the number of data points."""
        return len(self.temperature)


@dataclass
class Endpoints:
    """Represents the endpoints for baseline subtraction.

    Attributes:
        lower: Lower temperature endpoint
        upper: Upper temperature endpoint
        method: Method used for point selection
    """

    lower: float
    upper: float
    method: Literal["innermost", "outmost", "mid"] = "innermost"


@dataclass
class BaselineResult:
    """Represents the result of baseline subtraction.

    Attributes:
        original: Original thermogram data
        baseline: Calculated baseline
        subtracted: Baseline-subtracted data
        endpoints: Endpoints used for baseline subtraction
    """

    original: ThermogramData
    baseline: ThermogramData
    subtracted: ThermogramData
    endpoints: Endpoints


@dataclass
class InterpolatedResult:
    """Represents the result after interpolation to a uniform grid.

    Attributes:
        data: Interpolated thermogram data
        grid_temp: Temperature grid used for interpolation
        original_data: Optional original data before interpolation
        baseline_result: Optional baseline subtraction result
    """

    data: ThermogramData
    grid_temp: np.ndarray
    original_data: Optional[ThermogramData] = None
    baseline_result: Optional[BaselineResult] = None


@dataclass
class SignalDetectionResult:
    """Represents the result of signal detection.

    Attributes:
        is_signal: Whether the thermogram contains a signal
        confidence: Confidence level of the detection
        details: Optional additional detection details
    """

    is_signal: bool
    confidence: float
    details: Optional[dict] = None


@dataclass
class BatchProcessingResult:
    """Represents the result of batch processing multiple thermograms.

    Attributes:
        results: Dictionary mapping sample IDs to processing results
        grid_temp: Common temperature grid used for all samples
        processing_stats: Optional processing statistics
    """

    results: dict[str, InterpolatedResult]
    grid_temp: np.ndarray
    processing_stats: Optional[dict] = None
