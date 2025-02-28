"""Core type definitions for thermogram data processing."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl


@dataclass
class ThermogramData:
    """Represents thermogram data with temperature and heat capacity values.

    A container for thermogram measurement data with methods for DataFrame conversion
    and basic data operations.

    Attributes:
        temperature: Array of temperature values (°C)
        dcp: Array of excess heat capacity values (kJ/mol·K)

    Examples:
        >>> # Create from numpy arrays
        >>> data = ThermogramData(
        ...     temperature=np.array([25.0, 26.0, 27.0]),
        ...     dcp=np.array([1.5, 1.6, 1.7])
        ... )
        >>>
        >>> # Create from DataFrame
        >>> df = pl.DataFrame({
        ...     "Temperature": [25.0, 26.0],
        ...     "dCp": [1.5, 1.6]
        ... })
        >>> data = ThermogramData.from_dataframe(df)
        >>>
        >>> # Convert back to DataFrame
        >>> df = data.to_dataframe()
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
            temp_col: Name of temperature column. Defaults to "Temperature".
            dcp_col: Name of heat capacity column. Defaults to "dCp".

        Returns:
            ThermogramData: New instance containing the DataFrame data.

        Raises:
            KeyError: If specified columns are not found in DataFrame.
            ValueError: If DataFrame is empty or contains invalid values.
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

    Defines temperature points used to anchor baseline calculation.

    Attributes:
        lower: Lower temperature endpoint (°C)
        upper: Upper temperature endpoint (°C)
        method: Strategy used for endpoint selection:
            - "innermost": Points closest to transition
            - "outmost": Points farthest from transition
            - "mid": Points in middle of ranges
            Defaults to "innermost".

    Examples:
        >>> endpoints = Endpoints(lower=55.0, upper=85.0, method="innermost")
        >>> print(f"Lower endpoint: {endpoints.lower}°C")
    """

    lower: float
    upper: float
    method: Literal["innermost", "outmost", "mid"] = "innermost"


@dataclass
class BaselineResult:
    """Represents the result of baseline subtraction.

    Contains original data, calculated baseline, and corrected data.

    Attributes:
        original: Original thermogram measurements
        baseline: Calculated baseline curve
        subtracted: Data after baseline subtraction
        endpoints: Endpoints used for baseline calculation

    Examples:
        >>> result = subtract_baseline(data, lower_temp=55, upper_temp=85)
        >>> corrected = result.subtracted
        >>> baseline = result.baseline
    """

    original: ThermogramData
    baseline: ThermogramData
    subtracted: ThermogramData
    endpoints: Endpoints


@dataclass
class InterpolatedResult:
    """Represents the result after interpolation to a uniform grid.

    Contains the interpolated data along with optional original data and baseline
    subtraction results for reference.

    Attributes:
        data: Interpolated thermogram data on uniform temperature grid
        grid_temp: Temperature points used for interpolation (°C)
        original_data: Optional original data before interpolation
        baseline_result: Optional baseline subtraction result if performed

    Examples:
        >>> result = interpolate_sample(data, grid_temp=np.arange(45, 90, 0.1))
        >>> processed = result.data
        >>> temps = result.grid_temp
        >>>
        >>> # Access original data if available
        >>> if result.original_data:
        ...     original = result.original_data
    """

    data: ThermogramData
    grid_temp: np.ndarray
    original_data: Optional[ThermogramData] = None
    baseline_result: Optional[BaselineResult] = None


@dataclass
class SignalDetectionResult:
    """Represents the result of signal detection analysis.

    Contains both the binary classification result and confidence measure,
    along with optional method-specific detection details.

    Attributes:
        is_signal: Boolean indicating presence of meaningful signal
        confidence: Confidence level in the detection (0.0 to 1.0)
        details: Optional dictionary with method-specific statistics:
            - peaks: Peak count, prominence values
            - arima: AIC values, model comparisons
            - adf: Test statistics, p-values

    Examples:
        >>> result = detect_signal(data, method="peaks")
        >>> if result.is_signal and result.confidence > 0.8:
        ...     print("High confidence signal detected")
        >>> print(f"Detection details: {result.details}")
    """

    is_signal: bool
    confidence: float
    details: Optional[dict] = None


@dataclass
class BatchProcessingResult:
    """Represents the result of batch processing multiple thermograms.

    Contains processing results for each sample along with shared parameters
    and optional processing statistics.

    Attributes:
        results: Dictionary mapping sample IDs to their processing results
        grid_temp: Common temperature grid used across all samples (°C)
        processing_stats: Optional dictionary with processing statistics:
            - processing_time: Time taken for each sample
            - error_count: Number of failed samples
            - warnings: Processing warnings by sample

    Examples:
        >>> batch_result = process_multiple(samples, max_workers=4)
        >>> for sample_id, result in batch_result.results.items():
        ...     processed_data = result.data
        ...     print(f"Processed {sample_id}")
        >>>
        >>> # Check processing statistics
        >>> if batch_result.processing_stats:
        ...     print(f"Errors: {batch_result.processing_stats['error_count']}")
    """

    results: dict[str, InterpolatedResult]
    grid_temp: np.ndarray
    processing_stats: Optional[dict] = None
