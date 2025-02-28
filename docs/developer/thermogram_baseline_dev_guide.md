# Thermogram Baseline Developer Guide

This guide provides information for developers who want to understand, maintain, or extend the `thermogram_baseline` package.

## Architecture Overview

The `thermogram_baseline` package is organized into several modules, each with a specific focus:

thermogram_baseline/
├── init.py            # Package exports
├── types.py               # Type definitions and common structures
├── detection.py           # Endpoint detection algorithms
├── subtraction.py         # Baseline subtraction algorithms
├── interpolation.py       # Interpolation utilities
├── baseline.py            # Main workflow integration
├── signal.py              # Signal detection algorithms
├── batch.py               # Batch processing utilities
└── visualization.py       # Plotting utilities

### Module Relationships

- `types.py` defines data structures used throughout the package
- `detection.py`, `subtraction.py`, and `interpolation.py` implement core algorithms
- `baseline.py` integrates these core algorithms into a unified workflow
- `signal.py` provides signal detection capabilities
- `batch.py` enables processing multiple thermograms
- `visualization.py` creates interactive plots of results

## Design Principles

### 1. Type Safety

The package uses dataclasses and type hints to provide clear interfaces and improve code readability. All functions include complete type annotations.

### 2. Flexible Inputs

Functions accept multiple input types where sensible:

- `ThermogramData` objects for internal use
- Polars DataFrames for user convenience
- Various dictionary formats for batch processing

### 3. Comprehensive Results

Processing functions return rich result objects that contain:

- Original input data
- Intermediate results
- Final processed data
- Processing parameters

### 4. Error Handling

Functions include robust error handling:

- Input validation with clear error messages
- Graceful handling of edge cases
- Error tracking in batch processing

### 5. Performance Focus

Performance optimizations include:

- Vectorized operations with NumPy
- Efficient data processing with Polars
- Parallel execution for batch operations

## Extending the Package

### Adding New Baseline Methods

To add a new baseline subtraction method:

1. Add the new method to `subtraction.py`
2. Update the `subtract_baseline` function to support the new method
3. Add tests for the new method
4. Update documentation

Example:

```python
def subtract_baseline_polynomial(
    data: ThermogramData,
    degree: int = 2,
    regions: Optional[List[Tuple[float, float]]] = None,
) -> BaselineResult:
    """Subtract a polynomial baseline from thermogram data."""
    # Implementation here
    
    return BaselineResult(...)
```

### Adding Visualization Types

To add a new visualization:

1. Add the new function to visualization.py
2. Ensure it returns a compatible Plotly figure dictionary
3. Add tests for the new function
4. Update documentation

Example:

```python
def plot_derivative_thermogram(
    data: Union[ThermogramData, pl.DataFrame],
    window_size: int = 5,
    title: str = "Derivative Thermogram",
) -> dict:
    """Create a plot of the derivative of a thermogram."""
    # Implementation here
    
    return fig
```

### Creating Custom Workflows

You can create custom workflows by combining existing functions:

```python
def custom_workflow(
    data: pl.DataFrame,
    custom_param: float,
) -> InterpolatedResult:
    """Custom thermogram processing workflow."""
    # Implementation combining existing functions
    
    return result
```

## Testing Guidelines

### Test Categories

- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test function combinations
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark key operations

### Test Data Generation

The `test_data_utils.py` module provides functions for generating synthetic test data:

- `generate_simple_thermogram()`: Creates a thermogram with a single peak
- `generate_multi_peak_thermogram()`: Creates a thermogram with multiple peaks
- `generate_real_like_thermogram()`: Creates a realistic thermogram

### Mocking

When testing functions that depend on other components, consider using mocks:

```python
from unittest.mock import patch

def test_with_mock():
    with patch('thermogram_baseline.detection.detect_endpoints') as mock_detect:
        mock_detect.return_value = Endpoints(lower=50.0, upper=85.0)
        # Test code that uses detect_endpoints
```

## Performance Considerations

### Memory Management

- For large datasets, consider processing in batches
- Use `Polars` for efficient data handling
- Release large objects when no longer needed

### Computational Efficiency

- Use vectorized operations where possible
- Profile code to identify bottlenecks

### Parallel Processing

- Use `concurrent.futures` for parallelization
- Consider the overhead of parallelization for small tasks
- Set appropriate `max_workers` values for your hardware

## Coding Standards

- Follow PEP 8 for Python style guidelines
- Add comprehensive docstrings in Google-format
- Include type hints for all functions and classes
- Write unit tests for new functionality
- Keep functions focused on a single responsibility

## Documentation Standards

- Document all public functions and classes
- Include examples in docstrings
- Explain parameters and return values
- Document exceptions that may be raised
- Use cross-references to related functionality

## Pull Request Process

When contributing to the project:

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Implement your changes
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

## Common Issues and Solutions

### Endpoint Detection

**Issue**: Endpoint detection fails to find suitable endpoints

**Solution**:

- Adjust window size and exclusion zones
- Use different point selection method
- Check input data for anomalies

### Signal Detection

**Issue**: Signal detection gives inconsistent results

**Solution**:

- Try different detection methods
- Adjust threshold parameters
- Preprocess data to reduce noise

### Performance Issues

**Issue**: Slow processing for large datasets

**Solution**:

- Use batch processing with appropriate batch sizes
- Optimize parallel processing parameters
- Profile code to identify bottlenecks
  