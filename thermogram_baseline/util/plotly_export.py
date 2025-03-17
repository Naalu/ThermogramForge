"""
Utility module for handling Plotly exports with optional kaleido dependency.

This module provides a safe way to export Plotly figures to static images,
handling the case when kaleido is not available.
"""

import warnings
from pathlib import Path
from typing import Optional, Union

from plotly import graph_objs as go  # type: ignore

try:
    import kaleido  # type: ignore

    version = kaleido.__version__

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


def export_plotly_image(
    fig: go.Figure,
    file_path: Union[str, Path],
    format: str = "png",
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 1.0,
) -> bool:
    """
    Export a Plotly figure to a static image file.

    This function safely handles the case when kaleido is not available.

    Args:
        fig: Plotly figure object
        file_path: Path to save the image
        format: Image format (png, jpg, svg, pdf, etc.)
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for the image

    Returns:
        True if export was successful, False otherwise
    """
    if not KALEIDO_AVAILABLE:
        warnings.warn(
            "Kaleido is not installed. Cannot export to static image. "
            "Install with 'pip install kaleido' or 'pip install \".[export]\"'"
        )

        # Try to save as HTML instead
        try:
            html_path = Path(str(file_path).replace(f".{format}", ".html"))
            fig.write_html(str(html_path))
            warnings.warn(f"Saved as HTML instead: {html_path}")
            return False
        except Exception as e:
            warnings.warn(f"Failed to save as HTML: {str(e)}")
            return False

    try:
        # Convert Path to string if needed
        file_path_str = str(file_path)

        # Set width and height if provided
        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)

        # Write image
        fig.write_image(file_path_str, scale=scale)
        return True
    except Exception as e:
        warnings.warn(f"Failed to export image: {str(e)}")

        # Try to save as HTML as fallback
        try:
            html_path = Path(str(file_path).replace(f".{format}", ".html"))
            fig.write_html(str(html_path))
            warnings.warn(f"Saved as HTML instead: {html_path}")
        except Exception as e2:
            warnings.warn(f"Failed to save as HTML: {str(e2)}")

        return False
