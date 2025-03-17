from typing import Any, Dict, List, Optional, Union

from plotly.graph_objects import Figure

# This is a minimal stub file for plotly.subplots to satisfy type checking

def make_subplots(
    rows: int = 1,
    cols: int = 1,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    start_cell: str = "top-left",
    print_grid: bool = False,
    vertical_spacing: Optional[float] = None,
    horizontal_spacing: Optional[float] = None,
    subplot_titles: Optional[Union[List[str], tuple]] = None,
    row_heights: Optional[List[float]] = None,
    column_widths: Optional[List[float]] = None,
    specs: Optional[List[List[Dict[str, Any]]]] = None,
    **kwargs: Any,
) -> Figure: ...
