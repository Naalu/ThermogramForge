"""
Custom Cell Renderer components for Dash AG Grid.

Note: These Python components define the structure.
Corresponding JavaScript functions must be added to the assets folder
to handle user interactions (button clicks, checkbox changes) within the grid cells
and potentially trigger Dash callbacks via `cellRendererData` or other methods.
See: https://dash.plotly.com/dash-ag-grid/cell-renderer-components
"""

import dash_bootstrap_components as dbc


# Placeholder Checkbox Renderer
# JS Needed: Function named 'CheckboxRenderer' in assets/ag_grid_renderers.js
# This JS function would receive 'params' and should:
# 1. Render a checkbox input.
# 2. Set its initial state based on `params.value`.
# 3. Add an event listener for changes.
# 4. On change, use `params.setValue(newValue)` to update the grid's data internally
#    and/or trigger a Dash callback (e.g., via `cellValueChanged` event on the grid).
def CheckboxRenderer(params=None):  # params provided by AG Grid JS
    # Again, the actual rendering and interaction logic belongs in the JS counterpart.
    checked = params["value"] if params and "value" in params else False
    field = params["colDef"]["field"] if params and "colDef" in params else "unknown"
    # Use placeholder index for ID, JS needed for real row context
    return dbc.Checkbox(
        id={"type": f"grid-checkbox-{field}", "index": -1},
        checked=bool(checked),
        style={
            "display": "inline-block",
            "margin-left": "10px",
            "cursor": "pointer",
        },  # Style for appearance
    )
