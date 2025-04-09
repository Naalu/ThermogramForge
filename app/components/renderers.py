"""
Custom Cell Renderer components for Dash AG Grid.

Note: These Python components define the structure.
Corresponding JavaScript functions must be added to the assets folder
to handle user interactions (button clicks, checkbox changes) within the grid cells
and potentially trigger Dash callbacks via `cellRendererData` or other methods.
See: https://dash.plotly.com/dash-ag-grid/cell-renderer-components
"""

import dash_bootstrap_components as dbc
from dash import html


# Placeholder Actions Renderer
# JS Needed: Function named 'ActionsCellRenderer' in assets/dashAgGridComponentFunctions.js
# This JS function would receive 'params' (including row data) and should:
# 1. Render the buttons.
# 2. Add event listeners to the buttons.
# 3. On click, potentially use `params.context` or a Dash client-side callback
#    to trigger a server-side callback, passing the button type ('lower'/'upper')
#    and the row's sample_id.
def ActionsCellRenderer(params=None):  # params provided by AG Grid JS
    # This Python function might not be directly called if using JS renderer.
    # It serves as a conceptual placeholder. The actual rendering happens in JS.
    # If we were using Dash components *without* JS interaction, it would look like this:
    # sample_id = params["data"]["sample_id"] if params and "data" in params else "unknown"
    # We use placeholders for IDs now, real interaction requires JS to get row index/data
    return html.Div(
        [
            dbc.Button(
                "Set Low",
                id={"type": "grid-set-lower-btn", "index": -1},  # Placeholder index
                size="sm",
                color="primary",
                outline=True,
                className="me-1",
                # disabled=True # Initially disabled until JS is added
            ),
            dbc.Button(
                "Set Up",
                id={"type": "grid-set-upper-btn", "index": -1},  # Placeholder index
                size="sm",
                color="primary",
                outline=True,
                # disabled=True
            ),
        ]
    )


# Placeholder Checkbox Renderer
# JS Needed: Function named 'CheckboxRenderer' in assets/dashAgGridComponentFunctions.js
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
        # disabled=True # Initially disabled until JS is added
        style={
            "display": "inline-block",
            "margin-left": "10px",
            "cursor": "pointer",
        },  # Style for appearance
    )
