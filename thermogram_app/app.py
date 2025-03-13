"""
Main entry point for the Thermogram Analysis application.
"""

import dash  # type: ignore
from dash import html

import thermogram_baseline
import tlbparam

# Create the Dash app
app = dash.Dash(
    __name__,
    title="Thermogram Analysis",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

# Create the app layout
app.layout = html.Div(
    className="app-container",
    children=[
        # Header
        html.Div(
            className="header",
            children=[
                html.H1("Thermogram Analysis", className="app-title"),
                html.Div(
                    className="header-info",
                    children=[
                        html.Span(
                            f"ThermogramBaseline v{thermogram_baseline.__version__}"
                        ),
                        html.Span(" | "),
                        html.Span(f"TLBParam v{tlbparam.__version__}"),
                    ],
                ),
            ],
        ),
        # Main content
        html.Div(
            className="main-content",
            children=[
                html.Div(
                    className="welcome-message",
                    children=[
                        html.H2("Welcome to Thermogram Analysis"),
                        html.P(
                            "This application provides tools for analyzing "
                            "thermogram data from thermal liquid biopsy (TLB)."
                        ),
                        html.P(
                            "The application is currently under development. "
                            "Stay tuned for more features!"
                        ),
                        html.Div(
                            className="feature-list",
                            children=[
                                html.H3("Planned Features:"),
                                html.Ul(
                                    children=[
                                        html.Li("Thermogram Baseline Subtraction"),
                                        html.Li("Endpoint Detection"),
                                        html.Li("Peak and Valley Metrics"),
                                        html.Li("Batch Processing"),
                                        html.Li("Interactive Visualization"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Footer
        html.Div(
            className="footer",
            children=[
                html.P("© 2025 NAU Thermogram Analysis Project"),
            ],
        ),
    ],
)

# Add some simple CSS
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }
            .app-container {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .app-title {
                margin: 0;
                font-size: 1.8rem;
            }
            .header-info {
                font-size: 0.9rem;
                opacity: 0.8;
            }
            .main-content {
                flex: 1;
                padding: 2rem;
                max-width: 1200px;
                margin: 0 auto;
                width: 100%;
                box-sizing: border-box;
            }
            .welcome-message {
                background-color: white;
                border-radius: 8px;
                padding: 2rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            .feature-list {
                margin-top: 2rem;
            }
            .footer {
                background-color: #2c3e50;
                color: white;
                text-align: center;
                padding: 1rem;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Define main entry point
if __name__ == "__main__":
    app.run_server(debug=True)
