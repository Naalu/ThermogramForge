Usage
=====

Baseline Subtraction
-------------------

.. code-block:: python

    import thermogram_baseline as tb

    # Load thermogram data
    import polars as pl
    data = pl.read_csv("your_thermogram_data.csv")

    # Detect endpoints
    from thermogram_baseline.endpoint_detection import detect_endpoints
    endpoints = detect_endpoints(data)

    # Subtract baseline
    from thermogram_baseline.baseline import subtract_baseline
    baseline_subtracted = subtract_baseline(data, endpoints.lower, endpoints.upper)

Metrics Calculation
-----------------

.. code-block:: python

    import tlbparam as tlb

    # Detect peaks
    from tlbparam.peak_detection import PeakDetector

    detector = PeakDetector()
    peaks = detector.detect_peaks(baseline_subtracted)

    # Calculate metrics
    from tlbparam.metrics import generate_summary
    metrics = generate_summary(baseline_subtracted)

Visualization
-----------

.. code-block:: python

    from tlbparam.visualization import plot_thermogram, plot_with_peaks

    # Visualize thermogram data
    fig = plot_thermogram(baseline_subtracted)
    fig.show()  # Display in browser or notebook

    # Visualize with detected peaks
    fig_peaks = plot_with_peaks(baseline_subtracted, peaks)
    fig_peaks.show()

    # Save visualization as HTML or image
    fig_peaks.write_html("thermogram_with_peaks.html")
    fig_peaks.write_image("thermogram_with_peaks.png")

Interactive Web Application
-------------------------

ThermogramForge includes a web application for interactive analysis:

.. code-block:: python

    # Run the application
    python -m thermogram_app.app

    # Then open your browser to http://127.0.0.1:8050/
