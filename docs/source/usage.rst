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

Interactive Web Application
-------------------------

ThermogramForge includes a web application for interactive analysis:

.. code-block:: python

    # Run the application
    python -m thermogram_app.app
    
    # Then open your browser to http://127.0.0.1:8050/