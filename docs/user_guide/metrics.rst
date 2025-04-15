.. _user_guide_metrics:

Calculated Metrics
==================

ThermogramForge calculates several metrics from the processed (baseline-subtracted) thermogram data. These metrics can be selected for inclusion in generated reports via the "Report Builder" tab.

Available Metrics
-----------------

The following metrics are available for reporting:

*   **TPeak_F (°C):** Temperature of the peak corresponding to the Fibrinogen temperature region (typically 47-60°C). Requires peak detection to be successful in this region.
*   **TPeak_1 (°C):** Temperature of the first major detected peak.
*   **TPeak_2 (°C):** Temperature of the second major detected peak (if present).
*   **TPeak_3 (°C):** Temperature of the third major detected peak (if present).
*   **TMax (°C):** Temperature corresponding to the maximum observed dCp value across the entire curve.
*   **TMin (°C):** Temperature corresponding to the minimum observed dCp value across the entire curve.
*   **Area:** Total area under the baseline-subtracted curve, calculated using the trapezoidal rule. Units depend on the dCp units (e.g., kcal/mol/°C * °C -> kcal/mol).
*   **TFM (°C):** Temperature of the First Moment. This is a measure of the "center of mass" of the curve, calculated as the weighted average of temperature by the dCp values.
*   **FWHM (°C):** Full Width at Half Maximum. The width of the primary peak (usually TPeak_1 if present) measured at half of its maximum height above the baseline. Provides an indication of the transition's sharpness.
*   **TV12 (°C):** Temperature of the valley (minimum dCp) located between TPeak_1 and TPeak_2, if both peaks are present.
*   **V12:** Height (dCp value) of the valley located between TPeak_1 and TPeak_2, if both peaks are present.

Default Selection
-----------------

By default, the following metrics are pre-selected in the Report Builder:

*   TPeak_1
*   TPeak_2
*   TPeak_3
*   Area
*   FWHM
*   TFM

You can modify this selection using the checklist and control buttons ("Select All", "Clear All", "Reset Selection") in the Report Builder tab before generating a report.

Notes on Calculation
---------------------

*   All metrics are calculated using the **baseline-subtracted** dCp data.
*   Peak-related metrics (TPeak_*, FWHM, TV12, V12) depend on the success and results of the underlying peak detection algorithm (``scipy.signal.find_peaks`` and region-specific logic). If peaks are not detected or are ambiguous, the corresponding metrics may be missing or ``NaN`` (Not a Number).
*   Area and TFM are calculated across the entire temperature range defined by the baseline endpoints.
*   Ensure your data quality and baseline correction are adequate for meaningful metric calculation. Poor baseline subtraction can significantly affect results. 