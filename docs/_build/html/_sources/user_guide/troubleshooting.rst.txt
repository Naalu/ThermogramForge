.. _user_guide_troubleshooting:

Troubleshooting
===============

This section provides solutions to common issues you might encounter while using ThermogramForge.

Installation Issues
-------------------

*   **Command not found (``python``, ``pip``, ``git``):** Ensure Python and Git are installed correctly and their executable paths are included in your system's PATH environment variable. Refer to the :doc:`installation` guide.
*   **Error during ``pip install -r requirements.txt``:**
    *   **Network Issues:** Check your internet connection. If you are behind a firewall or proxy, pip might need specific configuration.
    *   **Dependency Conflicts:** If errors mention conflicting package versions, try creating a fresh virtual environment (``python -m venv venv``, activate it) before running ``pip install``. Sometimes, older cached packages can cause issues. You might need to clear pip's cache (``pip cache purge``).
    *   **Missing System Dependencies:** Some Python packages might rely on system libraries (especially for plotting or numerical computation). Check the error messages for clues about missing C libraries or compilers and install them using your system's package manager (e.g., ``apt``, ``brew``, ``yum``).

Application Startup Issues
--------------------------

*   **``Address already in use`` error:** This usually means another process is already using the default port (8050).
    *   Stop any previous instances of ThermogramForge running in other terminals (use Ctrl+C).
    *   If you can't find the process, you might need to identify and terminate the process using port 8050 (commands vary by OS, e.g., ``lsof -i :8050`` on macOS/Linux, ``netstat -ano | findstr :8050`` on Windows, then use ``kill`` or Task Manager).
    *   Alternatively, you can run the app on a different port: ``python main.py --port 8051`` (Note: This requires modifying ``main.py`` to handle command-line arguments or directly changing the ``app.run`` call).
*   **ModuleNotFoundError:** Ensure you have activated the correct virtual environment where you installed the dependencies. Make sure you are running ``python main.py`` from the root ``ThermogramForge`` directory.

Data Upload Issues
------------------

*   **File Not Uploading:** Check the file size (default limit might be around 20MB) and ensure the file type is supported (``.csv``, ``.xls``, ``.xlsx``).
*   **Error Message after Upload ("Could not identify any valid sample data columns..."):**
    *   Verify your file structure matches the expected formats described in :doc:`data_formats`.
    *   Check for consistent column naming, especially the "Temperature" column and the T[ID]/[ID] pattern for multi-sample files.
    *   Ensure column headers are present in the first row.
    *   Look for typos or special characters in column headers.
*   **Error Message after Upload ("No samples remained after temperature filtering..."):**
    *   Check the Min/Max Temperature values you entered in the upload modal. Ensure they encompass the actual temperature range in your data file.
    *   Verify the "Temperature" column in your file contains valid numeric data within the expected range.

Analysis and Display Issues
---------------------------

*   **Plots Not Appearing or Showing Errors:**
    *   Ensure a dataset and then a specific sample are selected in the "Review Endpoints" tab.
    *   Check the browser's developer console (usually F12) for any JavaScript errors that might indicate problems with plotting libraries or data processing.
    *   Ensure the selected sample data is valid (numeric Temperature and dCp).
*   **Incorrect Baseline:**
    *   The automatic baseline detection might struggle with very noisy data or unusual curve shapes. Use the manual endpoint adjustment feature described in :doc:`reviewing_samples`.
    *   Consider adjusting the "Advanced Options" (Exclusion Temperatures, Point Selection Method) in the Upload Modal if automatic detection consistently fails for your data type.
*   **Missing Metrics in Report:**
    *   Ensure you selected a *processed* dataset in the Report Builder.
    *   Peak-related metrics (Tm, FWHM, etc.) depend on successful peak detection. If peaks are noisy, broad, or absent, these metrics might not be calculated. Check the baseline-subtracted plot for clear peak features.
    *   Verify that the required metrics were selected in the checklist before generating the report.

Getting Further Help
--------------------

If these tips don't resolve your issue, consider opening an issue on the project's GitHub repository, providing details about:

*   The version of ThermogramForge you are using.
*   Your operating system and Python version.
*   The exact steps to reproduce the problem.
*   Any error messages from the terminal or browser console.
*   (If possible and non-sensitive) An example of the data file causing the issue. 