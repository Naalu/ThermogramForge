.. _user_guide_interface:

Interface Overview
==================

ThermogramForge presents a tabbed interface to guide you through the data analysis workflow. Here's an overview of the main components:

.. figure:: /_static/images/interface/main_tabs.png
   :alt: Main application tabs: Data Overview, Review Endpoints, Report Builder.
   :align: center

   The three main tabs for navigating the application workflow.

Main Tabs
---------

The application is organized into three primary tabs:

1. **Data Overview Tab:**

   * **Purpose:** Provides a summary of all data currently loaded or generated within the application session.
   * **Sections:**
     
     * **Raw Thermogram Files:** Lists the raw data files that have been uploaded. Includes a button to upload new raw data.
     * **Processed Thermograms:** Lists datasets that have been processed (e.g., after baseline review). Includes a button to upload previously processed data.
     * **Reports Generated:** Lists any analysis reports that have been generated during the session.

   .. figure:: /_static/images/interface/overview_tab.png
      :alt: Data Overview tab showing lists for raw files, processed data, and reports.
      :align: center

      Example view of the Data Overview tab.

2. **Review Endpoints Tab:**

   * **Purpose:** This is the main workspace for inspecting individual thermogram samples, reviewing automatically detected baseline endpoints, and making manual adjustments.
   * **Workflow:**
     
     1. **Select Dataset:** Choose an uploaded *raw* dataset from the dropdown menu.

        .. figure:: /_static/images/interface/review_selector.png
           :alt: Dataset selector dropdown on the Review Endpoints tab.
           :width: 60%
           :align: center

           Selecting a raw dataset for review.

     2. **Sample Overview Grid:** An AG Grid table appears, listing all samples within the selected dataset. It shows sample IDs, current lower/upper baseline endpoints (and their source: auto or manual), and checkboxes for review status and exclusion. Click a row to select a sample.

        .. figure:: /_static/images/interface/review_grid.png
           :alt: AG Grid showing sample IDs, endpoints, and review status.
           :align: center

           Sample Overview Grid displaying data for the selected dataset.

     3. **Control Panel:** Displays the details and controls for the *currently selected sample* from the grid. Here you can:
        
        * View the current lower and upper baseline endpoints.
        * Click buttons ("Manually Adjust Lower/Upper Endpoint") to enter selection mode on the plots.
        * Mark the sample for exclusion from analysis.
        * Discard any manual changes made to the current sample.
        * Mark the sample as reviewed and automatically advance to the next sample in the grid.
        * Navigate to the previous/next sample.

        .. figure:: /_static/images/interface/review_control_panel.png
           :alt: Control panel showing endpoint details and action buttons.
           :width: 70%
           :align: center

           Control panel for the selected sample.

     4. **Plot Area:** Shows visualizations for the selected sample. Tabs allow switching between:
        
        * **Baseline Subtracted:** Shows the dCp data after baseline subtraction, along with the calculated baseline itself. Vertical lines indicate the currently selected endpoints. This is the primary plot for adjusting endpoints.
        * **Raw Thermogram:** Shows the original, unprocessed dCp vs. Temperature data.

        .. figure:: /_static/images/interface/review_plots.png
           :alt: Plot area showing baseline subtracted and raw thermogram tabs.
           :align: center

           Plotting area displaying the thermogram for the selected sample.

     5. **Save Processed Data:** Once satisfied with the endpoint reviews for the dataset, use the "Save Processed Data" button (located above the grid) to save the results. This processed dataset will then appear in the "Processed Thermograms" list on the Data Overview tab and become available for report generation.

3. **Report Builder Tab:**

   * **Purpose:** Allows you to generate summary reports containing calculated metrics for *processed* datasets.
   * **Workflow:**
     
     1. **Select Dataset:** Choose a *processed* dataset (one that has been reviewed and saved) from the dropdown.
     2. **Report Configuration:** Set the desired name for the report file and choose the output format (CSV or Excel).
     3. **Select Metrics:** Choose which calculated metrics (e.g., Tm, Onset) to include in the report using the checklist. Tooltips provide descriptions for each metric. Buttons allow selecting/clearing/resetting the metric selection.
     4. **Report Preview:** A table shows a preview of the report based on the selected dataset and metrics.
     5. **Generate Report:** Click the "Generate Report" button to create and download the report file. The generated report will also be listed on the Data Overview tab.

   .. figure:: /_static/images/interface/report_builder_tab.png
      :alt: Report Builder tab showing dataset selection, configuration, metrics, and preview.
      :align: center

      Example view of the Report Builder tab.

Modals
------

* **Upload Raw Data Modal:** Accessed via buttons on the "Data Overview" and "Review Endpoints" tabs. Allows uploading new ``.csv`` or ``.xlsx`` files containing raw thermogram data. Includes options for setting initial temperature filtering ranges and advanced parameters for automatic endpoint detection.

  .. figure:: /_static/images/interface/upload_raw_modal.png
     :alt: Modal dialog for uploading raw thermogram files.
     :width: 80%
     :align: center

     Raw data upload modal.

* **Upload Processed Data Modal:** Accessed via a button on the "Data Overview" tab. Allows uploading ``.json`` files containing previously processed and saved ThermogramForge datasets.

  .. figure:: /_static/images/interface/upload_processed_modal.png
     :alt: Modal dialog for uploading processed data JSON files.
     :width: 80%
     :align: center

     Processed data upload modal.

Navigation Tips
---------------

* Use the main tabs to switch between different stages of the workflow.
* On the "Review Endpoints" tab, selecting a dataset populates the grid, and selecting a sample in the grid activates the control panel and updates the plots.
* On the "Report Builder" tab, selecting a processed dataset enables the configuration and preview sections.