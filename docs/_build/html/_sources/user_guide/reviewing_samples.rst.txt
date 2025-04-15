.. _user_guide_reviewing_samples:

Reviewing Samples and Adjusting Endpoints
=========================================

The "Review Endpoints" tab is the primary workspace for inspecting individual samples, evaluating the automatically determined baseline endpoints, making manual adjustments if necessary, and preparing the data for metric calculation.

Workflow Steps
--------------

1.  **Select Raw Dataset:**
    *   Begin by selecting an uploaded raw dataset from the "Select Dataset for Review" dropdown. This action populates the "Sample Overview" grid below it with all the samples detected in that dataset.

2.  **Select Sample from Grid:**
    *   Click on a row in the "Sample Overview" grid to select a specific sample for review.
    *   The selected sample's ID will appear in the "Review Samples" control panel, and the plots to the right will update to display this sample's data.
    *   The grid shows the current lower and upper baseline endpoints and indicates their source (`auto` or `manual`). It also has checkboxes showing if a sample has been marked as `reviewed` or `exclude`.

3.  **Evaluate Endpoints:**
    *   Examine the **Baseline Subtracted** plot (usually active by default).
    *   The solid line represents the dCp data after subtracting the calculated baseline (dashed red line).
    *   Red vertical lines indicate the current lower and upper temperature endpoints used for the baseline calculation.
    *   Assess if the dashed baseline accurately represents the non-transition regions of the curve. Pay attention to the beginning and end of the curve.
    *   If the baseline and endpoints look reasonable, you can proceed to mark the sample as reviewed.

4.  **Manually Adjust Endpoints (If Necessary):**
    *   If the automatic endpoints are incorrect or the baseline fit is poor, you can manually adjust the endpoints.
    *   In the "Review Samples" control panel, click "Manually Adjust Lower Endpoint" or "Manually Adjust Upper Endpoint".
    *   The corresponding plot will enter selection mode. Click directly on the **Baseline Subtracted** plot at the desired temperature to set the new endpoint.
    *   The plots and baseline calculation will update dynamically as you set new manual endpoints.
    *   The endpoint source in the control panel and grid will change to `manual`.

5.  **Exclude Sample (Optional):**
    *   If a sample is deemed unusable (e.g., noisy data, failed experiment), check the "Exclude this Sample" checkbox in the control panel.
    *   Excluded samples will be skipped during metric calculation and report generation.

6.  **Mark as Reviewed / Navigate:**
    *   **Mark Reviewed & Next:** Once you are satisfied with the endpoints (or have marked the sample for exclusion), click this button.

        *   The "Reviewed" checkbox for the current sample will be checked in the grid.
        *   The selection will automatically move to the *next unreviewed* sample in the grid.
        *   Any manual changes (endpoints, exclusion status) are saved temporarily for this session.
    *   **Previous Sample:** Use this button to navigate back to the previously selected sample.
    *   **Discard Changes:** If you made manual adjustments to the current sample but want to revert to its state *before* you selected it in the grid (either the original auto-endpoints or previously saved manual endpoints), click this button.

7.  **Save Processed Data:**
    *   After reviewing all (or a satisfactory number) of the samples in the dataset, click the **"Save Processed Data"** button located above the grid.
    *   This action permanently saves the current state (endpoints, review status, exclusion status) for all samples in the dataset.
    *   A confirmation message will appear, and the dataset will now be listed under "Processed Thermograms" on the "Data Overview" tab.
    *   **Important:** Only saved *processed* datasets can be used in the "Report Builder" tab.

Tips for Review
---------------

*   Focus on ensuring the baseline (dashed red line) looks flat or follows a consistent trend in the regions *outside* the main transition peaks.
*   The goal is to define endpoints that capture the stable pre-transition and post-transition phases of the thermogram.
*   Use the "Raw Thermogram" plot tab if needed to see the original data without baseline subtraction. 