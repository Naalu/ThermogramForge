Quickstart Tutorial
===================

This guide provides a quick walkthrough to get you started with ThermogramForge after you have completed the :doc:`installation`.

1.  **Navigate to the Project Directory:**

    Make sure you are in the `ThermogramForge` directory where you cloned the repository.

    .. code-block:: bash

        cd path/to/ThermogramForge

2.  **Activate Virtual Environment (if applicable):**

    If you followed the recommended installation steps, activate your virtual environment:

    *   On **macOS and Linux**:

        .. code-block:: bash

            source venv/bin/activate

    *   On **Windows**:

        .. code-block:: bash

            venv\Scripts\activate

3.  **Run the Application:**

    Start the Dash application by running `main.py`:

    .. code-block:: bash

        python main.py

    The application will start, and you'll see output in your terminal indicating the server is running, typically on `http://127.0.0.1:8050/`.

4.  **Access the Application:**

    Open your web browser and navigate to the address shown in the terminal (usually `http://127.0.0.1:8050/`).

5.  **Upload Data:**

    Use the upload component in the application interface to select and upload your thermogram data files. Refer to the :doc:`data_formats` section for details on supported file types and structures.

6.  **Explore and Analyze:**

    Once data is uploaded, you can:
    *   View raw and processed thermograms.
    *   Adjust baseline correction parameters.
    *   Compare multiple samples.
    *   Explore the interactive plots and tables.

    For a detailed overview of the interface elements, see the :doc:`interface` guide.

Further Steps
-------------

This quickstart covers the basic steps. Explore the other sections of the User Guide for more in-depth information on features like baseline correction, normalization, and data comparison. 