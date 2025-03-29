SplineFitter Analysis
====================

Comparison with R's smooth.spline
--------------------------------

This document analyzes the current implementation of ``SplineFitter`` in ``thermogram_baseline/spline_fitter.py`` compared to R's ``smooth.spline`` function to ensure mathematical equivalence.

Algorithm Comparison
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Aspect
     - R smooth.spline
     - Current SplineFitter
     - Notes
   * - GCV Formula
     - RSS/(n*(1-df/n)²)
     - Same formula but with different df calculation
     - Python implementation approximates df based on spline coefficients
   * - DF Calculation
     - Based on trace of hat matrix
     - Approximated using number of spline coefficients
     - This approximation may cause differences in optimization
   * - Optimization
     - Uses an efficient algorithm to optimize spar (smoothing parameter)
     - Uses scipy.optimize.minimize with Nelder-Mead method
     - R's optimization may be more robust for certain datasets
   * - Spar Parameter
     - Scale from 0-1, then transform to λ
     - Maps between spar and scipy's s parameter using approximate relationship
     - Mapping is heuristic and does not exactly match R's transformation

Key Findings
~~~~~~~~~~

1. **Degrees of Freedom (df) Calculation**: The most significant difference is in how degrees of freedom are calculated. R uses the trace of the hat matrix, which is more accurate but computationally intensive. Our Python implementation approximates this based on the number of spline coefficients, which may lead to different smoothing parameter selection.

2. **Spar to Smoothing Parameter Mapping**: The relationship between R's ``spar`` parameter and scipy's ``s`` parameter is approximated. In R, the transformation is:

   .. code-block:: none

      λ = ratio * 16^(6*spar - 2)

   where ``ratio`` is the ratio of tr(X'WX) to tr(Ω). Our mapping is simpler and doesn't precisely capture this relationship.

3. **Spline Knot Selection**: R adaptively selects knots based on the number of unique x-values. Our implementation tries to replicate this but the algorithm isn't exactly the same, which can affect the resulting spline.

4. **Cross-Validation Implementation**: R's GCV and LOOCV implementations are highly optimized and leverage mathematical properties of splines. Our implementation uses a more direct approach that may be less numerically stable.

Recommended Changes
~~~~~~~~~~~~~~~~~

1. **Improve DF Calculation**: Implement a more accurate calculation of degrees of freedom based on the trace of the smoothing matrix, which would more closely match R's behavior.

2. **Refine Spar Parameter Mapping**: Develop a more precise mapping between R's ``spar`` parameter and scipy's smoothing parameter, considering the proper scaling factors.

3. **Enhance Knot Selection**: Implement a knot selection algorithm that more closely matches R's behavior, particularly for large datasets.

4. **Robust Cross-Validation**: Improve the cross-validation implementation to handle edge cases better and enhance numerical stability.

5. **Direct R Integration Option**: Keep the option to use R directly through rpy2, as this guarantees exact equivalence for critical applications.

Performance Comparison
-------------------

Initial testing shows that the Python implementation is generally faster for small to medium datasets, but R may have an advantage for very large datasets due to its optimized algorithms. More comprehensive benchmarking is needed.

Validation Testing
--------------

Current validation tests show that the Python implementation typically achieves relative differences less than 1% compared to R's output for most datasets, but can go up to 5-10% for challenging cases (e.g., very noisy data).

Additional test cases should be developed to cover edge cases such as:

- Datasets with very few points
- Data with extreme values or outliers
- Very noisy data
- Data with exact duplicate x-values