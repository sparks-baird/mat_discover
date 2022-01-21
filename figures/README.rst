.. role:: raw-html-m2r(raw)
   :format: html



Various figures, both interactive and non-interactive, related to the DiSCoVeR algorithm as applied to compounds and clusters. For more details, see `the ChemRXiv paper <https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8>`_.

Compound-wise and cluster-wise interactive figures produced via `Plotly <https://plotly.com/python/>`_.

Hover your mouse over datapoints to see more information (e.g. composition, bulk
modulus, cluster ID). Alternatively, you can download the ``html`` file at
`mat_discover/figures <https://github.com/sparks-baird/mat_discover/figures>`_ and open
it in your browser of choice.

Compound-Wise
-------------

Pareto Front Training Contribution to Validation Log Density Proxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html
   :file: pf-train-contrib-proxy.html

Pareto Front k-Nearest Neighbor Average Proxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html
   :file: pf-peak-proxy.html

Pareto Front Density Proxy (Train and Validation Points)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html
   :file: pf-dens-proxy.html

Cluster Properties
------------------

Cluster Count Histogram
^^^^^^^^^^^^^^^^^^^^^^^

\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://github.com/sparks-baird/mat_discover/blob/main/figures/cluster-count-hist.png?raw=true" alt="Cluster Count Histogram" width=350>`

DensMAP Embeddings Colored by Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html
   :file: px-umap-cluster-scatter.html

Cluster-Wise
------------

Pareto Front Validation Fraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html
   :file: pf-frac-proxy.html

Leave-one-cluster-out Cross-validation k-Nearest Neighbor Average Parity Plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html
   :file: gcv-pareto.html

Density and Target Characteristics
----------------------------------

Density Scatter Plot of 2D DensMAP Embeddings\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://github.com/sparks-baird/mat_discover/blob/main/figures/dens-scatter.png?raw=true" alt="Density Scatter Plot of 2D DensMAP Embeddings" width=350>`


Target Scatter Plot of 2D DensMAP Embeddings\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://github.com/sparks-baird/mat_discover/blob/main/figures/target-scatter.png?raw=true" alt="Density Scatter Plot of 2D DensMAP Embeddings" width=350>`

Density Scatter Plot with Bulk Modulus Overlay in 2D DensMAP Embedding Space\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://github.com/sparks-baird/mat_discover/blob/main/figures/dens-targ-scatter.png?raw=true" alt="Density Scatter Plot with Bulk Modulus Overlay in 2D DensMAP Embedding Space" width=350>`

Adaptive Design Comparison
--------------------------

Take an initial training set of 100 chemical formulas and associated Materials Project bulk moduli followed by 900 adaptive design iterations (x-axis) using random search, novelty-only (performance weighted at 0), a 50/50 weighting split, and performance-only (novelty weighted at 0). These are the columns. The rows are the total number of observed "extraordinary" compounds (top 2%), the total number of additional unique atoms, and total number of additional unique chemical formulae templates. These are the rows. In other words:

- How many "extraordinary" compounds have been observed so far?
- How many unique atoms have been explored so far? (not counting atoms already in the starting 100 formulas)
- How many unique chemical templates (e.g. A2B3, ABC, ABC2) have been explored so far? (not counting templates already in the starting 100 formulas)

Random search is compared with DiSCoVeR performance/proxy weights. The 50/50 weighting split offers a good trade-off between performance and novelty.

.. raw:: html
   :file: ad-compare.html
