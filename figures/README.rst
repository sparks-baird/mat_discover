.. role:: raw-html-m2r(raw)
   :format: html



Various figures, both interactive and non-interactive, related to the DiSCoVeR algorithm as applied to compounds and clusters. For more details, see `the ChemRXiv paper <https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8>`_.

Interactive Figures
-------------------

Compound-wise and cluster-wise interactive figures produced via `Plotly <https://plotly.com/python/>`_.

Hover your mouse over datapoints to see more information (e.g. composition, bulk
modulus, cluster ID). Alternatively, you can download the ``html`` file at
`mat_discover/figures <https://github.com/sparks-baird/mat_discover/figures>`_ and open
it in your browser of choice.

Compound-Wise
^^^^^^^^^^^^^

Pareto Front Training Contribution to Validation Log Density Proxy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html
   :file: pf-train-contrib-proxy.html

Pareto Front k-Nearest Neighbor Average Proxy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html
   :file: pf-peak-proxy.html

Pareto Front Density Proxy (Train and Validation Points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html
   :file: pf-dens-proxy.html

Cluster-Wise
^^^^^^^^^^^^

Pareto Front Validation Fraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html
   :file: pf-frac-proxy.html

Leave-one-cluster-out Cross-validation k-Nearest Neighbor Average Parity Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html
   :file: gcv-pareto.html

Non-interactive Figures
-----------------------

The following are non-interactive figures focused on cluster properties and density/target characteristics within DensMAP embedding space.

Cluster Properties
^^^^^^^^^^^^^^^^^^

Cluster Count Histogram\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://sparks-baird.github.io/mat_discover/figures/cluster-count-hist.png" alt="Cluster Count Histogram" width=350>`

DensMAP Embeddings Colored by Cluster\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://sparks-baird.github.io/mat_discover/figures/umap-cluster-scatter.png" alt="DensMAP Embeddings Colored by Cluster" width=350>`>

Density and Target Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Density Scatter Plot of 2D DensMAP Embeddings\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://sparks-baird.github.io/mat_discover/figures/dens-scatter.png" alt="Density Scatter Plot of 2D DensMAP Embeddings" width=350>`


Target Scatter Plot of 2D DensMAP Embeddings\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://sparks-baird.github.io/mat_discover/figures/target-scatter.png" alt="Density Scatter Plot of 2D DensMAP Embeddings" width=350>`

Density Scatter Plot with Bulk Modulus Overlay in 2D DensMAP Embedding Space\ :raw-html-m2r:`<br>`
:raw-html-m2r:`<img src="https://sparks-baird.github.io/mat_discover/figures/dens-targ-scatter.png" alt="Density Scatter Plot with Bulk Modulus Overlay in 2D DensMAP Embedding Space" width=350>`