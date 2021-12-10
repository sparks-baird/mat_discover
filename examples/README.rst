.. role:: raw-html-m2r(raw)
   :format: html

Getting Started
---------------
This is a bare-bones example for running `mat_discover`:
.. literalinclude:: mat_discover_example.py
  :language: python

This is the file that was used to produce the figures in the first DiSCoVeR paper.

Hyperparameter Combinations
---------------------------
You can test the effect of various hyperparameters (e.g. `Scaler`, `pred_weight`, and
`proxy_weight`) on the model outputs:
.. literalinclude:: hyperparameter_combinations.py
  :language: python

Varying these parameters (`Scaler`, `pred_weight`, and `proxy_weight`) affect the
unscaled target predictions nor the unscaled proxy values. These parameters only affect
how the scores/rankings are produced, and therefore do not affect the default plots that
are produced. The tables which contain the top-100 rankings for each hyperparameter
combination are given in `hyperparameter_combinations/
<https://github.com/sparks-baird/mat_discover/tree/main/examples/hyperparameter_combinations>`_

Without Dimension Reduction
---------------------------
To test the effect of removing the UMAP dimensionality reduction step:
.. literalinclude:: no_dim_reduce_compare.py
  :language: python

About twice as many clusters are produced, and the rate of unclassified points increases
from ~5% to ~25%.

.. Various figures, both interactive and non-interactive, related to the DiSCoVeR algorithm as applied to compounds and clusters. For more details, see `the ChemRXiv paper <https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8>`_.