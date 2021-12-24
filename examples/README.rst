.. role:: raw-html-m2r(raw)
   :format: html

.. seealso::
    `Basic Usage <../../README.md#basic-usage>`_
        Import, instantiate, fit, predict, plot, save, and print top-10 ranked compounds
        in 7 lines. Instructions for importing custom datasets and using other built-in
        datasets (via `CrabNet.data.materials_data`) and downloading data from Materials
        Project.

Reproduce Paper Results
-----------------------
This simple example for running `mat_discover` was used to produce the figures in the first DiSCoVeR paper:

.. literalinclude:: ../../examples/mat_discover_example.py
    :caption: `/examples/mat_discover_example.py <https://github.com/sparks-baird/mat_discover/blob/main/examples/mat_discover_example.py>`_
    :language: python

Hyperparameter Combinations
---------------------------
You can test the effect of various hyperparameters (e.g. `Scaler`, `pred_weight`, and
`proxy_weight`) on the model outputs:

.. literalinclude:: ../../examples/hyperparameter_combinations.py
    :caption: `/examples/hyperparameter_combinations.py <https://github.com/sparks-baird/mat_discover/blob/main/examples/hyperparameter_combinations.py>`_
    :language: python

Varying these parameters (`Scaler`, `pred_weight`, and `proxy_weight`) affect the
unscaled target predictions nor the unscaled proxy values. These parameters only affect
how the scores/rankings are produced, and therefore do not affect the default plots that
are produced. The tables which contain the top-100 rankings for each hyperparameter
combination are given in `hyperparameter_combinations/
<https://github.com/sparks-baird/mat_discover/tree/main/examples/hyperparameter_combinations>`_

Without Dimension Reduction
---------------------------
By removing the UMAP dimensionality reduction step, about twice as many clusters are
produced, and the rate of unclassified points increases from ~5% to ~25%.

.. literalinclude:: ../../examples/no_dim_reduce_compare.py
    :caption: `/examples/no_dim_reduce_compare.py <https://github.com/sparks-baird/mat_discover/blob/main/examples/no_dim_reduce_compare.py>`_
    :language: python

Real Predictions
----------------
In other words, moving past the validation study into a practical use-case. The
practical use-case shown here takes the Materials Project compounds which contain
elasticity data and those which do not as training and validation data, respectively.
This is a work-in-progress.

.. literalinclude:: ../../examples/real_predictions.py
    :caption: `/examples/real_predictions.py <https://github.com/sparks-baird/mat_discover/blob/main/examples/real_predictions.py>`_
    :language: python

Adaptive Design Comparison
--------------------------
Use the `Adapt()` class to perform adaptive design for several hyperparameter
combinations and compare against random search.

.. literalinclude:: ../../examples/adaptive_design_compare.py
    :caption: `/examples/adaptive_design_compare.py <https://github.com/sparks-baird/mat_discover/blob/main/examples/adaptive_design_compare.py>`_
    :language: python

Bare Bones
----------
Looking to implement DiSCoVeR yourself or better understand it, but without all the
Python class trickery and imports from multiple files?

.. literalinclude:: ../../examples/bare_bones.py
    :caption: `/examples/bare_bones.py <https://github.com/sparks-baird/mat_discover/blob/main/examples/bare_bones.py>`_
    :language: python