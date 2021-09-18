"""
Test DISCOVER algorithm.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets

Run using elm2d_ environment.

# Stick with ~10k elasticity datapoints
# Perform UMAP and HDBSCAN
for cluster in clusters:
    for i in [0, 1, 5, 10]:
        n = ceil(len(cluster) - i, 0)
        clustertmp = clusters
        remove cluster[0:n] from clustertmp
        train crabnet
        predict on removed cluster, store MAE
        calculate MAE for targets above some threshold
        test distance to pareto front for "kept" (?) cluster points for various metrics
        calculate the distribution of distances to the pareto front
# FIGURE: cluster-wise distributions of target values (6 clusters with highest (mean?) target value) 2x3 tile
# TODO: parameter - highest mean vs. highest single target value
# SUPP-FIGURE: cluster-wise distributions of target values for all clusters (Nx3)
# FIGURE: cluster-wise cross-val parity plots (6 clusters with highest (mean?) target value) - 2x3 tile
# SUPP-FIGURE: cluster-wise cross-val parity plots for all clusters (Nx3)

Created on Mon Sep 6 23:15:27 2021.

@author: sterg
"""
# %% Setup
# imports
from os.path import join, expanduser

import pandas as pd

from discover import Discover

disc = Discover()

# load validation data
data_dir = expanduser(
    join(
        "~",
        "Documents",
        "GitHub",
        "sparks-baird",
        "ElM2D",
        "CrabNet",
        "model_predictions",
    )
)
name = "elasticity_val_output.csv"  # "example_materials_property_val_output.csv"
fpath = join(data_dir, name)
val_df = pd.read_csv(fpath)

# take small subset
n = 10000
tmp_df = val_df.iloc[:n, :]
# %% fit
disc.fit(tmp_df)

# %% CODE GRAVEYARD
