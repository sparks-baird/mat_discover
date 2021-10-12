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
from os.path import join
import pandas as pd
from discover_ import Discover

disc = Discover()

# load data
# HACK: relative path
data_dir = join("CrabNet", "data", "materials_data", "elasticity")
name = "train.csv"
fpath = join(data_dir, name)
df = pd.read_csv(fpath)

grp_df = (
    df.reset_index()
    .groupby(by="formula")
    .agg({"index": lambda x: tuple(x), "target": "max"})
    .reset_index()
)

# REVIEW: drop pure elements here?

# %% fit
# faster if umap_random_state is None
disc.fit(grp_df, umap_random_state=42)
# %% plot
disc.plot()
# %% CODE GRAVEYARD
# %% group-cv
disc.group_cross_val(grp_df)
print("scaled error = ", disc.scaled_error)

# this is just step 1, seeing which clusters are more favorable
# there could be some filtering of which ones are better based
# on the densities as well
# once clusters are selected, we can take the approach of filtering
# based on predicted target (and possibly density relative to the training
# clusters as well)
