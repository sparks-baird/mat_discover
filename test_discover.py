"""
Test DISCOVER algorithm.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets
     - high mean cluster target/high fraction of validation points within cluster

Run using discover environment.

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
from sklearn.model_selection import train_test_split
from discover_ import Discover
from discover.utils.Timer import Timer

dummy_run = False
disc = Discover(dummy_run=dummy_run)

# load validation data
# HACK: absolute path while stick working out dependency structure
data_dir = join("CrabNet", "data", "materials_data", "elasticity")
name = "train.csv"  # "example_materials_property_val_output.csv", #elasticity_val_output.csv"
fpath = join(data_dir, name)
df = pd.read_csv(fpath)

# df = df.groupby(by="formula", as_index=False).mean()
# if there are two compounds with the same formula, we're more interested in the higher GPa
group_filter = "max"  # "mean"
grp_df = (
    df.reset_index()
    .groupby(by="formula")
    .agg({"index": lambda x: tuple(x), "target": "max"})
    .reset_index()
)

# REVIEW: drop pure elements here?

# take small subset
if dummy_run:
    n = 100
    n2 = 10
    train_df = grp_df.iloc[:n, :]
    val_df = grp_df.iloc[n : n + n2, :]
else:
    train_df, val_df = train_test_split(grp_df, train_size=0.8)
# %% fit
# slower if umap_random_state is not None
with Timer("DISCOVER-fit"):
    disc.fit(train_df)
# %% predict
with Timer("DISCOVER-predict"):
    score = disc.predict(val_df)
# %% plot
with Timer("DISCOVER-plot"):
    disc.plot()
# %% group-cv
cat_df = pd.concat((train_df, val_df), axis=0)
with Timer("DISCOVER-group-cross-val"):
    disc.group_cross_val(cat_df)
print("scaled test error = ", disc.scaled_error)
# %% CODE GRAVEYARD
# from os.path import join, expanduser
# "~",
# "Documents",
# "GitHub",
# "sparks-baird",
# "ElM2D",
