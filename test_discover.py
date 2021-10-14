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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from discover_ import Discover
from discover.utils.Timer import Timer
import pickle

dummy_run = False
# dummy_run = True
disc = Discover(dummy_run=dummy_run)

# load validation data
# HACK: absolute path while still working out dependency structure
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

# drop noble gases
# noble_ids = np.nonzero(np.isin(grp_df.formula, ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]))[0]
# grp_df.drop(noble_ids, inplace=True)
# take small subset
if dummy_run:
    n = 100
    n2 = 10
    train_df = grp_df.iloc[:n, :]
    val_df = grp_df.iloc[n : n + n2, :]
else:
    # REVIEW: consider changing train_size to 0.2 for cluster pareto plot
    test_size = 0.1
    val_size = 0.2
    tv_df, test_df = train_test_split(grp_df, test_size=test_size)
    train_df, val_df = train_test_split(tv_df, test_size=val_size / (1 - test_size))

# %% fit
# slower if umap_random_state is not None
with Timer("DISCOVER-fit"):
    disc.fit(train_df)

# %% predict
with Timer("DISCOVER-predict"):
    score = disc.predict(val_df, umap_random_state=42)

# %% plot
with Timer("DISCOVER-plot"):
    disc.plot()
1 + 1

# %% group-cv
# cat_df = pd.concat((train_df, val_df), axis=0)
# with Timer("DISCOVER-group-cross-val"):
#     disc.group_cross_val(cat_df)
# print("scaled test error = ", disc.scaled_error)

# %% compound rankings
print(disc.dens_score_df)
print(disc.peak_score_df)

disc.dens_score_df.rename(columns={"score": "Density Score"}, inplace=True)
disc.peak_score_df.rename(columns={"score": "Peak Score"}, inplace=True)

comb_score_df = disc.dens_score_df[["formula", "Density Score"]].merge(
    disc.peak_score_df[["formula", "Peak Score"]], how="outer"
)

comb_formula = disc.dens_score_df.head(10)[["formula"]].merge(
    disc.peak_score_df.head(10)[["formula"]], how="outer"
)

comb_out_df = comb_score_df[np.isin(comb_score_df.formula, comb_formula)]

disc.dens_score_df.head(10).to_csv("dens-score.csv", index=False, float_format="%.3f")
disc.peak_score_df.head(10).to_csv("peak-score.csv", index=False, float_format="%.3f")
comb_out_df.to_csv("comb-score.csv", index=False, float_format="%.3f")

# %% save
with open("disc.pkl", "wb") as f:
    pickle.dump(disc, f)

# %% CODE GRAVEYARD
# from os.path import join, expanduser
# "~",
# "Documents",
# "GitHub",
# "sparks-baird",
# "ElM2D",
1 + 1
