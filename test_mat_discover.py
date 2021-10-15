"""
Test DISCOVER algorithm.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets
     - high mean cluster target/high fraction of validation points within cluster

Run using mat_discover environment.

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
# import faulthandler

# retrieve static file from package: https://stackoverflow.com/a/20885799/13697228
from importlib.resources import open_text

import pandas as pd
from sklearn.model_selection import train_test_split

from mat_discover.CrabNet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover, groupby_formula
from mat_discover.utils.Timer import Timer

# Due to some issues with Plotly and Pytest (https://stackoverflow.com/a/65826036/13697228)
# faulthandler.disable()

# dummy_run = False
dummy_run = True
disc = Discover(dummy_run=dummy_run)

# load data

# this is "training data", but only in the context of a real study predicting on many new materials
# As a validation study, this is further split into train/val sets.
train_csv = open_text(elasticity, "train.csv")
df = pd.read_csv(train_csv)

# group identical compositions
grp_df = groupby_formula(df, how="max")

if dummy_run:
    n = 100
    n2 = 10
    train_df = grp_df.iloc[:n, :]
    val_df = grp_df.iloc[n : n + n2, :]
else:
    # REVIEW: consider changing train_size to 0.2 for cluster pareto plot
    # test_size = 0.1
    val_size = 0.2
    # tv_df, test_df = train_test_split(grp_df, test_size=test_size)
    # train_df, val_df = train_test_split(tv_df, test_size=val_size / (1 - test_size))
    train_df, val_df = train_test_split(grp_df, test_size=val_size)

# %% Test Functions


def test_fit():
    """Test fit method."""
    with Timer("DISCOVER-fit"):
        # supposedely slower if umap_random_state is not None
        disc.fit(train_df)


def test_predict():
    """Test predict method."""
    with Timer("DISCOVER-predict"):
        score = disc.predict(val_df, umap_random_state=42)


def test_group_cross_val():
    """Test leave-one-cluster-out cross-validation."""
    cat_df = pd.concat((train_df, val_df), axis=0)
    with Timer("DISCOVER-group-cross-val"):
        disc.group_cross_val(cat_df, umap_random_state=42)
    print("scaled test error = ", disc.scaled_error)


def test_plot():
    """Test plotting functions."""
    with Timer("DISCOVER-plot"):
        disc.plot()


def test_save():
    """Test saving the model to disc.pkl."""
    with Timer("DISCOVER-save"):
        disc.save()


def test_load():
    """Test loading the model from disc.pkl."""
    with Timer("DISCOVER-load"):
        disc.load()


def main(dummy_run=True, gcv=False):
    """Run through most, if not all the tests."""
    disc.dummy_run = dummy_run
    test_fit()
    test_predict()
    if gcv:
        test_group_cross_val()
    test_plot()
    test_save()
    test_load()


# faulthandler.enable()

if __name__ == "__main__":
    main()

# %% CODE GRAVEYARD
# from os.path import join, expanduser
# "~",
# "Documents",
# "GitHub",
# "sparks-baird",
# "ElM2D",
1 + 1

# drop pure elements?

# drop noble gases
# noble_ids = np.nonzero(np.isin(grp_df.formula, ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]))[0]
# grp_df.drop(noble_ids, inplace=True)
# take small subset

# data_dir = join("mat_discover", "CrabNet", "data", "materials_data", "elasticity")
# name = "train.csv"  # "example_materials_property_val_output.csv", #elasticity_val_output.csv"
# fpath = join(data_dir, name)
