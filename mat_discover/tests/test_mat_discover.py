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
"""
# %% imports
import pandas as pd

from crabnet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover

# %% Test Functions
def test_mat_discover():
    disc = Discover(dummy_run=True)
    train_df, val_df = disc.data(elasticity, "train.csv", dummy=True)
    disc.fit(train_df)
    score = disc.predict(val_df, umap_random_state=42)
    cat_df = pd.concat((train_df, val_df), axis=0)
    disc.group_cross_val(cat_df, umap_random_state=42)
    print("scaled test error = ", disc.scaled_error)
    disc.plot()
    # disc.save() #doesn't work with pytest for some reason (pickle: object not the same)
    # disc.load()
