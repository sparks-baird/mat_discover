"""Perform clustering without dimensionality reduction and check noise proportion."""
# %% imports
from operator import attrgetter
import numpy as np
import pandas as pd
from crabnet.data.materials_data import elasticity
import hdbscan
from chem_wasserstein.ElM2D_ import ElM2D
from mat_discover.mat_discover_ import Discover

# %% setup
# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy = False
# set gcv to False for a quicker run --> group-cross validation can take a while
gcv = False
disc = Discover(dummy_run=dummy, device="cuda")
train_df, val_df = disc.data(elasticity, "train.csv", dummy=dummy)
cat_df = pd.concat((train_df, val_df), axis=0)

all_formula = cat_df["formula"]
mapper = ElM2D()
mapper.fit(all_formula)
# https://github.com/scikit-learn-contrib/hdbscan/issues/71
dm = mapper.dm.astype(np.float64)

min_cluster_size = 50
min_samples = 1

clusterer = hdbscan.HDBSCAN(
    min_samples=min_samples,
    cluster_selection_epsilon=0.63,
    min_cluster_size=min_cluster_size,
    metric="precomputed",
).fit(dm)
labels, probabilities = attrgetter("labels_", "probabilities_")(clusterer)

# np.unique while preserving order https://stackoverflow.com/a/12926989/13697228
lbl_ids = np.unique(labels, return_index=True)[1]
unique_labels = [labels[lbl_id] for lbl_id in sorted(lbl_ids)]

nclusters = len(lbl_ids)
print("nclusters: ", nclusters)

class_ids = labels != -1
unclass_ids = np.invert(class_ids)
unclass_frac = np.sum(unclass_ids) / len(labels)

print("unclass_frac: ", unclass_frac)
