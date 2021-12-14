"""Suggest next best experiments in the DiSCoVeR sense."""
# %% imports
from crabnet.data.materials_data import elasticity
from mat_discover.utils.data import data
from mat_discover.utils.Timer import Timer
from mat_discover.utils.extraordinary import (
    extraordinary_split,
    extraordinary_histogram,
)
from mat_discover.adaptive_design import Adapt

# %% setup
train_df, val_df = data(elasticity, "train.csv", dummy=False)
train_df, val_df, extraordinary_thresh = extraordinary_split(train_df, val_df)
extraordinary_histogram(train_df, val_df)

# set dummy_run to True for a quicker run --> small dataset, MDS instead of UMAP
dummy_run = True
if dummy_run:
    val_df = val_df.iloc[:100]
adapt = Adapt(train_df, val_df, dummy_run=dummy_run, device="cuda")

with Timer("closed-loop-adaptive-design"):
    experiment_df = adapt.closed_loop_adaptive_design(n_experiments=10)

print(experiment_df)
1 + 1
