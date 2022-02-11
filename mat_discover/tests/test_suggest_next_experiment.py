"""Suggest next best experiment in the DiSCoVeR sense."""
# %% imports
import numpy as np
from crabnet.data.materials_data import elasticity
from mat_discover.utils.data import data
from mat_discover.utils.extraordinary import extraordinary_split
from mat_discover.adaptive_design import Adapt

# %% setup
train_df, val_df = data(elasticity, "train.csv", dummy=False)
train_df, val_df, extraordinary_thresh = extraordinary_split(train_df, val_df)

val_df = val_df.iloc[:100]
adapt = Adapt(train_df, val_df, dummy_run=False, device="cuda")

# first experiment, which is more time-intensive
first_experiment = adapt.suggest_first_experiment()
assert adapt.train_df.shape[0] == 101
assert adapt.val_df.shape[0] == 99
first_emb = first_experiment["emb"]
first_index = first_experiment["index"]

# next experiment, which reuses the DensMAP densities
second_experiment = adapt.suggest_next_experiment()
assert adapt.train_df.shape[0] == 102
assert adapt.val_df.shape[0] == 98
assert second_experiment["formula"] != first_experiment["formula"]
second_emb = second_experiment["emb"]
emb_check = adapt.train_df[adapt.train_df.index == first_index]["emb"]
print(emb_check)
assert emb_check.iloc[0] == first_emb
assert np.all(adapt.val_df.index != first_index)

# third experiment, which reuses the DensMAP densities
third_experiment = adapt.suggest_next_experiment()
assert adapt.train_df.shape[0] == 103
assert adapt.val_df.shape[0] == 97
assert third_experiment["formula"] != second_experiment["formula"]

1 + 1
