"""Load some data, fit Discover(), predict on validation data (three chunks), make some plots, and save the model."""
# %% imports
from os.path import join
import numpy as np
from crabnet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover

# %% instantiate
# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy = False
disc = Discover(dummy_run=dummy, dens_lambda=2, device="cuda", dist_device="cpu")

# %% data
train_df = disc.data(elasticity, "train.csv", split=False, dummy=dummy)
val_df = disc.data(elasticity, "val.csv", split=False, dummy=dummy)

# remove rows corresponding to formulas in val_df that overlap with train_df
indices = np.invert(np.in1d(val_df.formula, train_df.formula))
val_df = val_df.iloc[indices]

# split into 3 chunks
# HACK: "hardcoded" for less than ~100000 total compounds, and max 10000 training (mem)
val_df = val_df.sample(frac=1, random_state=42)
val_dfs = np.array_split(val_df, 3)
# val_df = val_df[:30000]

table_dirs = [join("tables", val) for val in ["val1", "val2", "val3"]]
figure_dirs = [join("figures", val) for val in ["val1", "val2", "val3"]]

# %% fit, predict, plot
disc.fit(train_df)
for i, val_df in enumerate(val_dfs):
    # change output dirs
    disc.table_dir = table_dirs[i]
    disc.figure_dir = figure_dirs[i]

    # %% predict
    score = disc.predict(val_df, umap_random_state=42)

    # %% plot and save
    disc.plot()
    disc.save(dummy=dummy)

# %% Code Graveyard
# set gcv to False for a quicker run --> group-cross validation can take a while
# gcv = False
#     # %% leave-one-cluster-out cross-validation
#     if gcv:
#         disc.group_cross_val(cat_df, umap_random_state=42)
#         print("scaled test error = ", disc.scaled_error)

# np.random.seed(42)
# val_df = np.random.shuffle(val_df)
