"""Load some data, fit Discover(), predict on validation data, and plot."""
# %% imports
from os.path import join
import pandas as pd
from crabnet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover
from mat_discover.utils.pareto import pareto_plot

# %% setup
# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy = False
disc = Discover(dummy_run=dummy, device="cuda")
train_df, val_df = disc.data(elasticity, fname="train.csv", dummy=dummy)
cat_df = pd.concat((train_df, val_df), axis=0)

# %% fit
disc.fit(train_df)

# %% predict
score = disc.predict(val_df, umap_random_state=42)

# Interactive scatter plot colored by clusters
x = "DensMAP Dim. 1"
y = "DensMAP Dim. 2"
umap_df = pd.DataFrame(
    {
        x: disc.std_emb[:, 0],
        y: disc.std_emb[:, 1],
        "cluster ID": disc.labels,
        "formula": disc.all_formula,
    }
)
fig = pareto_plot(
    umap_df,
    x=x,
    y=y,
    color="cluster ID",
    fpath=join(disc.figure_dir, "px-umap-cluster-scatter"),
    pareto_front=False,
    parity_type=None,
)
