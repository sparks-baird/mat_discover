"""Suggest next best experiment in the DiSCoVeR sense."""
# %% imports
from crabnet.data.materials_data import elasticity
from mat_discover.utils.data import data
from mat_discover.utils.Timer import Timer
from mat_discover.utils.extraordinary import extraordinary_split
from mat_discover.adaptive_design import Adapt

# %% setup
train_df, val_df = data(elasticity, "train.csv", dummy=False)
train_df, val_df, extraordinary_thresh = extraordinary_split(train_df, val_df)

# set dummy_run to True for a quicker run --> small dataset, MDS instead of UMAP
dummy_run = True
if dummy_run:
    val_df = val_df.iloc[:100]
adapt = Adapt(train_df, val_df, dummy_run=dummy_run, device="cuda")

# first experiment, which is more time-intensive
with Timer("First-Experiment"):
    first_experiment = adapt.suggest_first_experiment()
    print(adapt.dens_score_df.head(10))

# second experiment, which reuses the DensMAP densities
with Timer("Second-Experiment"):
    second_experiment = adapt.suggest_next_experiment()
    print(adapt.dens_score_df.head(10))

# third experiment, which reuses the DensMAP densities
with Timer("Third-Experiment"):
    third_experiment = adapt.suggest_next_experiment()
    print(adapt.dens_score_df.head(10))

print(
    [
        experiment["formula"]
        for experiment in [first_experiment, second_experiment, third_experiment]
    ]
)

1 + 1
