"""Train DiSCoVeR using the `sklearn` novelty detector: LocalOutlierFactor."""
# %% imports
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from crabnet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover

# %% setup
# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy = True
# set gcv to False for a quicker run --> group-cross validation can take a while
gcv = False
disc = Discover(
    dummy_run=dummy, novelty_learner=LocalOutlierFactor(novelty=True), device="cuda"
)
train_df, val_df = disc.data(elasticity, fname="train.csv", dummy=dummy)
cat_df = pd.concat((train_df, val_df), axis=0)

# %% fit
disc.fit(train_df)

# %% predict
score = disc.predict(val_df)

# %% leave-one-cluster-out cross-validation
if gcv:
    disc.group_cross_val(cat_df)
    print("scaled test error = ", disc.scaled_error)

# %% plot and save
# disc.plot() # plotting not compatible for non-DiSCoVeR learners, raise GitHub issue if
# this is a desired functionality: https://github.com/sparks-baird/mat_discover/issues/new/choose
disc.save(dummy=dummy)

print(disc.dens_score_df.head(100)[["formula", "prediction", "density"]])

1 + 1
