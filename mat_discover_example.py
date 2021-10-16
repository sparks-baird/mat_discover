# %% imports
import pandas as pd
from mat_discover.CrabNet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover

# %% setup
disc = Discover(dummy_run=True)
train_df, val_df = disc.data(elasticity, "train.csv", dummy=True)
cat_df = pd.concat((train_df, val_df), axis=0)

# %% fit
disc.fit(train_df)

# %% predict
score = disc.predict(val_df, umap_random_state=42)

# %% leave-one-cluster-out cross-validation
# disc.group_cross_val(cat_df, umap_random_state=42)
# print("scaled test error = ", disc.scaled_error)

# %% plot and save
disc.plot()
disc.save()
