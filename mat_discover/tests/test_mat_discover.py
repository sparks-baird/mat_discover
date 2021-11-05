"""
Test DISCOVER algorithm.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets
     - high mean cluster target/high fraction of validation points within cluster
"""
# %% imports
import pandas as pd

from crabnet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover

# %% Test Functions
def test_mat_discover():
    """Perform a simple run of mat_discover to ensure it runs without errors.

    This does not involve checking to verify that the output is correct, and
    additionally it uses a `dummy_run` and `dummy` such that MDS is used on a small
    dataset rather than UMAP on a large dataset (for faster runtimes).
    """
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
