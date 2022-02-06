"""A self-contained, bare-bones example of the DiSCoVeR algorithm.

Steps
-----
1. Load some data
2. CrabNet target predictions
3. ElM2D distance calculations
4. DensMAP embeddings and densities
5. Train contribution to validation density
6. Nearest neighbor properties
7. Calculation of weighted scores
"""
# %% Imports
from operator import attrgetter

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

import umap

from chem_wasserstein.ElM2D_ import ElM2D
from crabnet.train_crabnet import get_model
from crabnet.data.materials_data import elasticity
from crabnet.model import data

# %% 1. Data
train_df, val_df = data(elasticity, fname="train.csv")

# %% 2. CrabNet predictions
crabnet_model = get_model(train_df=train_df)

train_true, train_pred, _, train_sigma = crabnet_model.predict(train_df)

val_true, val_pred, _, val_sigma = crabnet_model.predict(val_df)

pred = np.concatenate((train_pred, val_pred), axis=0)

val_rmse = mean_squared_error(val_true, val_pred, squared=False)

print("val RMSE: ", val_rmse)

# %% Setup
train_formula = train_df["formula"]
train_target = train_df["target"]
val_formula = val_df["formula"]
val_target = val_df["target"]

all_formula = pd.concat((train_formula, val_formula), axis=0)
all_target = pd.concat((train_target, val_target), axis=0)

ntrain, nval = len(train_formula), len(val_formula)
ntot = ntrain + nval
train_ids, val_ids = np.arange(ntrain), np.arange(ntrain, ntot)

# %% 3. Distance calculations
mapper = ElM2D()
mapper.fit(all_formula)
dm = mapper.dm

# %% 4. DensMAP embeddings and densities
umap_trans = umap.UMAP(
    densmap=True,
    output_dens=True,
    dens_lambda=1.0,
    n_neighbors=30,
    min_dist=0,
    n_components=2,
    metric="precomputed",
    random_state=42,
    low_memory=False,
).fit(dm)


# Extract densMAP embedding and radii
umap_emb, r_orig_log, r_emb_log = attrgetter("embedding_", "rad_orig_", "rad_emb_")(
    umap_trans
)
umap_r_orig = np.exp(r_orig_log)

# %% 5. Train contribution to validation density
train_emb = umap_emb[:ntrain]
train_r_orig = umap_r_orig[:ntrain]
val_emb = umap_emb[ntrain:]
val_r_orig = umap_r_orig[ntrain:]

train_df["emb"] = list(map(tuple, train_emb))
train_df["r_orig"] = train_r_orig
val_df["emb"] = list(map(tuple, val_emb))
val_df["r_orig"] = val_r_orig


def my_mvn(mu_x, mu_y, r):
    """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
    return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])


mvn_list = list(map(my_mvn, train_emb[:, 0], train_emb[:, 1], train_r_orig))
pdf_list = [mvn.pdf(val_emb) for mvn in mvn_list]
val_dens = np.sum(pdf_list, axis=0)
val_log_dens = np.log(val_dens)

val_df["dens"] = val_dens

# %% 6. Nearest neighbor calculations
r_strength = 1.5
mean, std = (np.mean(dm), np.std(dm))
radius = mean - r_strength * std
n_neighbors = 10
NN = NearestNeighbors(radius=radius, n_neighbors=n_neighbors, metric="precomputed")
NN.fit(dm)

neigh_ind = NN.kneighbors(return_distance=False)
num_neigh = n_neighbors * np.ones(neigh_ind.shape[0])

neigh_target = np.array([pred[ind] for ind in neigh_ind], dtype="object")
k_neigh_avg_targ = np.array(
    [np.mean(t) if len(t) > 0 else float(0) for t in neigh_target]
)

val_k_neigh_avg = k_neigh_avg_targ[val_ids]

# %% 7. Weighted scores
def weighted_score(pred, proxy, pred_weight=1.0, proxy_weight=1.0):
    """Calculate weighted discovery score using the predicted target and proxy."""
    pred = pred.ravel().reshape(-1, 1)
    proxy = proxy.ravel().reshape(-1, 1)
    # Scale and weight the cluster data
    pred_scaler = RobustScaler().fit(pred)
    pred_scaled = pred_weight * pred_scaler.transform(pred)
    proxy_scaler = RobustScaler().fit(-1 * proxy)
    proxy_scaled = proxy_weight * proxy_scaler.transform(-1 * proxy)

    # combined cluster data
    comb_data = pred_scaled + proxy_scaled
    comb_scaler = RobustScaler().fit(comb_data)

    # cluster scores range between 0 and 1
    score = comb_scaler.transform(comb_data).ravel()
    return score


peak_score = weighted_score(val_pred, val_k_neigh_avg)
dens_score = weighted_score(val_pred, val_dens)
