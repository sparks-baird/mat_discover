"""Testing out supplying a custom metric directly to UMAP."""
# %% Imports
import os
from copy import deepcopy
from operator import attrgetter

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

import umap

# os.environ["COLUMNS"] = "100"
from chem_wasserstein.ElM2D_ import ElM2D
from chem_wasserstein.utils.Timer import Timer

# from dist_matrix.utils.cpu_metrics import wasserstein_distance
from crabnet.train_crabnet import get_model
from crabnet.data.materials_data import elasticity
from crabnet.model import data


def my_wasserstein_distance(u_uw, v_vw):
    """
    Return Earth Mover's distance using concatenated values and weights.

    Parameters
    ----------
    u_uw : 1D numeric array
        Horizontally stacked values and weights of first distribution.
    v_vw : TYPE
        Horizontally stacked values and weights of second distribution.

    Returns
    -------
    distance : numeric scalar
        Earth Mover's distance given two distributions.

    """
    # split into values and weights
    n = len(u_uw)
    i = n // 2
    u = u_uw[0:i]
    uw = u_uw[i:n]
    v = v_vw[0:i]
    vw = v_vw[i:n]
    # calculate distance
    distance = scipy_wasserstein_distance(u, v, u_weights=uw, v_weights=vw)
    return distance


def join_wasserstein(U, V, Uw, Vw):
    """
    Horizontally stack values and weights for each distribution.

    Weights are added as additional columns to values.

    Example:
        u_uw, v_vw = join_wasserstein(u, v, uw, vw)
        d = my_wasserstein_distance(u_uw, v_vw)
        cdist(u_uw, v_vw, metric=my_wasserstein_distance)

    Parameters
    ----------
    u : 1D or 2D numeric array
        First set of distribution values.
    v : 1D or 2D numeric array
        Second set of values of distribution values.
    uw : 1D or 2D numeric array
        Weights for first distribution.
    vw : 1D or 2D numeric array
        Weights for second distribution.

    Returns
    -------
    u_uw : 1D or 2D numeric array
        Horizontally stacked values and weights of first distribution.
    v_vw : TYPE
        Horizontally stacked values and weights of second distribution.

    """
    U_Uw = np.concatenate((U, Uw), axis=1)
    V_Vw = np.concatenate((V, Vw), axis=1)
    return U_Uw, V_Vw


# %% 1. Data
train_df, val_df = data(elasticity, fname="train.csv", dummy=True)

# %% 2. CrabNet predictions
crabnet_model = get_model(train_df=train_df, learningcurve=False)

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

mapper = ElM2D()
mapper.fit(all_formula)
dm = mapper.dm
U = mapper.U
V = deepcopy(U)
U_weights = mapper.U_weights
V_weights = deepcopy(U_weights)

U_Uw, V_Vw = join_wasserstein(U, V, U_weights, V_weights)


# %% 4. DensMAP embeddings and densities

with Timer("DensMAP, custom metric"):
    umap_trans = umap.UMAP(
        densmap=True,
        output_dens=True,
        dens_lambda=1.0,
        n_neighbors=30,
        min_dist=0,
        n_components=2,
        metric=my_wasserstein_distance,
        random_state=42,
        low_memory=False,
    ).fit(U_Uw)


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
