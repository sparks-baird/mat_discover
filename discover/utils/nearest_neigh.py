"""Nearest neighbor helper functions for DISCOVER."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def nearest_neigh_props(
    X,
    target,
    r_strength=None,
    radius=None,
    n_neighbors=10,
    metric="precomputed",
    **NN_kwargs,
):
    rad_neigh_avg_targ, num_neigh = _nearest_neigh_props(
        X,
        target,
        type="radius",
        r_strength=r_strength,
        radius=radius,
        n_neighbors=n_neighbors,
        metric=metric,
    )[1:3]

    k_neigh_avg_targ = _nearest_neigh_props(
        X,
        target,
        type="kneighbors",
        r_strength=r_strength,
        radius=radius,
        n_neighbors=n_neighbors,
        metric=metric,
    )[1]
    return rad_neigh_avg_targ, k_neigh_avg_targ


def _nearest_neigh_props(
    X,
    target,
    type="radius",
    r_strength=None,
    radius=None,
    n_neighbors=10,
    metric="precomputed",
    **NN_kwargs,
):
    if radius is None and metric == "precomputed":
        if r_strength is None:
            r_strength = 1.5
        mean, std = (np.mean(X), np.std(X))
        radius = mean - r_strength * std

    if n_neighbors > X.shape[0]:
        n_neighbors = X.shape[0]
    NN = NearestNeighbors(
        radius=radius, n_neighbors=n_neighbors, metric="precomputed", **NN_kwargs
    )
    NN.fit(X)
    if type == "radius":
        neigh_ind = NN.radius_neighbors(return_distance=False)
        num_neigh = np.array([len(ind) for ind in neigh_ind])
    elif type == "kneighbors":
        neigh_ind = NN.kneighbors(return_distance=False)
        num_neigh = n_neighbors * np.ones(neigh_ind.shape[0])

    neigh_target = np.array([target[ind] for ind in neigh_ind], dtype="object")
    neigh_avg_targ = np.array(
        [np.mean(t) if len(t) > 0 else float(0) for t in neigh_target]
    )

    return neigh_target, neigh_avg_targ, num_neigh
