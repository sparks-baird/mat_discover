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
    """Compute nearest neighbor properties for peak proxy using radius and kNN.

    Parameters
    ----------
    X : 2d array
        Pairwise distance matrix (within single set).
    target : 1d array
        Target property values.
    r_strength : float or None, optional
        Radius strength used as a scaling value for `radius`, by default None.
        If None, then a default value based on mean and standard deviation is used. See `_nearest_neigh_props`.
    radius : float, optional
        The radius within which to consider nearest neighbors, by default None
    n_neighbors : int, optional
        The number of nearest neighbors (kNNs) to consider for computing `k_neigh_avg_targ`, by default 10.
    metric : str or callable
        "The distance metric to use for the tree. The default metric is minkowski, and
        with p=2 is equivalent to the standard Euclidean metric. See the documentation
        of DistanceMetric for a list of available metrics. If metric is "precomputed", X
        is assumed to be a distance matrix and must be square during fit. X may be a
        sparse graph, in which case only "nonzero" elements may be considered
        neighbors." (source: `sklearn.neighbors.NearestNeighbors` docs). By default "precomputed".

    Returns
    -------
    rad_neigh_avg_targ, k_neigh_avg_targ : 1d array (X.shape[0],)
        Radius- and kNN-based average of neighbor targets, respectively.

    See Also
    --------
    sklearn.neighbors.NearestNeighbors : Unsupervised learner for implementing neighbor
    searches. Text source: `sklearn.neighbors.NearestNeighbors` docs
    """
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
    """Compute nearest neighbor properties.

    Parameters
    ----------
    X : 2d array
        Pairwise distance matrix (within single set).
    target : 1d array
        Target property values.
    type : str, optional
        [description], by default "radius"
    r_strength : float or None, optional
        Radius strength used as a scaling value for `radius`, by default None.
        If None, then a default value based on mean and standard deviation is used. See `_nearest_neigh_props`.
    radius : float, optional
        The radius within which to consider nearest neighbors, by default None
    n_neighbors : int, optional
        The number of nearest neighbors (kNNs) to consider for computing `k_neigh_avg_targ`, by default 10.
    "metric : str or callable
        The distance metric to use for the tree. The default metric is minkowski, and
        with p=2 is equivalent to the standard Euclidean metric. See the documentation
        of DistanceMetric for a list of available metrics. If metric is "precomputed", X
        is assumed to be a distance matrix and must be square during fit. X may be a
        sparse graph, in which case only "nonzero" elements may be considered
        neighbors." (source: `sklearn.neighbors.NearestNeighbors` docs). By default
        "precomputed".
    **NN_kwargs : `sklearn.neighbors.NearestNeighbors` keyword arguments.

    Returns
    -------
    neigh_target : ndarray with dtype=object
        Target properties of the neighbors.

    neigh_avg_targ : 1d array
        Average of neighbor targets for each compound.

    num_neigh : 1d array
        Number of neighbors for each compound.

    See Also
    --------
    sklearn.neighbors.NearestNeighbors : Unsupervised learner for implementing neighbor
    searches. Text source: `sklearn.neighbors.NearestNeighbors` docs
    """
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
