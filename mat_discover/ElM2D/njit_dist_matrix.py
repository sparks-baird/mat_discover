"""
Nopython version of dist_matrix.

Author: Sterling Baird
"""
import os

# import json
import numpy as np

from numba import prange, njit
from numba.types import int32, float32, int64, float64

# from mat_discover.ElM2D.metrics import njit_wasserstein_distance as wasserstein_distance
# from mat_discover.ElM2D.metrics import euclidean_distance
from mat_discover.ElM2D.cpu_metrics import wasserstein_distance, euclidean_distance

# settings
inline = os.environ.get("INLINE", "never")
fastmath = bool(os.environ.get("FASTMATH", "1"))
cols = os.environ.get("COLUMNS")
USE_64 = bool(os.environ.get("USE_64", "0"))
target = "cpu"

if USE_64:
    bits = 64
    nb_float = float64
    nb_int = int64
    np_float = np.float64
    np_int = np.int64
else:
    bits = 32
    nb_float = float32
    nb_int = int32
    np_float = np.float32
    np_int = np.int32

fastmath = True
parallel = True
debug = False


# def myjit(f):
#     """
#     f : function
#     Decorator to assign the right jit for different targets
#     In case of non-cuda targets, all instances of `cuda.local.array`
#     are replaced by `np.empty`. This is a dirty fix, hopefully in the
#     near future numba will support numpy array allocation and this will
#     not be necessary anymore
#     Source: https://stackoverflow.com/a/47039836/13697228
#     """
#     if target == "cuda":
#         return cuda.jit(f, device=True)
#     elif target == "cpu":
#         source = inspect.getsource(f).splitlines()
#         source = "\n".join(source[1:]) + "\n"
#         source = source.replace("cuda.local.array", "np.empty")
#         exec(source)
#         fun = eval(f.__name__)
#         newfun = jit(fun, nopython=True)
#         # needs to be exported to globals
#         globals()[f.__name__] = newfun
#         return newfun


# wasserstein_distance = myjit(njit_wasserstein_distance.py_func)

# wasserstein_distance = njit(
#     wasserstein_distance.py_func, fastmath=fastmath, debug=debug
# )

# # TODO: switch to the more hard-coded version (faster than the NumPy functions)
# @njit(fastmath=fastmath, debug=debug)
# def cdf_distance(
#     u_values,
#     v_values,
#     u_weights,
#     v_weights,
#     p,
#     presorted,
#     cumweighted,
#     prepended,
# ):
#     r"""
#     Compute first Wasserstein distance.

#     Compute, between two one-dimensional distributions :math:`u` and
#     :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
#     statistical distance that is defined as:
#     .. math::
#         l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
#     p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
#     gives the energy distance.

#     Source:
#         https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L8404

#     Parameters
#     ----------
#     u_values, v_values : array_like
#         Values observed in the (empirical) distribution.
#     u_weights, v_weights : array_like, optional
#         Weight for each value. If unspecified, each value is assigned the same
#         weight.
#         `u_weights` (resp. `v_weights`) must have the same length as
#         `u_values` (resp. `v_values`). If the weight sum differs from 1, it
#         must still be positive and finite so that the weights can be normalized
#         to sum to 1.

#     Returns
#     -------
#     distance : float
#         The computed distance between the distributions.

#     Notes
#     -----
#     The input distributions can be empirical, therefore coming from samples
#     whose values are effectively inputs of the function, or they can be seen as
#     generalized functions, in which case they are weighted sums of Dirac delta
#     functions located at the specified values.

#     References
#     ----------
#     .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
#             Munos "The Cramer Distance as a Solution to Biased Wasserstein
#             Gradients" (2017). :arXiv:`1705.10743`.
#     """
#     if not presorted:
#         u_sorter = np.argsort(u_values)
#         v_sorter = np.argsort(v_values)

#         u_values = u_values[u_sorter]
#         v_values = v_values[v_sorter]

#         u_weights = u_weights[u_sorter]
#         v_weights = v_weights[v_sorter]

#     all_values = np.concatenate((u_values, v_values))
#     # all_values.sort(kind='mergesort')
#     all_values.sort()

#     # Compute the differences between pairs of successive values of u and v.
#     deltas = np.diff(all_values)

#     # Get the respective positions of the values of u and v among the values of
#     # both distributions.
#     u_cdf_indices = np.searchsorted(u_values, all_values[:-1], side="right")
#     v_cdf_indices = np.searchsorted(v_values, all_values[:-1], side="right")

#     zero = np.array([0])
#     # Calculate the CDFs of u and v using their weights, if specified.
#     if u_weights is None:
#         u_cdf = u_cdf_indices / u_values.size
#     else:
#         uw_cumsum = np.cumsum(u_weights)
#         u_sorted_cumweights = np.concatenate((zero, uw_cumsum))
#         u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

#     if v_weights is None:
#         v_cdf = v_cdf_indices / v_values.size
#     else:
#         vw_cumsum = np.cumsum(v_weights)
#         v_sorted_cumweights = np.concatenate((zero, vw_cumsum))
#         v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

#     # Compute the value of the integral based on the CDFs.
#     # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
#     # of about 15%.
#     if p == 1:
#         return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
#     if p == 2:
#         return np.sqrt(np.sum(np.multiply(np.power(u_cdf - v_cdf, 2), deltas)))
#     return np.power(
#         np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), deltas)), 1 / p
#     )


# @njit(fastmath=fastmath, debug=debug)
# def wasserstein_distance(
#     u,
#     v,
#     u_weights,
#     v_weights,
#     presorted,
#     cumweighted,
#     prepended,
# ):
#     r"""
#     Compute the first Wasserstein distance between two 1D distributions.

#     This distance is also known as the earth mover's distance, since it can be
#     seen as the minimum amount of "work" required to transform :math:`u` into
#     :math:`v`, where "work" is measured as the amount of distribution weight
#     that must be moved, multiplied by the distance it has to be moved.

#     Source
#     ------
#     https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L8245-L8319 # noqa

#     Parameters
#     ----------
#     u_values, v_values : array_like
#         Values observed in the (empirical) distribution.
#     u_weights, v_weights : array_like, optional
#         Weight for each value. If unspecified, each value is assigned the same
#         weight.
#         `u_weights` (resp. `v_weights`) must have the same length as
#         `u_values` (resp. `v_values`). If the weight sum differs from 1, it
#         must still be positive and finite so that the weights can be normalized
#         to sum to 1.

#     Returns
#     -------
#     distance : float
#         The computed distance between the distributions.

#     Notes
#     -----
#     The input distributions can be empirical, therefore coming from samples
#     whose values are effectively inputs of the function, or they can be seen as
#     generalized functions, in which case they are weighted sums of Dirac delta
#     functions located at the specified values.

#     References
#     ----------
#     .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
#            Munos "The Cramer Distance as a Solution to Biased Wasserstein
#            Gradients" (2017). :arXiv:`1705.10743`.
#     """
#     return cdf_distance(
#         u,
#         v,
#         u_weights,
#         v_weights,
#         1,
#         presorted=False,
#         cumweighted=False,
#         prepended=False,
#     )


# @njit(fastmath=fastmath, debug=debug)
# def euclidean_distance(a, b):
#     """
#     Calculate Euclidean distance between vectors a and b.

#     Parameters
#     ----------
#     a : 1D array
#         First vector.
#     b : 1D array
#         Second vector.

#     Returns
#     -------
#     d : numeric scalar
#         Euclidean distance between vectors a and b.
#     """
#     d = 0
#     for i in range(len(a)):
#         d += (b[i] - a[i]) ** 2
#     d = np.sqrt(d)
#     return d


# @njit(fastmath=fastmath, debug=debug)
@njit(
    "float{0}(float{0}[:], float{0}[:], float{0}[:], float{0}[:], int{0})".format(bits),
    fastmath=fastmath,
    debug=debug,
)
def compute_distance(u, v, u_weights, v_weights, metric_num):
    """
    Calculate weighted distance between two vectors, u and v.

    Parameters
    ----------
    u : 1D array of float
        First vector.
    v : 1D array of float
        Second vector.
    u_weights : 1D array of float
        Weights for u.
    v_weights : 1D array of float
        Weights for v.
    metric_num : int
        Which metric to use (0 == "euclidean", 1=="wasserstein").

    Raises
    ------
    NotImplementedError
        "Specified metric is mispelled or has not been implemented yet.
        If not implemented, consider submitting a pull request."

    Returns
    -------
    d : float
        Weighted distance between u and v.

    """
    if metric_num == 0:
        # d = np.linalg.norm(vec - vec2)
        d = euclidean_distance(u, v)
    elif metric_num == 1:
        # d = my_wasserstein_distance(vec, vec2)
        # d = wasserstein_distance(
        #     u, v, u_weights=u_weights, v_weights=v_weights, p=1, presorted=True
        # )
        d = wasserstein_distance(u, v, u_weights, v_weights, True, True, True)
    else:
        raise NotImplementedError(
            "Specified metric is mispelled or has not been implemented yet. \
                If not implemented, consider submitting a pull request."
        )
    return d


# @njit(fastmath=fastmath, parallel=parallel, debug=debug)
@njit(
    "void(float{0}[:,:], float{0}[:,:], float{0}[:,:], float{0}[:,:], int{0}[:,:], float{0}[:], int{0})".format(
        bits
    ),
    fastmath=fastmath,
    parallel=parallel,
    debug=debug,
)
def sparse_distance_matrix(U, V, U_weights, V_weights, pairs, out, metric_num):
    """
    Calculate sparse pairwise distances between two sets of vectors for pairs.

    Parameters
    ----------
    mat : numeric cuda array
        First set of vectors for which to compute a single pairwise distance.
    mat2 : numeric cuda array
        Second set of vectors for which to compute a single pairwise distance.
    pairs : cuda array of 2-tuples
        All pairs for which distances are to be computed.
    out : numeric cuda array
        The initialized array which will be populated with distances.

    Raises
    ------
    ValueError
        Both matrices should have the same number of columns.

    Returns
    -------
    None.

    """
    npairs = pairs.shape[0]

    for k in prange(npairs):
        pair = pairs[k]
        i, j = pair

        u = U[i]
        v = V[j]
        uw = U_weights[i]
        vw = V_weights[j]

        d = compute_distance(u, v, uw, vw, metric_num)
        out[k] = d


# @njit(fastmath=fastmath, parallel=parallel, debug=debug)
@njit(
    "void(float{0}[:,:], float{0}[:,:], float{0}[:,:], int{0})".format(bits),
    fastmath=fastmath,
    parallel=parallel,
    debug=debug,
)
def one_set_distance_matrix(U, U_weights, out, metric_num):
    """
    Calculate pairwise distances within single set of vectors.

    Parameters
    ----------
    U : 2D array of float
        Vertically stacked vectors.
    U_weights : 2D array of float
        Vertically stacked weight vectors.
    out : 2D array of float
        Initialized matrix to populate with pairwise distances.
    metric_num : int
        Which metric to use (0 == "euclidean", 1=="wasserstein").

    Returns
    -------
    None.

    """
    dm_rows = U.shape[0]
    dm_cols = U.shape[0]

    for i in prange(dm_rows):
        for j in range(dm_cols):
            if i < j:
                u = U[i]
                v = U[j]
                uw = U_weights[i]
                vw = U_weights[j]
                d = compute_distance(u, v, uw, vw, metric_num)
                out[i, j] = d
                out[j, i] = d


# faster compilation *and* runtimes with explicit signature (tested on cuda.jit)
# @njit(fastmath=fastmath, parallel=parallel, debug=debug)
@njit(
    "void(float{0}[:,:], float{0}[:,:], float{0}[:,:], float{0}[:,:], float{0}[:,:], int{0})".format(
        bits
    ),
    fastmath=fastmath,
    parallel=parallel,
    debug=debug,
)
def two_set_distance_matrix(U, V, U_weights, V_weights, out, metric_num):
    """Calculate distance matrix between two sets of vectors."""
    dm_rows = U.shape[0]
    dm_cols = V.shape[0]

    for i in prange(dm_rows):
        for j in range(dm_cols):
            u = U[i]
            v = V[j]
            uw = U_weights[i]
            vw = V_weights[j]
            d = compute_distance(u, v, uw, vw, metric_num)
            out[i, j] = d


def dist_matrix(
    U, V=None, U_weights=None, V_weights=None, pairs=None, metric="euclidean"
):
    """
    Compute pairwise distances using Numba/CUDA.

    Parameters
    ----------
    mat : array
        First set of vectors for which to compute pairwise distances.

    mat2 : array, optional
        Second set of vectors for which to compute pairwise distances. If not specified,
        then mat2 is a copy of mat.

    pairs : array, optional
        List of 2-tuples which contain the indices for which to compute distances for.
        If mat2 was specified, then the second index accesses mat2 instead of mat.
        If not specified, then the pairs are auto-generated. If mat2 was specified,
        all combinations of the two vector sets are used. If mat2 isn't specified,
        then only the upper triangle (minus diagonal) pairs are computed.

    metric : str, optional
        Possible options are 'euclidean', 'wasserstein'.
        Defaults to Euclidean distance. These are converted to integers internally
        due to Numba's lack of support for string arguments (2021-08-14).
        See compute_distance() for other keys. For example, 0 corresponds to Euclidean
        distance and 1 corresponds to Wasserstein distance.

    Returns
    -------
    out : array
        A pairwise distance matrix, or if pairs are specified, then a vector of
        distances corresponding to the pairs.

    """
    # is it distance matrix between two sets of vectors rather than within a single set?
    isXY = V is not None

    # were pairs specified? (useful for sparse matrix generation)
    pairQ = pairs is not None

    # assign metric_num based on specified metric (Numba doesn't support strings)
    metric_dict = {"euclidean": np_int(0), "wasserstein": np_int(1)}
    metric_num = metric_dict[metric]

    m = U.shape[0]

    if isXY:
        m2 = V.shape[0]
    else:
        m2 = m

    if pairQ:
        npairs = pairs.shape[0]
        shape = (npairs,)
    else:
        shape = (m, m2)

    # sorting and cumulative weights
    if metric == "wasserstein":
        # presort values (and weights by sorted value indices)
        U_sorter = np.argsort(U)
        U = np.take_along_axis(U, U_sorter, axis=-1)
        U_weights = np.take_along_axis(U_weights, U_sorter, axis=-1)

        # calculate cumulative weights
        U_weights = np.cumsum(U_weights, axis=1)

        # prepend a column of zeros
        zero = np.zeros((U_weights.shape[0], 1))
        U_weights = np.column_stack((zero, U_weights))

        # do the same for V and V_weights
        if isXY:
            V_sorter = np.argsort(V)
            V = np.take_along_axis(V, V_sorter, axis=-1)
            V_weights = np.take_along_axis(V_weights, V_sorter, axis=-1)
            V_weights = np.cumsum(V_weights, axis=1)
            V_weights = np.column_stack((zero, V_weights))

    out = np.zeros(shape, dtype=np_float)
    U = U.astype(np_float)
    if V is not None:
        V = V.astype(np_float)
    if U_weights is not None:
        U_weights = U_weights.astype(np_float)
    if V_weights is not None:
        V_weights = V_weights.astype(np_float)

    if isXY and not pairQ:
        # distance matrix between two sets of vectors
        two_set_distance_matrix(U, V, U_weights, V_weights, out, metric_num)

    elif not isXY and pairQ:
        # specified pairwise distances within single set of vectors
        sparse_distance_matrix(U, U, U_weights, U_weights, pairs, out, metric_num)

    elif not isXY and not pairQ:
        # distance matrix within single set of vectors
        one_set_distance_matrix(U, U_weights, out, metric_num)

    elif isXY and pairQ:
        # specified pairwise distances between two sets of vectors
        sparse_distance_matrix(U, V, U_weights, V_weights, pairs, out, metric_num)

    return out


# %% Code Graveyard
# from Lib import inspect

# with open("dist_matrix_settings.json", "r") as f:
#     settings = json.load(f)
# inline = settings.get("INLINE", "never")
# fastmath = settings.get("FASTMATH", True)
# cols = settings.get("COLUMNS")
# USE_64 = settings.get("USE_64", False)


# settings = {
#     "INLINE": "never",
#     "FASTMATH": True,
#     "COLUMNS": None,
#     "USE_64": False,
# }
# inline, fastmath, cols, USE_64 = [
#     settings.get(key, value) for key, value in settings.items()
# ]
