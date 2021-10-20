"""
Compute a distance matrix for various cases.

Cases:
    within a single set of vectors (like pdist)
    between two sets of vectors (like cdist)
    between prespecified pairs (i.e. sparse) for either one or two sets of
        vectors.

Various distance metrics are available.
"""
import os
import numpy as np
from math import ceil
from .metrics import wasserstein_distance, euclidean_distance

from numba import cuda, jit  # noqa
from numba.types import int32, float32, int64, float64  # noqa

inline = os.environ.get("INLINE", "never")
fastmath = bool(os.environ.get("FASTMATH", "1"))
cols = os.environ.get("COLUMNS")
USE_64 = bool(os.environ.get("USE_64", "0"))
target = "cuda"


if USE_64 is None:
    USE_64 = False
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

# # override types
# if target == "cpu":
#     nb_float = np_float
#     nb_int = np_int

if cols is not None:
    cols = int(cols)
    cols_plus_1 = cols + 1
    tot_cols = cols * 2
    tot_cols_minus_1 = tot_cols - 1
else:
    raise KeyError(
        "For performance reasons and architecture constraints "
        "the number of columns of U (which is the same as V) "
        "must be defined as the environment variable, COLUMNS, "
        'via e.g. `os.environ["COLUMNS"] = "100"`.'
    )

# threads-per-block
TPB = 16

# TODO: explicit signature?
@cuda.jit(
    "float{0}(float{0}[:], float{0}[:], float{0}[:], float{0}[:], int{0})".format(bits),
    device=True,
    inline=inline,
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
        d = euclidean_distance(u, v)
    elif metric_num == 1:
        # assume u and v are presorted, and weights are sorted by u and v
        d = wasserstein_distance(u, v, u_weights, v_weights, True, True, True)
    else:
        raise NotImplementedError(
            "Specified metric is mispelled or has not been implemented yet. "
            "If not implemented, consider submitting a pull request."
        )
    return d


@cuda.jit(
    "void(float{0}[:,:], float{0}[:,:], float{0}[:,:], float{0}[:,:], int{0}[:,:], float{0}[:], int{0})".format(
        bits
    ),
    fastmath=fastmath,
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
    k = cuda.grid(1)

    npairs = pairs.shape[0]

    if k < npairs:
        pair = pairs[k]
        # extract indices of the pair for which the distance will be computed
        i, j = pair

        u = U[i]
        v = V[j]
        uw = U_weights[i]
        vw = V_weights[j]

        out[k] = compute_distance(u, v, uw, vw, metric_num)


@cuda.jit(
    "void(float{0}[:,:], float{0}[:,:], float{0}[:,:], int{0})".format(bits),
    fastmath=fastmath,
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
    i, j = cuda.grid(2)

    dm_rows = U.shape[0]
    dm_cols = U.shape[0]

    if i < j and i < dm_rows and j < dm_cols and i != j:
        u = U[i]
        v = U[j]
        uw = U_weights[i]
        vw = U_weights[j]
        d = compute_distance(u, v, uw, vw, metric_num)
        out[i, j] = d
        out[j, i] = d


# faster compilation *and* runtimes with explicit signature
@cuda.jit(
    "void(float{0}[:,:], float{0}[:,:], float{0}[:,:], float{0}[:,:], float{0}[:,:], int{0})".format(
        bits
    ),
    fastmath=fastmath,
)
def two_set_distance_matrix(U, V, U_weights, V_weights, out, metric_num):
    """
    Calculate pairwise distances between two sets of vectors.

    Parameters
    ----------
    U : 2D array of float
        Vertically stacked vectors.
    V : 2D array of float
        Vertically stacked vectors.
    U_weights : 2D array of float
        Vertically stacked weight vectors.
    V_weights : 2D array of float
        Vertically stacked weight vectors.
    out : 2D array of float
        Pairwise distance matrix between two sets of vectors.
    metric_num : int
        Distance metric number {"euclidean": 0, "wasserstein": 1}.

    Returns
    -------
    None.

    """
    i, j = cuda.grid(2)

    # distance matrix shape
    dm_rows = U.shape[0]
    dm_cols = V.shape[0]

    if i < dm_rows and j < dm_cols:
        u = U[i]
        v = V[j]
        uw = U_weights[i]
        vw = V_weights[j]
        d = compute_distance(u, v, uw, vw, metric_num)
        out[i, j] = d


def dist_matrix(
    U, V=None, U_weights=None, V_weights=None, pairs=None, metric="euclidean"
):  # noqa
    """
    Compute distance matrices using Numba/CUDA.

    Parameters
    ----------
    mat : array
        First set of vectors for which to compute pairwise distances.

    mat2 : array, optional
        Second set of vectors for which to compute pairwise distances.
        If not specified, then mat2 is a copy of mat.

    pairs : array, optional
        List of 2-tuples which contain the indices for which to compute
        distances. If mat2 was specified, then the second index accesses
        mat2 instead of mat. If not specified, then the pairs are
        auto-generated. If mat2 was specified, all combinations of the two
        vector sets are used. If mat2 isn't specified, then only the upper
        triangle (minus diagonal) pairs are computed.

    metric : str, optional
        Possible options are 'euclidean', 'wasserstein'.
        Defaults to Euclidean distance. These are converted to integers
        internally due to Numba's lack of support for string arguments
        (2021-08-14). See compute_distance() for other keys.

        For example:
            0 - 'euclidean'
            1 - 'wasserstein'
    target : str, optional
        Which target to use: "cuda" or "cpu". Default is "cuda".
    inline : str, optional
        Whether to inline functions: "always" or "never". Default is "never".
    fastmath : bool, optional
        Whether to use fastmath or not. The default is True.
    USE_64 : bool, optional
        Whether to use 64 bit or 34 bit types. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    out : array
        A pairwise distance matrix, or if pairs are specified, then a
        vector of distances corresponding to the pairs.

    """
    rows, cols_check = U.shape

    if cols_check != cols:
        raise KeyError(
            "For performance reasons and architecture constraints "
            "the number of columns of U (which is the same as V) "
            "must be defined as the environment variable: COLUMNS. "
            'However, os.environ["COLUMNS"] does not match the '
            "number of columns in U. Reset the environment variable "
            'via e.g. `os.environ["COLUMNS"] = str(U.shape[0])` '
            "defined at the top of your script and make sure that "
            "dist_matrix is reloaded. You may also need to restart "
            "Python."
        )

    if V is not None and cols is not V.shape[1]:
        raise ValueError("The number of columns for U and V should match")

    # is it a distance matrix between two sets of vectors?
    # (rather than within a single set)
    isXY = V is not None

    # were pairs specified? (useful for sparse matrix generation)
    pairQ = pairs is not None

    # block, grid, and out shape
    if pairQ:
        block_dim = TPB
        npairs = pairs.shape[0]
        grid_dim = ceil(npairs / block_dim)
    else:
        block_dim = (TPB, TPB)
        blockspergrid_x = ceil(rows / block_dim[0])
        if isXY:
            blockspergrid_y = ceil(V.shape[0] / block_dim[1])
        else:
            blockspergrid_y = ceil(rows / block_dim[1])
        grid_dim = (blockspergrid_x, blockspergrid_y)

    # CUDA setup
    stream = cuda.stream()

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

    # assign dummy arrays
    if V is None:
        V = np.zeros((2, 2))

    if U_weights is None:
        U_weights = np.zeros((2, 2))

    if V_weights is None:
        V_weights = np.zeros((2, 2))

    if pairs is None:
        pairs = np.zeros((2, 2))

    # assign metric_num based on specified metric (no string support)
    metric_dict = {"euclidean": 0, "wasserstein": 1}
    metric_num = metric_dict[metric]

    # # same for target_num
    # target_dict = {"cuda": 0, "cpu": 1}
    # target_num = target_dict[target]

    m = U.shape[0]

    if isXY:
        m2 = V.shape[0]
    else:
        m2 = m

    if pairQ:
        shape = (npairs,)
    else:
        shape = (m, m2)

    # values should be initialized instead of using cuda.device_array
    out = np.zeros(shape)

    if target == "cuda":
        # copying/allocating to GPU
        cU = cuda.to_device(np.asarray(U, dtype=np_float), stream=stream)

        cU_weights = cuda.to_device(
            np.asarray(U_weights, dtype=np_float), stream=stream
        )

        if isXY or pairQ:
            cV = cuda.to_device(np.asarray(V, dtype=np_float), stream=stream)
            cV_weights = cuda.to_device(
                np.asarray(V_weights, dtype=np_float), stream=stream
            )

        if pairQ:
            cpairs = cuda.to_device(np.asarray(pairs, dtype=np_int), stream=stream)

        cuda_out = cuda.to_device(np.asarray(out, dtype=np_float), stream=stream)

    elif target == "cpu":
        cU = U
        cV = V
        cU_weights = U_weights
        cV_weights = V_weights
        cpairs = pairs
        cuda_out = out

    # distance matrix between two sets of vectors
    if isXY and not pairQ:
        fn = two_set_distance_matrix
        if target == "cuda":
            fn = fn[grid_dim, block_dim]
        fn(cU, cV, cU_weights, cV_weights, cuda_out, metric_num)

    # specified pairwise distances within single set of vectors
    elif not isXY and pairQ:
        fn = sparse_distance_matrix
        if target == "cuda":
            fn = fn[grid_dim, block_dim]
        fn(cU, cU, cU_weights, cU_weights, cpairs, cuda_out, metric_num)

    # distance matrix within single set of vectors
    elif not isXY and not pairQ:
        fn = one_set_distance_matrix
        if target == "cuda":
            fn = fn[grid_dim, block_dim]
        fn(cU, cU_weights, cuda_out, metric_num)

    # specified pairwise distances within single set of vectors
    elif isXY and pairQ:
        fn = sparse_distance_matrix
        if target == "cuda":
            fn = fn[grid_dim, block_dim]
        fn(cU, cV, cU_weights, cV_weights, cpairs, cuda_out, metric_num)

    out = cuda_out.copy_to_host(stream=stream)

    return out


# %% Code Graveyard

# u_stack = cuda.local.array(cols, nb_int)
# v_stack = cuda.local.array(cols, nb_int)

# copy(u_sorted, utmp)
# copy(v_sorted, vtmp)

# copy(u_weights, u_sorted_weights)
# copy(v_weights, v_sorted_weights)

# stack = [0] * size
# ids = list(range(len(arr)))

# u_sorted_weights = cuda.local.array(cols, nb_float)
# v_sorted_weights = cuda.local.array(cols, nb_float)

# sorting stack
# uv_stack = cuda.local.array(tot_cols, nb_int)

# def sort_cum_prepend(X, X_weights):
#     # presort values (and weights by sorted value indices)
#     X_sorter = np.argsort(X)
#     X = np.take_along_axis(X, X_sorter, axis=-1)
#     X_weights = np.take_along_axis(X_weights, X_sorter, axis=-1)
#     # calculate cumulative weights
#     X_weights = np.cumsum(X_weights)
#     # prepend a column of zeros
#     zero = np.zeros((X_weights.shape[0], 1))
#     X_weights = np.column_stack((zero, X_weights))

# def mean_squared_error(y, y_pred, squared=False):
#     """
#     Return mean squared error (MSE) without using sklearn.

#     If squared == True, then return root mean squared error (RMSE).

#     Parameters
#     ----------
#     y : 1D numeric array
#         "True" or first set of values.
#     y_pred : 1D numeric array
#         "Predicted" or second set of values.

#     Returns
#     -------
#     rmse : numeric scalar
#         DESCRIPTION.

#     """
#     mse = np.mean((y - y_pred)**2)
#     if squared is True:
#         rmse = np.sqrt(mse)
#         return rmse
#     else:
#         return mse
#         return mse

# opt = False
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"
# opt = True
# os.environ["NUMBA_BOUNDSCHECK"] = "0"
# os.environ["OPT"] = "0"


# HACK
# a "hacky" way to get compatibility between @njit and @cuda.jit
# myjit = get_myjit(target=target, inline=inline)
# mycudajit = get_myjit(device=False, target=target, inline=inline)

# local array shapes needs to be defined globally due to lack of dynamic
# array allocation support. Don't wrap with np.int32, etc. see
# https://github.com/numba/numba/issues/7314

# np.savetxt('100-zeros.csv',
#           np.zeros(tmp, dtype=np.int32),
#           delimiter=",")
# OK to take cols from .shape
# cols = np.genfromtxt('100-zeros.csv',
#                      dtype=np.int32,
#                      delimiter=",").shape[0]


# HACK from numba.cuda.kernels.device.myjit import get_myjit
# from numba.cuda.cudadrv.driver import driver
# TPB = driver.get_device().MAX_THREADS_PER_BLOCK # max TPB causes crash..


# os.environ["MACHINE_BITS"] = str(bits)

# CUDA Simulator not working..
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

# inline = os.environ.get("INLINE", "never")
# fastmath = bool(os.environ.get("FASTMATH", "1"))
# cols = os.environ.get("COLUMNS")  # 121 for ElM2D repo
# USE_64 = bool(os.environ.get("USE_64", "0"))
# target = os.environ.get("TARGET", "cuda")
