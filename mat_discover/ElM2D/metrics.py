"""
Numba/CUDA-compatible distance metrics.

Created on Wed Sep  8 14:47:43 2021

@author: sterg
"""
# from unittest.mock import patch
import os
import numpy as np
from . import helper as hp
from math import sqrt

from numba.types import int32, float32, int64, float64  # noqa

from numba import cuda  # noqa

inline = os.environ.get("INLINE", "never")
fastmath = bool(os.environ.get("FASTMATH", "1"))
cols = os.environ.get("COLUMNS")
USE_64 = bool(os.environ.get("USE_64", "0"))
target = os.environ.get("TARGET", "cuda")

# if target == "cuda":


# elif target == "cpu":
# patch("cuda.local.array", np.zeros)


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

if target == "cpu":
    nb_float = np_float
    nb_int = np_int

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


@cuda.jit(
    "float{0}(float{0}[:], float{0}[:], float{0}[:], float{0}[:], int{0}, boolean, boolean, boolean)".format(
        bits
    ),
    device=True,
    inline=inline,
)
def cdf_distance(
    u, v, u_weights, v_weights, p, presorted, cumweighted, prepended
):  # noqa
    r"""# noqa
    Compute distance between two 1D distributions :math:`u` and :math:`v`.

    The respective CDFs are :math:`U` and :math:`V`, and the
    statistical distance is defined as:
    .. math::
        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
    p is a positive parameter; p = 1 gives the Wasserstein distance,
    p = 2 gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like
        Weight for each value.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from
        1, it must still be positive and finite so that the weights can
        be normalized to sum to 1.
    p : scalar
        positive parameter that determines the type of cdf distance.
    presorted : bool
        Whether u and v have been sorted already *and* u_weights and
        v_weights have been sorted using the same indices used to sort
        u and v, respectively.
    cumweighted : bool
        Whether u_weights and v_weights have been converted to their
        cumulative weights via e.g. np.cumsum().
    prepended : bool
        Whether a zero has been prepended to accumated, sorted
        u_weights and v_weights.

    By setting presorted, cumweighted, *and* prepended to False, the
    computationproceeds proceeds in the same fashion as _cdf_distance
    from scipy.stats.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from
    samples whose values are effectively inputs of the function, or
    they can be seen as generalized functions, in which case they are
    weighted sums of Dirac delta functions located at the specified
    values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan,
            Hoyer, Munos "The Cramer Distance as a Solution to Biased
            Wasserstein Gradients" (2017). :arXiv:`1705.10743`.
    """
    # allocate local float arrays
    # combined vector
    uv = cuda.local.array(tot_cols, nb_float)
    uv_deltas = cuda.local.array(tot_cols_minus_1, nb_float)

    # CDFs
    u_cdf = cuda.local.array(tot_cols_minus_1, nb_float)
    v_cdf = cuda.local.array(tot_cols_minus_1, nb_float)

    # allocate local int arrays
    # CDF indices via binary search
    u_cdf_indices = cuda.local.array(tot_cols_minus_1, nb_int)
    v_cdf_indices = cuda.local.array(tot_cols_minus_1, nb_int)

    u_cdf_sorted_cumweights = cuda.local.array(tot_cols_minus_1, nb_float)
    v_cdf_sorted_cumweights = cuda.local.array(tot_cols_minus_1, nb_float)

    # short-circuit
    if presorted and cumweighted and prepended:
        u_sorted = u
        v_sorted = v

        u_0_cumweights = u_weights
        v_0_cumweights = v_weights

    # sorting, accumulating, and prepending (for compatibility)
    else:
        # check arguments
        if not presorted and (cumweighted or prepended):
            raise ValueError(
                "if cumweighted or prepended are True, then presorted cannot be False"
            )  # noqa

        if (not presorted or not cumweighted) and prepended:
            raise ValueError(
                "if prepended is True, then presorted and cumweighted must both be True"
            )  # noqa

        # sorting
        if not presorted:
            # local arrays
            u_sorted = cuda.local.array(cols, nb_float)
            v_sorted = cuda.local.array(cols, nb_float)

            u_sorter = cuda.local.array(cols, nb_int)
            v_sorter = cuda.local.array(cols, nb_int)

            u_sorted_weights = cuda.local.array(cols, nb_float)
            v_sorted_weights = cuda.local.array(cols, nb_float)

            # local copy since quickArgSortIterative sorts in-place
            hp.copy(u, u_sorted)
            hp.copy(v, v_sorted)

            # sorting
            hp.insertionArgSort(u_sorted, u_sorter)
            hp.insertionArgSort(v_sorted, v_sorter)

            # inplace to avoid extra cuda local array
            hp.sort_by_indices(u_weights, u_sorter, u_sorted_weights)
            hp.sort_by_indices(v_weights, v_sorter, v_sorted_weights)

        # cumulative weights
        if not cumweighted:
            # local arrays
            u_cumweights = cuda.local.array(cols, nb_float)
            v_cumweights = cuda.local.array(cols, nb_float)
            # accumulate
            hp.cumsum(u_sorted_weights, u_cumweights)
            hp.cumsum(v_sorted_weights, v_cumweights)

        # prepend weights with zero
        if not prepended:
            zero = cuda.local.array(1, nb_float)

            u_0_cumweights = cuda.local.array(cols_plus_1, nb_float)
            v_0_cumweights = cuda.local.array(cols_plus_1, nb_float)

            hp.concatenate(zero, u_cumweights, u_0_cumweights)
            hp.concatenate(zero, v_cumweights, v_0_cumweights)

    # concatenate u and v into uv
    hp.concatenate(u_sorted, v_sorted, uv)

    # sorting
    # quickSortIterative(uv, uv_stack)
    hp.insertionSort(uv)

    # Get the respective positions of the values of u and v among the
    # values of both distributions. See also np.searchsorted
    hp.bisect_right(u_sorted, uv[:-1], u_cdf_indices)
    hp.bisect_right(v_sorted, uv[:-1], v_cdf_indices)

    # empirical CDFs
    hp.sort_by_indices(u_0_cumweights, u_cdf_indices, u_cdf_sorted_cumweights)
    hp.divide(u_cdf_sorted_cumweights, u_0_cumweights[-1], u_cdf)

    hp.sort_by_indices(v_0_cumweights, v_cdf_indices, v_cdf_sorted_cumweights)
    hp.divide(v_cdf_sorted_cumweights, v_0_cumweights[-1], v_cdf)

    # # Integration
    hp.diff(uv, uv_deltas)  # See also np.diff

    out = hp.integrate(u_cdf, v_cdf, uv_deltas, p)

    return out


@cuda.jit(
    "float{0}(float{0}[:], float{0}[:], float{0}[:], float{0}[:], boolean, boolean, boolean)".format(
        bits
    ),
    device=True,
    inline=inline,
)
def wasserstein_distance(
    u, v, u_weights, v_weights, presorted, cumweighted, prepended
):  # noqa
    r"""
    Compute the first Wasserstein distance between two 1D distributions.

    This distance is also known as the earth mover's distance, since it can be
    seen as the minimum amount of "work" required to transform :math:`u` into
    :math:`v`, where "work" is measured as the amount of distribution weight
    that must be moved, multiplied by the distance it has to be moved.

    Source
    ------
    https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L8245-L8319 # noqa

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.
    """
    return cdf_distance(
        u, v, u_weights, v_weights, np_int(1), presorted, cumweighted, prepended
    )


# TODO: explicit signature?
@cuda.jit(
    "float{0}(float{0}[:], float{0}[:])".format(bits),
    device=True,
    inline=inline,
)
def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between vectors a and b.

    Parameters
    ----------
    a : 1D array
        First vector.
    b : 1D array
        Second vector.

    Returns
    -------
    d : numeric scalar
        Euclidean distance between vectors a and b.
    """
    d = 0
    for i in range(len(a)):
        d += (b[i] - a[i]) ** 2
    d = sqrt(d)
    return d


# %% Code Graveyard
# inline = os.environ.get("INLINE", "never")
# fastmath = bool(os.environ.get("FASTMATH", "1"))
# cols = os.environ.get("COLUMNS")  # 121 for ElM2D repo
# USE_64 = bool(os.environ.get("USE_64", "0"))
# target = os.environ.get("TARGET", "cuda")
