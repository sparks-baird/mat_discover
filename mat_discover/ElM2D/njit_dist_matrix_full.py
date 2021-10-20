"""Nopython version of dist_matrix."""
import os
import numpy as np
from math import sqrt

from numba import prange, njit
from numba.types import int32, float32, int64, float64

from . import cpu_helper as hp

# settings
inline = os.environ.get("INLINE", "never")
fastmath = bool(os.environ.get("FASTMATH", "1"))
parallel = bool(os.environ.get("PARALLEL", "1"))
debug = bool(os.environ.get("DEBUG", "0"))


def dist_matrix(
    U,
    V=None,
    U_weights=None,
    V_weights=None,
    pairs=None,
    metric="euclidean",
    USE_64=False,
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
    cols = U.shape[1]

    # %% Metrics specific to njit / CPU implementations
    cols_plus_1 = cols + 1
    tot_cols = cols * 2
    tot_cols_minus_1 = tot_cols - 1

    if USE_64:
        bits = 64
        bytes = 8
        nb_float = float64
        nb_int = int64
        np_float = np.float64
        np_int = np.int64
    else:
        bits = 32
        bytes = 4
        nb_float = float32
        nb_int = int32
        np_float = np.float32
        np_int = np.int32

    # @njit(fastmath=fastmath, debug=debug)
    @njit(
        "f{0}(f{0}[:], f{0}[:], f{0}[:], f{0}[:], i{0}, b1, b1, b1)".format(bytes),
        fastmath=fastmath,
        debug=debug,
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
        uv = np.zeros(tot_cols, dtype=np_float)
        uv_deltas = np.zeros(tot_cols_minus_1, dtype=np_float)

        # CDFs
        u_cdf = np.zeros(tot_cols_minus_1, dtype=np_float)
        v_cdf = np.zeros(tot_cols_minus_1, dtype=np_float)

        # allocate local int arrays
        # CDF indices via binary search
        u_cdf_indices = np.zeros(tot_cols_minus_1, dtype=np_int)
        v_cdf_indices = np.zeros(tot_cols_minus_1, dtype=np_int)

        u_cdf_sorted_cumweights = np.zeros(tot_cols_minus_1, dtype=np_float)
        v_cdf_sorted_cumweights = np.zeros(tot_cols_minus_1, dtype=np_float)

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
                u_sorted = np.zeros(cols, dtype=np_float)
                v_sorted = np.zeros(cols, dtype=np_float)

                u_sorter = np.zeros(cols, dtype=np_int)
                v_sorter = np.zeros(cols, dtype=np_int)

                u_sorted_weights = np.zeros(cols, dtype=np_float)
                v_sorted_weights = np.zeros(cols, dtype=np_float)

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
                u_cumweights = np.zeros(cols, dtype=np_float)
                v_cumweights = np.zeros(cols, dtype=np_float)
                # accumulate
                hp.cumsum(u_sorted_weights, u_cumweights)
                hp.cumsum(v_sorted_weights, v_cumweights)

            # prepend weights with zero
            if not prepended:
                zero = np.zeros(1, dtype=np_float)

                u_0_cumweights = np.zeros(cols_plus_1, dtype=np_float)
                v_0_cumweights = np.zeros(cols_plus_1, dtype=np_float)

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

    # @njit(fastmath=fastmath, debug=debug)
    @njit(
        "f{0}(f{0}[:], f{0}[:], f{0}[:], f{0}[:], b1, b1, b1)".format(bytes),
        fastmath=fastmath,
        debug=debug,
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
        )  # noqa

    # @njit(fastmath=fastmath, debug=debug)
    @njit(
        "f{0}(f{0}[:], f{0}[:])".format(bytes),
        fastmath=fastmath,
        debug=debug,
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

    # %% Top-level distance calculation functions
    # @njit(fastmath=fastmath, debug=debug)
    @njit(
        "f{0}(f{0}[:], f{0}[:], f{0}[:], f{0}[:], i{0})".format(bytes),
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
        "void(f{0}[:,:], f{0}[:,:], f{0}[:,:], f{0}[:,:], i{0}[:,:], f{0}[:], i{0})".format(
            bytes
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
        "void(f{0}[:,:], f{0}[:,:], f{0}[:,:], i{0})".format(bytes),
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
        "void(f{0}[:,:], f{0}[:,:], f{0}[:,:], f{0}[:,:], f{0}[:,:], i{0})".format(
            bytes
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
    if pairs is not None:
        pairs = pairs.astype(np_int)

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
