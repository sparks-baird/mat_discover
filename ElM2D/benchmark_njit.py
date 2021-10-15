# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 01:36:19 2021

@author: sterg
"""
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from scipy.spatial.distance import cdist

# %% example runs
pairtest = np.array([(0, 1), (1, 2), (2, 3)])
Utest = U[0:6]
Vtest = V[0:6]
Uwtest = U_weights[0:6]
Vwtest = V_weights[0:6]

one_set = dist_matrix(Utest, Uw=Uwtest, metric="wasserstein")
print(one_set)

two_set = dist_matrix(Utest, V=Vtest, Uw=Uwtest, Vw=Vwtest, metric="wasserstein")
print(two_set)

pairs = np.array([(0, 1), (1, 2), (2, 3)])

one_set_sparse = dist_matrix(Utest, Uw=Uwtest, pairs=pairtest, metric="wasserstein")
print(one_set_sparse)

two_set_sparse = dist_matrix(
    Utest, V=Vtest, Uw=Uwtest, Vw=Vwtest, pairs=pairtest, metric="wasserstein"
)
print(two_set_sparse)

# %% timing of large distance matrices


# for compilation purposes, maybe just once is necessary?
dist_matrix(U, Uw=U_weights, metric="wasserstein")
dist_matrix(Utest, V=Vtest, Uw=Uwtest, Vw=Vwtest, metric="wasserstein")
dist_matrix(Utest, Uw=Uwtest, pairs=pairtest, metric="wasserstein")
dist_matrix(Utest, V=Vtest, Uw=Uwtest, Vw=Vwtest, pairs=pairtest, metric="wasserstein")

with Timer("cdist Euclidean"):
    d = cdist(U, V)

with Timer("two-set dist_matrix Euclidean"):
    d = dist_matrix(U, V=V, Uw=U_weights, Vw=V_weights, metric="euclidean")

with Timer("cdist SciPy Wasserstein"):
    d = cdist(U_Uw, V_Vw, metric=scipy_wasserstein_distance)

with Timer("cdist Wasserstein"):
    d = cdist(U_Uw, V_Vw, metric=my_wasserstein_distance)

with Timer("two-set dist_matrix Wasserstein"):
    d = dist_matrix(U, V=V, Uw=U_weights, Vw=V_weights, metric="wasserstein")
