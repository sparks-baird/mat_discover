# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 00:47:33 2021

@author: sterg
"""

import numpy as np
import cupy as cp
from time import time
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances


def l1_distance(arr):
    return np.linalg.norm(arr, 1)


X = np.random.randint(low=0, high=255, size=(700, 4096))
distance = np.empty((700, 700))

start = time()
cdist(X, X)
end = time()
print("Elapsed = %s" % (end - start))

from sklearn.metrics import pairwise_distances

start = time()
pairwise_distances(X, Y=X, n_jobs=-1)
end = time()
print("Elapsed = %s" % (end - start))


def l1_distance(arr):
    return cp.linalg.norm(arr, 1)


X = cp.random.randint(low=0, high=255, size=(700, 4096))
distance = cp.empty((700, 700))

start = time()
for i in range(700):
    distance[i, :] = cp.abs(cp.broadcast_to(X[i, :], X.shape) - X).sum(axis=1)
end = time()
print("Elapsed = %s" % (end - start))
