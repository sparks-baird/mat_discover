# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 02:10:00 2021

@author: sterg
"""
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
import numpy as np
from numba import cuda

from pdb import set_trace

@cuda.jit
def vec_add(A, B, out):
    x = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    if x == 1 and bx == 3: set_trace()
    i = bx * bdx + x
    out[i] = A[i] + B[i]
    
A = np.random.rand(100)
B = np.random.rand(100)

out = np.zeros_like(A)
vec_add[10,10](A,B,out)

print(out)
