# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 00:39:42 2021

@author: sterg
"""

import numpy as np
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
from numba import cuda

from itertools import product

from pdb import set_trace

USE_64 = True

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32

@cuda.jit("void(float{}[:], float{}[:])".format(bits, bits), debug=True)
def distance_matrix(mat, mat2, pairs, out):
    """
    Compute a single pairwise distance between two sets of vectors.

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
    # for debugging
    x = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    
    # Are there two sets of vectors for which to compute pairwise distances?
    isXY = mat2 is not None
    
    if not isXY:
        # then there's only a single set of vectors for the distance matrix
        mat2 = mat
    
    # unpack shapes
    m, n = mat.shape
    m2, n2 = mat2.shape
    npairs = len(pairs)
    if n != n2:
        raise ValueError('Both matrices should have the same number of columns')
    
    # initialize the distance to zero
    d = 0
    
    #extract the pair number
    pair_num = cuda.grid(1)
    
    # skip pair numbers which are larger than the total # of pairs
    if pair_num < npairs:
        # extract the indices of the pair for which the distance will be computed
        pair = pairs[pair_num]
        i, j = pair
        
        # is it in the upper triangle, excluding diagonal? (only relevant if single set of vectors)
        uppertriQ = i > int(m/2) and j > int(m/2)
    
        #if x == 1 and bx == 3: set_trace() #debugging
        
        # skip indices that are larger than the vector size or are in the lower triangle for single set of vectors
        if i < m and j < m2 and (isXY or uppertriQ):
            # loop through elements of each vector simultaneously
            for k in range(n):
                #unpack the k-th element for each vector
                el = mat[i, k]
                el2 = mat2[j, k]
                # subtract the two elements
                diff = el - el2
                # square the difference and perform addition assignment
                d += diff ** 2
            # assign the output
            out[i, j] = d
            
            if not isXY:
                out[j, i] = d

def gpu_dist_matrix(mat, mat2 = None, pairs = None):
    """
    Compute pairwise distances on a GPU using a numba-compiled function.

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

    Returns
    -------
    out : array
        A pairwise distance matrix.

    """
    # how many vectors do you have?
    rows = mat.shape[0]
    
    # is it a distance matrix between two sets of vectors instead of within a single set?
    isXY = mat2 is not None
    
    if isXY:
        rows2 = mat2.shape[0]
    else:
        rows2 = rows
        
    # should we generate pairs, or were they pre-specified?
    generate_pairs = pairs is None
    
    if generate_pairs:
        if isXY:
            # all pairs
            pairs = list(product(range(rows), range(rows2)))
        else:                
            # upper triangle only
            pairs = list(zip(*np.triu_indices(rows)))

    npairs = len(pairs)
    
    # GPU block and grid sizes
    block_dim = 2
    grid_dim = int(npairs/block_dim + 1)

    # CUDA setup
    stream = cuda.stream()
    
    # copying data to GPU
    cuda_mat = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    if isXY:
        cuda_mat2 = cuda.to_device(np.asarray(mat2, dtype=np_type), stream=stream)
    else:
        cuda_mat2 = None
    cuda_out = cuda.device_array((rows, rows2))
    
    # compute the distance_matrix
    distance_matrix[grid_dim, block_dim](cuda_mat, cuda_mat2, pairs, cuda_out)
    
    # copy it to the CPU
    out = cuda_out.copy_to_host(stream=stream)

    return out

rows = 10
cols = 4
mat = np.random.rand(rows, cols)
mat2 = np.random.rand(rows-1, cols)

out = gpu_dist_matrix(mat, mat2)
print(out)



"""CODE GRAVEYARD"""
"""
# @cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits), debug=True)
# def distance_matrix(mat, mat2, pairs, out): 
    
    

#     Parameters
#     ----------
#     mat : numeric cuda array
#         First set of vectors for which to compute a single pairwise distance.
#     mat2 : numeric cuda array
#         Second set of vectors for which to compute a single pairwise distance.
#     pairs : cuda array of 2-tuples
#         All pairs for which distances are to be computed.
#     out : numeric cuda array
#         The initialized array which will be populated with distances.

#     Raises
#     ------
#     ValueError
#         Both matrices should have the same number of columns.

#     Returns
#     -------
#     None.

#     # # for debugging
#     # x = cuda.threadIdx.x
#     # bx = cuda.blockIdx.x
#     # bdx = cuda.blockDim.x
#     # if x == 1 and bx == 3:
#     #     from pdb import set_trace; set_trace()
    
#     # Are there two sets of vectors for which to compute pairwise distances?
#     isXY = mat2 is not None
    
#     if not isXY:
#         # then there's only a single set of vectors for the distance matrix
#         mat2 = mat
    
#     m, n = mat.shape
#     m2, n2 = mat2.shape
    
#     if n != n2:
#         raise ValueError('Both matrices should have the same number of columns')
    
#     # initialize the distance to zero
#     d = 0
    
#     # extract the indices of the pair for which the distance will be computed
#     i, j = cuda.grid(2)
#     # is it in the upper triangle, excluding diagonal? (assuming single set of vectors)
#     uppertriQ = i > int(m/2) and j > int(m/2)
#     # skip indices that are larger than the vector size or are in the lower triangle (for single set of vectors)
#     if i < m and j < m2 and (isXY or uppertriQ):
#         # loop through elements of each vector simultaneously
#         for k in range(n):
#             #unpack the k-th element for each vector
#             el = mat[i, k]
#             el2 = mat2[j, k]
#             # subtract the two elements
#             diff = el - el2
#             # square the difference and perform addition assignment
#             d += diff ** 2
#         # assign the output
#         out[i, j] = d

# def gpu_dist_matrix(mat, mat2 = None, pairs = None):
    
#     Compute pairwise distances on a GPU using a numba-compiled function.

#     Parameters
#     ----------
#     mat : array
#         First set of vectors for which to compute pairwise distances.
        
#     mat2 : array, optional
#         Second set of vectors for which to compute pairwise distances. If not specified,
#         then mat2 is a copy of mat.
        
#     pairs : array, optional
#         List of 2-tuples which contain the indices for which to compute distances for.
#         If mat2 was specified, then the second index accesses mat2 instead of mat.
#         If not specified, then the pairs are auto-generated. If mat2 was specified,
#         all combinations of the two vector sets are used. If mat2 isn't specified,
#         then only the upper triangle (minus diagonal) pairs are computed.

#     Returns
#     -------
#     out : array
#         A pairwise distance matrix.

    
#     # how many vectors do you have?
#     rows = mat.shape[0]
    
#     # is it a distance matrix between two sets of vectors instead of within a single set?
#     isXY = mat2 is not None
    
#     if isXY:
#         rows2 = mat2.shape[0]
#     else:
#         rows2 = rows
        
#     # should we generate pairs, or were they pre-specified?
#     generate_pairs = pairs is None
    
#     if generate_pairs:
#         if isXY:
#             # all pairs
#             pairs = list(product(range(rows), range(rows2)))
#         else:                
#             # upper triangle only
#             pairs = list(zip(*np.triu_indices(rows)))

#     # GPU block and grid sizes
#     block_dim = (16, 16)
#     grid_dim = (int(rows/block_dim[0] + 1), int(rows2/block_dim[1] + 1))

#     # CUDA setup
#     stream = cuda.stream()
    
#     # copying data to GPU
#     cuda_mat = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
#     if isXY:
#         cuda_mat2 = cuda.to_device(np.asarray(mat2, dtype=np_type), stream=stream)
#     else:
#         cuda_mat2 = None
#     cuda_out = cuda.device_array((rows, rows2))
    
#     # compute the distance_matrix
#     distance_matrix[grid_dim, block_dim](cuda_mat, cuda_mat2, cuda_out)
    
#     # copy it to the CPU
#     out = cuda_out.copy_to_host(stream=stream)

#     return out

# @cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
# def pairwise_distances(mat, out, mat2=None, pairs=None):
#     m = mat.shape[0]
#     n = mat.shape[1]
#     i, j = cuda.grid(2)
#     d = 0
#     if i < m and j < m:
#         for k in range(n):
#             tmp = mat[i, k] - mat[j, k]
#             d += tmp * tmp
#         out[i, j] = d

# def gpu_sparse_dist_matrix(mat, mat2, pairs):
#     rows = mat.shape[0]

#     block_dim = (16, 16)
#     grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))

#     stream = cuda.stream()
#     mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
#     out2 = cuda.device_array((rows, rows))
#     distance_matrix[grid_dim, block_dim](mat2, out2)
#     out = out2.copy_to_host(stream=stream)

#     return out
"""