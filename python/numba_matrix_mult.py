import numpy as np
from scipy import spatial
from scipy import linalg as LA
import scipy.sparse.linalg as SP
import time
import sys
import sklearn
from numba import cuda
import time
import pyculib
from numba.decorators import jit, autojit

#np.set_printoptions(precision=2)
#
#USE_64 = True
#
#if USE_64:
#    bits = 64
#    np_type = np.float64
#else:
#    bits = 32
#    np_type = np.float32
#
#def two_dot_python(a, b):
#    
#    assert len(set(list(a.shape) + list(b.shape))) == 1
#    
#    m = a.shape[0]
#    
#    c = np.zeros([a.shape[0], b.shape[1]])
#    
#    for i in range(m):
#        for j in range(m):
#            val = 0
#            for k in range(m):
#                val2 = 0
#                for l in range(m):
#                    val2 += a[i, l] * b[l, k]
#                val += val2 * a[k, j]
#            c[i, j] = val
#            
#    return c
#
#
#two_dot_numba = autojit(two_dot_python)
#
#
#@cuda.jit("void(float{}, float{}[:, :], float{}[:, :])".format(bits, bits, bits))
#def distance_matrix(x, mat, d_sq):
#    m = mat.shape[0]
#    n = mat.shape[1]
#    i, j = cuda.grid(2)
#    d = 0
#    if i < m and j < m:
#        for k in range(n):
#            tmp = mat[i, k] - mat[j, k]
#            d += tmp * tmp
#        d_sq[i, j] = j#d
#    
#format_str = "void(float{}, float{}[:, :], float{}[:, :], float{}[:, :])"
#@cuda.jit(format_str.format(bits, bits, bits, bits))
#def mat_mul(x, a, b, c):
#    rows = a.shape[0]
#    cols = b.shape[1]
#    
#    i, j = cuda.grid(2)
#    val = 0
#    if (i < rows) and (j < cols):
#        for k in range(a.shape[1]):
#            val += a[i, k] * b[k, j]
#        c[i, j] = x * val
#        
#def use_gpu_dot(x, a, b):
#    
#    stream = cuda.stream()
#    a_gpu = cuda.to_device(np.asarray(a, dtype=np_type), stream=stream)
#    b_gpu = cuda.to_device(np.asarray(b, dtype=np_type), stream=stream)
#    c_gpu = cuda.device_array((a.shape[0], b.shape[1]))
#    
#    block_dim = (16, 16)
#    grid_dim = (a.shape[0]/block_dim[0] + 1, b.shape[1]/block_dim[1] + 1)
#    
#    mat_mul[grid_dim, block_dim](1., a_gpu, b_gpu, c_gpu)
#    
#    c = c_gpu.copy_to_host(stream=stream)
#    
#    return c
    
rows = 100
cols = 100

a = np.random.randn(rows, cols)
b = np.random.randn(cols, rows)
x = 1.

#stream = cuda.stream()
#a_gpu = cuda.to_device(np.asarray(a, dtype=np_type), stream=stream)
#b_gpu = cuda.to_device(np.asarray(b, dtype=np_type), stream=stream)
#c_gpu = cuda.device_array((a.shape[0], b.shape[1]))
#
#block_dim = (16, 16)
#grid_dim = (a.shape[0]/block_dim[0] + 1, b.shape[1]/block_dim[1] + 1)
#
#mat_mul[grid_dim, block_dim](1., a_gpu, b_gpu, c_gpu)
##distance_matrix[grid_dim, block_dim](1., a_gpu, c_gpu)
#
#c = c_gpu.copy_to_host(stream=stream)
#
#print ((c - a.dot(b)) ** 2).sum()