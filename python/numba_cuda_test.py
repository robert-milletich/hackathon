import numba
from numba import cuda
import numpy as np
from scipy import spatial
import time

@cuda.jit('int32(int32, int32)', device=True)
def bar(a, b):
    return a + b

@cuda.jit('void(int32[:], int32[:], int32[:])')
def use_bar(list_a, list_b, out):
    i = cuda.grid(1) # global position of the thread for a 1D grid.
    out[i] = bar(list_a[i], list_b[i])


a = np.array(list(range(10)), dtype=np.int32) + 4
b = np.array(list(range(10)), dtype=np.int32) + 3
c = np.array(list(range(10)), dtype=np.int32)

griddim = 3
blockdim = 4

use_bar[griddim, blockdim](a, b, c)
#print foo.ptx

def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

USE_64 = True

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32

@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def distance_matrix(mat, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        out[i, j] = d

@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def dist_off_diag(mat, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    if i < m:
        d = 0
        d2 = 0
        not_j = m - j
        if j > i:
            for k in range(n):
                tmp = mat[i, k] - mat[j, k]
                d += tmp * tmp
            out[i, j] = d
            out[j, i] = d
        if not_j > i:
            for k in range(n):
                tmp = mat[i, k] - mat[not_j, k]
                d2 += tmp * tmp
            out[i, not_j] = d2
            out[not_j, i] = d2

@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def dist_zero_diag(mat, out):
    i = cuda.grid(1)
    out[i, i] = 0

rows = 100
cols = 10

mat = np.array(np.random.randn(rows, cols), dtype=np_type)

def gpu_dist_matrix(mat):
    rows = mat.shape[0]
    
    block_dim = (16, 16)
    grid_dim = (rows/block_dim[0] + 1, rows/block_dim[1] + 1)
    
    stream = cuda.stream()
    gpu_mat = cuda.to_device(mat, stream=stream)
    gpu_d_sq = cuda.device_array((rows, rows))
    distance_matrix[grid_dim, block_dim](gpu_mat, gpu_d_sq)
    
    d_sq = gpu_d_sq.copy_to_host(stream=stream)
    
    return d_sq

start = time.time()

block_dim = (16, 16)
grid_dim = (rows/block_dim[0] + 1, rows/block_dim[1] + 1)

stream = cuda.stream()
gpu_mat = cuda.to_device(mat, stream=stream)

gpu_d_sq = cuda.device_array((rows, rows))
distance_matrix[grid_dim, block_dim](gpu_mat, gpu_d_sq)
out = gpu_d_sq.copy_to_host(stream=stream)
print time.time() - start


#start = time.time()
#block_dim = (16, 1)
#grid_dim = (rows/block_dim[0] + 1, rows/(2*block_dim[1]) + 1)
#
#stream = cuda.stream()
#gpu_mat = cuda.to_device(mat, stream=stream)
#gpu_d_sq = cuda.device_array((rows, rows))
#
#dist_zero_diag[rows/256 + 1, 256](gpu_mat, gpu_d_sq)
#dist_off_diag[grid_dim, block_dim](gpu_mat, gpu_d_sq)
#out3 = gpu_d_sq.copy_to_host(stream=stream)
#print time.time() - start
#
start = time.time()
d_mat = spatial.distance.cdist(mat, mat)
d_sq = d_mat * d_mat
print time.time() - start

print "diff:", np.sum(d_sq - out)
#print "diff:", np.sum(d_sq - out3)