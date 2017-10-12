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

blas = pyculib.blas.Blas()

USE_64 = True

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32

@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def distance_matrix(mat, d_sq):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        d_sq[i, j] = d

def gpu_dist_center(mat):
    rows = mat.shape[0]
    n_ = rows
    
    block_dim = (16, 16)
    grid_dim = (rows/block_dim[0] + 1, rows/block_dim[1] + 1)
    
    stream = cuda.stream()
    gpu_mat = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    gpu_d_sq = cuda.device_array((rows, rows))
    
    gpu_j = cuda.to_device(np.identity(n_) - (np.ones([n_, n_]) / float(n_)),
                           stream=stream)
    gpu_jd = cuda.device_array((rows, rows))
    gpu_b = cuda.device_array((rows, rows))

    distance_matrix[grid_dim, block_dim](gpu_mat, gpu_d_sq)
    
    blas.gemm("N", "N", rows, rows, rows, 1, gpu_j, gpu_d_sq, 0, gpu_jd)
    blas.gemm("N", "N", rows, rows, rows, (-1./2.), gpu_jd, gpu_j, 0, gpu_b)
    
#    blas.symm('L', 'U', rows, rows, 1, gpu_j, gpu_d_sq, 0, gpu_jd)
#    blas.symm('L', 'U', rows, rows, (-1./2.), gpu_jd, gpu_j, 0, gpu_b)
    
    b_ = gpu_b.copy_to_host(stream=stream)
    
    return b_


def gpu_dist_matrix(mat):
    rows = mat.shape[0]
    
    block_dim = (16, 16)
    grid_dim = (rows/block_dim[0] + 1, rows/block_dim[1] + 1)
    
    stream = cuda.stream()
    mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    out2 = cuda.device_array((rows, rows))
    start = time.time()
    distance_matrix[grid_dim, block_dim](mat2, out2)
    out = out2.copy_to_host(stream=stream)
    print 'time:', time.time() - start
    
    return out

rows = 2000
cols = 1000

mat = np.random.randn(rows, cols)


d_mat = gpu_dist_matrix(mat)

#
#start = time.time()
#b_gpu = gpu_dist_center(mat)
#print 'time:', time.time() - start
#
#start = time.time()
#n_ = len(mat)
#d_mat = gpu_dist_matrix(mat)
#d_sq = d_mat * d_mat 
#start = time.time()
#j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
#b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)
#print 'time:', time.time() - start
#print np.sum((b_ - b_gpu) ** 2)
#
#start = time.time()
#n_ = len(mat)
#d_mat = spatial.distance.cdist(mat, mat)
#d_sq = d_mat * d_mat 
#start = time.time()
#j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
#b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)
#print 'time:', time.time() - start
#print np.sum((b_ - b_gpu) ** 2)