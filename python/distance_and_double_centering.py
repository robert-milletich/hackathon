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

np.set_printoptions(precision=2)

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
        
@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits), device=0)
def distance_matrix_cache(mat, d_sq):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        d_sq[i, j] = d

format_str = "void(float{}, float{}[:, :], float{}[:, :], float{}[:, :])"
@cuda.jit(format_str.format(bits, bits, bits, bits))
def mat_mul(x, a, b, c):
    rows = a.shape[0]
    cols = b.shape[1]
    
    i, j = cuda.grid(2)
    val = 0
    if (i < rows) and (j < cols):
        for k in range(a.shape[1]):
            val += a[i, k] * b[k, j]
        c[i, j] = x * val

format_str = "void(float{}, float{}[:, :], float{}[:, :], float{}[:, :])"
@cuda.jit(format_str.format(bits, bits, bits, bits))
def two_dot(x, a, b, c):
    m = a.shape[0]
    i, j = cuda.grid(2)
    if (i < m) and (j < m):
        val = 0
        for k in range(m):
            val2 = 0
            for l in range(m):
                val2 += a[i, l] * b[l, k]
            val += val2 * a[k, j]
        c[i, j] = x * val

format_str = "void(float{}[:, :], float{}[:], float{}, float{}[:, :])"
@cuda.jit(format_str.format(bits, bits, bits, bits))
def all_means(a, row_mean, all_mean, c):
    m = a.shape[0]
    i, j = cuda.grid(2)
    if (i < m) and (j < m):
        c[i, j] = a[i, j] - row_mean[i] - row_mean[j] + all_mean
        
def use_gpu_dot(x, a, b):
    
    stream = cuda.stream()
    a_gpu = cuda.to_device(np.asarray(a, dtype=np_type), stream=stream)
    b_gpu = cuda.to_device(np.asarray(b, dtype=np_type), stream=stream)
    c_gpu = cuda.device_array((a.shape[0], b.shape[1]))
    
    block_dim = (16, 16)
    grid_dim = (a.shape[0]/block_dim[0] + 1, b.shape[1]/block_dim[1] + 1)
    
    mat_mul[grid_dim, block_dim](1., a_gpu, b_gpu, c_gpu)
    
    c = c_gpu.copy_to_host(stream=stream)
    
    return c

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
    
#    two_dot[grid_dim, block_dim]((-1./2.), gpu_j, gpu_d_sq, gpu_b)
    mat_mul[grid_dim, block_dim](1, gpu_j, gpu_d_sq, gpu_jd)
    mat_mul[grid_dim, block_dim]((-1./2.), gpu_jd, gpu_j, gpu_b)
    
#    blas.gemm("N", "N", rows, rows, rows, 1, gpu_j, gpu_d_sq, 0, gpu_jd)
#    blas.gemm("N", "N", rows, rows, rows, (-1./2.), gpu_jd, gpu_j, 0, gpu_b)
    
#    blas.symm('L', 'U', rows, rows, 1, gpu_j, gpu_d_sq, 0, gpu_jd)
#    blas.symm('L', 'U', rows, rows, (-1./2.), gpu_jd, gpu_j, 0, gpu_b)
    
    b_ = gpu_b.copy_to_host(stream=stream)
    
    return b_

def gpu_dist_double_center(mat):
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
    
    mat_mul[grid_dim, block_dim](1, gpu_j, gpu_d_sq, gpu_jd)
    mat_mul[grid_dim, block_dim]((-1./2.), gpu_jd, gpu_j, gpu_b)
    
    b_ = gpu_b.copy_to_host(stream=stream)
    
    return b_

def gpu_dist_center_blas(mat):
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
#    return gpu_jd.copy_to_host(stream=stream)
    

#    mat_mul(1, gpu_j, gpu_d_sq, gpu_jd)
#    mat_mul((-1./2.), gpu_jd, gpu_j, gpu_b)
    
#    blas.gemm("N", "N", rows, rows, rows, 1, gpu_j, gpu_d_sq, 0, gpu_jd)
    
    blas.symm('L', 'U', rows, rows, 1., gpu_j, gpu_d_sq, 0, gpu_jd)
    blas.gemm("N", "N", rows, rows, rows, (-1./2.), gpu_jd, gpu_j, 0, gpu_b)
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
    distance_matrix[grid_dim, block_dim](mat2, out2)
    out = out2.copy_to_host(stream=stream)
    
    return out

def gpu_dist_matrix_cache(mat):
    rows = mat.shape[0]
    
    block_dim = (16, 16)
    grid_dim = (rows/block_dim[0] + 1, rows/block_dim[1] + 1)
    
    stream = cuda.stream()
    
    with stream.auto_synchronize():
        mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
        out2 = cuda.device_array((rows, rows))
        distance_matrix[grid_dim, block_dim](mat2, out2)
        out = out2.copy_to_host(stream=stream)
    
    return out

rows = 300
cols = 100

x = 1.

mat = np.random.randn(rows, cols)

#d_mat = gpu_dist_matrix(mat)

#start = time.time()
#n_ = len(mat)
#d_mat = spatial.distance.cdist(mat, mat)
#d_sq = d_mat * d_mat 
#j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
#b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)
#print '--------------------no gpu: {:.4f} sec'.format(time.time() - start)
#
#start = time.time()
#n_ = len(mat)
#d_sq = gpu_dist_matrix(mat)
#j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
#b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)
#print '---------gpu distance only: {:.4f} sec'.format(time.time() - start)

start = time.time()
n_ = len(mat)
d_sq = gpu_dist_matrix(mat)
d_s = (-1./2.)*d_sq
b_ = d_s - d_s.mean(axis=1) - d_s.mean(axis=0)[None].T + d_s.mean()
print 'gpu distance only w/double: {:.4f} sec'.format(time.time() - start)

start = time.time()
n_ = len(mat)
d_sq = gpu_dist_matrix_cache(mat)
d_s = (-1./2.)*d_sq
b_ = d_s - d_s.mean(axis=1) - d_s.mean(axis=0)[None].T + d_s.mean()
print 'gpu distance (cach) double: {:.4f} sec'.format(time.time() - start)

#start = time.time()
#n_ = len(mat)
#b_gpu = gpu_dist_double_center(mat)
#print '-----gpu distance w/center: {:.4f} sec |'.format(time.time() - start),
#print np.sum((b_ - b_gpu) ** 2)
#
#start = time.time()
#b_gpu = gpu_dist_center(mat)
#print '------------gpu everything: {:.4f} sec |'.format(time.time() - start),
#print np.sum((b_ - b_gpu) ** 2)
#
#start = time.time()
#b_gpu_blas = gpu_dist_center_blas(mat)
#print 'gpu everything with cublas: {:.4f} sec |'.format(time.time() - start),
#print np.sum((b_gpu_blas - b_gpu) ** 2)

d_s = (-1./2.)*d_sq
d_s - d_s.mean(axis=1) - d_s.mean(axis=0)[None].T + d_s.mean()