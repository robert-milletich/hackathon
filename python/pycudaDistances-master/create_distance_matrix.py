import numpy as np
from scipy import spatial
from numba import double
from numba.decorators import jit, autojit
import time
import pyculib
from numba import cuda
from distances import euclidean_distances

def pairwise_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))

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

#def pairwise_cuda(x):
#    m = x.shape[0]
#    n = x.shape[1]
#    d = np.empty((m, m), dtype=np.float)

pairwise_numba = autojit(pairwise_python)

rows = 100
cols = 100
mat = np.random.randn(rows, cols)

foo = pairwise_numba(np.random.randn(3, 3))
print 'hi'

n_ = len(mat)
d_mat = spatial.distance.cdist(mat, mat)
d_sq = d_mat * d_mat
j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)

start = time.time()
foo = spatial.distance.cdist(mat, mat) ** 2
print time.time() - start

start = time.time()
foo = pairwise_numba(mat)
print time.time() - start
#
#start = time.time()
#foo = pairwise_python(mat)
#print time.time() - start
#
#start = time.time()
#foo = pairwise_numpy(mat)
#print time.time() - start

blas = pyculib.blas.Blas()
np.set_printoptions(suppress=True)


def blas_gemm(a_, b_):
    start = time.time()
    a = a_.T.flatten().reshape(a_.shape)
    b = b_.T.flatten().reshape(b_.shape)
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    c = np.zeros([m, n])
    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    c_gpu = cuda.to_device(c)
    
    blas.gemm("N", "N", m, n, k, 1, a_gpu, b_gpu, 0, c_gpu)
    print time.time() - start
    
#    start = time.time()
#    blas.gemm("N", "N", m, n, k, 1, a_gpu, b_gpu, 0, c)
#    print time.time() - start
    
    return c.T

rows = 10
cols = 10

a = np.random.randn(rows, cols)
b = np.random.randn(cols, rows)

print
blas_gemm(a, b)
print
blas_gemm(b, a)