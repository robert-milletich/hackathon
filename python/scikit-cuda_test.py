import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from skcuda import cublas
from skcuda import linalg
from numba import cuda
from scipy import spatial
#
#x = np.random.rand(5).astype(np.float32)
#x_gpu = gpuarray.to_gpu(x)
#h = cublas.cublasCreate()
#m = cublas.cublasIsamax(h, x_gpu.size, x_gpu.gpudata, 1)
#cublas.cublasDestroy(h)
#np.allclose(m, np.argmax(abs(x.real) + abs(x.imag)))
#
#x = np.random.rand(5).astype(np.float32)
#x_gpu = cuda.to_device(x)
#h = cublas.cublasCreate()
#m = cublas.cublasIsamax(h, x_gpu.size, x_gpu.gpudata, 1)
#cublas.cublasDestroy(h)
#np.allclose(m, np.argmax(abs(x.real) + abs(x.imag)))

from skcuda import misc
misc.init_device()
misc.init()

rows = 10
cols = 10

n_ = rows
x_ = np.random.randn(rows, cols)
d_sq = spatial.distance.cdist(x_, x_) ** 2
j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)
b_gpu = gpuarray.to_gpu(b_)

print(linalg.eig(b_gpu, lib='cusolver'))