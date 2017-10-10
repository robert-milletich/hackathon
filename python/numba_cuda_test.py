import numba
from numba import cuda
import numpy as np

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

griddim = 2
blockdim = 4

use_bar[griddim, blockdim](a, b, c)
#print foo.ptx