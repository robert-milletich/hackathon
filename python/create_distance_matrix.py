import numpy as np
from scipy import spatial
import pycuda.gpuarray as gpuarray
from skcuda import linalg
import skcuda

print skcuda.__version__

rows = 1000
cols = 10
mat = np.random.randn(rows, cols)

d_mat = spatial.distance.cdist(mat, mat)
d_sq = d_mat * d_mat