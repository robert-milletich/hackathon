import numpy as np
import scipy.sparse.linalg as SP

mat = [[0, 1, 2, 3],
       [1, 0, 4, 5],
       [2, 4, 0, 6],
       [3, 5, 6, 0]]

mat = np.array(mat, dtype=np.float64)

print SP.eigsh(mat.T, k=3)