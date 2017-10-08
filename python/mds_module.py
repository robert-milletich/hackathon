"""Good for Python 2.7 and Python 3.3+"""

__all__ = ["MAX_MDS", "mds", "fast_mds_recursion", "fast_mds"]
__version__ = "1.4"
__author__ = "Paul Terwilliger"

import numpy as np
from scipy import spatial
from scipy import linalg as LA
import scipy.sparse.linalg as SP

MAX_MDS = 8500
MAX_MDS_2 = 1200
DIMENSIONS = 3

class MDSError(Exception):
    pass

def mds(input_matrix):
    """Take in a matrix and return classical mds in 3-dimensions

    Does NOT compute FastMDS"""

    assert len(input_matrix) > DIMENSIONS
    assert len(input_matrix) < MAX_MDS
    mat = np.array(input_matrix)

    # Squared proximity matrix D^(2)
    n_ = len(mat)
    d_mat = spatial.distance.cdist(mat, mat)
    d_sq = d_mat * d_mat
    j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
    b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)

    # Eigendecomposition
    eig_vals, eig_vecs = SP.eigsh(b_, k=DIMENSIONS)

    vecs = np.fliplr(eig_vecs)
    vals = eig_vals[::-1]
    x_ = np.dot(vecs, np.sqrt(np.diag(vals)))

    return np.real(x_)

prev = 0
def fast_mds_recursion(input_matrix):
    """Return FastMDS of the input matrix

    Need to add in analysis of RAM to compute whether the dataset is
    small enough for regular mds, or whether the subsets are too large for
    regular mds.  Need to rewrite to remove recursion.

    Benchmark for Paul's computer: 10 million rows 10 dimensions works
    in 382 seconds

    General naming conventions:
        ss - subset
        sp - sample
    """

    mat = np.array(input_matrix)
    assert len(mat) > DIMENSIONS
    assert len(set(len(row) for row in mat)) == 1

    # Calculate constants in order to partition the input matrix
    max_mds = MAX_MDS_2
    dims = len(mat[0])
    sp_size = dims + DIMENSIONS
    one_ss_size = int(np.ceil(np.sqrt(len(mat) * sp_size)))
    one_num_ss = int(np.ceil(float(len(mat)) / one_ss_size))
    if one_num_ss < DIMENSIONS:
        return mds(mat)
    ss_size = int(np.ceil(float(len(mat)) / (one_num_ss - 2)))
    num_ss = int(np.ceil(float(len(mat)) / ss_size))

    if max_mds < sp_size * 2:
        max_mds = sp_size * 2
    if max_mds > MAX_MDS:
        raise MDSError
    assert ss_size > sp_size
    assert ss_size > (sp_size * num_ss)

    # Partition the input matrix
    ss_mat = [mat[(i*ss_size) : int((i+1) * ss_size)] for i in range(num_ss)]
    to_flatten = [mtr[:sp_size] for mtr in ss_mat]
    d_align = np.array([item for sublist in to_flatten for item in sublist])

    # Decide whether to recursively implement FastMDS on each subset
    dMDS = []
    m_align = []
    if ss_size > max_mds:
        ss_size2 = int(np.ceil(np.sqrt(len(ss_mat[0]) * sp_size) + sp_size))
        if ss_size2 != prev:
            print("Subsets too large at " + \
                  "{}, subsetting recursively at {}".format(ss_size, ss_size2))
            prev = ss_size2
        if ss_size2 == ss_size:
            raise MDSError

        dMDS = [fast_mds_recursion(mtr) for mtr in ss_mat]
        m_align = fast_mds_recursion(d_align)
    else:
        dMDS = [mds(np.array(mtr)) for mtr in ss_mat]
        m_align = mds(d_align)

    # Compute dMDS and mMDS
    sub_dMDS = [mtr[:sp_size] for mtr in dMDS]
    sub_mMDS = [m_align[i*sp_size: (i+1)*sp_size] for i in range(num_ss)]

    assert len(sub_mMDS) == len(sub_dMDS)
    for r_i in range(len(sub_mMDS)):
        assert len(sub_dMDS[r_i]) == len(sub_mMDS[r_i])

    # Stitch together subsets to get the solution
    all_mds = []
    for m_i, mtr in enumerate(dMDS):
        sub_mMDS_i = sub_mMDS[m_i]
        sub_dMDS_i = np.c_[sub_dMDS[m_i], np.ones(len(sub_dMDS[m_i]))]
        lstsq = np.linalg.lstsq(sub_dMDS_i, sub_mMDS_i)[0]
        mtr_ones = np.c_[mtr, np.ones(len(mtr))]
        all_mds.append(np.dot(mtr_ones, lstsq))

    return np.array([item for sublist in all_mds for item in sublist])

def find_strain(matrix1, matrix2):
    """Return the strain between `matrix1` and `matrix2`"""

    mat1 = np.array(matrix1)
    mat2 = np.array(matrix2)
    assert len(mat1) == len(mat2)

    ss_size = int(np.floor(MAX_MDS * MAX_MDS * .1 / float(len(mat1))))
    num_ss = int(np.ceil(len(mat1) / float(ss_size)))
    ss_mat1 = [mat1[(i*ss_size) : int((i+1) * ss_size)] for i in range(num_ss)]
    ss_mat2 = [mat2[(i*ss_size) : int((i+1) * ss_size)] for i in range(num_ss)]

    strain_sq = 0
    for i in range(len(ss_mat1)):
        i1, i2 = ss_mat1[i], ss_mat2[i]
        d1 = spatial.distance.cdist(i1, mat1, metric='mahalanobis')
        d2 = spatial.distance.cdist(i2, mat2, metric='mahalanobis')
        d_diff = d1 - d2
        strain_sq += np.sum(d_diff * d_diff)

    return np.sqrt(strain_sq)