"""Good for Python 2.7 and Python 3.3+"""

__all__ = ["MAX_MDS", "mds", "fast_mds_recursion", "fast_mds"]
__version__ = "1.4"
__author__ = "Paul Terwilliger"

import numpy as np
from scipy import spatial
from scipy import linalg as LA
import scipy.sparse.linalg as SP
import time
import sys
import sklearn
from numba import cuda
import time

#sk_mds = sklearn.manifold.MDS
timer = 0

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

MAX_MDS = 8500
MAX_MDS_2 = 6000
DIMENSIONS = 3

class MDSError(Exception):
    pass

def mds(input_matrix):
    """Take in a matrix and return classical mds in 3-dimensions"""
    global timer

    assert len(input_matrix) > DIMENSIONS   
    assert len(input_matrix) < MAX_MDS
    mat = np.array(input_matrix)
    
    # Squared proximity matrix D^(2)
    n_ = len(mat)
#    d_mat = spatial.distance.cdist(mat, mat)
#    d_sq = d_mat * d_mat 
    d_sq = gpu_dist_matrix(mat)
    start = time.time()
    j_ = np.identity(n_) - (np.ones([n_, n_]) / float(n_))
    b_ = np.dot(np.dot((-1./2.) * j_, d_sq), j_)
    timer += time.time() - start
    
#     Attempt 1 of eig
    eig_vals, eig_vecs = SP.eigsh(b_, k=DIMENSIONS)
    
    vecs = np.fliplr(eig_vecs)
    vals = eig_vals[::-1]
    x_ = np.dot(vecs, np.sqrt(np.diag(vals)))
    
##     Attempt 2 of eig
#    e_vals, e_vecs = LA.eigh(b_, turbo=True, eigvals=(len(b_)-DIMENSIONS, len(b_)-1))
#    vecs = np.fliplr(e_vecs)
#    vals = e_vals[::-1]
#    x_ = np.dot(vecs, np.sqrt(np.diag(vals)))
    
#    # Attempt 3 of eig
#    eig_vals, eig_vecs = LA.eigh(b_, turbo=True) # The bottleneck at 50%
#    
#    eig = sorted(list(zip(eig_vals, np.transpose(eig_vecs))),\
#                                            reverse=True)[:DIMENSIONS]
#    vals_m = np.array([row[0] for row in eig])
#    vecs_m = np.transpose(np.array([row[1] for row in eig]))
#    x_ = np.dot(vecs_m, np.sqrt(np.diag(vals_m)))
    
    return np.real(x_)

prev = 0
def fast_mds_recursion(input_matrix):
    """Subsets mds using FastMDS and return mds in 3-dimensions
    
    Whole matrix is mat
    each subset is ss
    each sample is sp
    
    Need to add in computations for RAM and whether the dataset is small 
    enough for regular mds.  Need to rewrite to remove recursion
    
    10 million 10 dimensions works in 382 seconds
    """
    
    mat = np.array(input_matrix)
    assert len(mat) > DIMENSIONS
    assert len(set(len(row) for row in mat)) == 1
    
    # Partition matrix
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
    
    ss_mat = [mat[(i*ss_size) : int((i+1) * ss_size)] for i in range(num_ss)]
    to_flatten = [mtr[:sp_size] for mtr in ss_mat]
    d_align = np.array([item for sublist in to_flatten for item in sublist])
    
    
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
    
    sub_dMDS = [mtr[:sp_size] for mtr in dMDS]
    sub_mMDS = [m_align[i*sp_size: (i+1)*sp_size] for i in range(num_ss)]
    
    assert len(sub_mMDS) == len(sub_dMDS)
    for r_i in range(len(sub_mMDS)):
        assert len(sub_dMDS[r_i]) == len(sub_mMDS[r_i])
    
    all_mds = []
    for m_i, mtr in enumerate(dMDS):
        sub_mMDS_i = sub_mMDS[m_i]
        sub_dMDS_i = np.c_[sub_dMDS[m_i], np.ones(len(sub_dMDS[m_i]))]
        lstsq = np.linalg.lstsq(sub_dMDS_i, sub_mMDS_i)[0]
        mtr_ones = np.c_[mtr, np.ones(len(mtr))]
        all_mds.append(np.dot(mtr_ones, lstsq))
    
    return np.array([item for sublist in all_mds for item in sublist])

def partition_matrix(input_matrix):
    """Partition a matrix for mds"""

    assert len(set(len(row) for row in input_matrix)) == 1 # matrix-shaped
    assert len(input_matrix) > DIMENSIONS
    mat = np.array(input_matrix, dtype=np.float64)
    
    # Partition matrix
    max_mds = MAX_MDS_2
    dims = len(mat[0])
    sp_size = dims + DIMENSIONS
    one_ss_size = int(np.ceil(np.sqrt(len(mat) * sp_size)))
    one_num_ss = int(np.ceil(float(len(mat)) / one_ss_size))
    assert one_num_ss > DIMENSIONS
    ss_size = int(np.ceil(float(len(mat)) / (one_num_ss - 2)))
    num_ss = int(np.ceil(float(len(mat)) / ss_size))
    
    if max_mds < sp_size * 2:
        max_mds = sp_size * 2
    if max_mds > MAX_MDS:
        raise MDSError
        
    assert ss_size > sp_size
    assert ss_size > (sp_size * num_ss)
    
    ss_mat = [mat[(i*ss_size) : int((i+1) * ss_size)] for i in range(num_ss)]
    
    to_flatten = [mtr[:sp_size] for mtr in ss_mat]
    d_align = np.array([item for sublist in to_flatten for item in sublist])
    
    output = [["pre-mds", mtr] for mtr in ss_mat]
    output.append(["pre-mds", d_align])
    output.append(["ind", [sp_size, num_ss]])
    
    return output
    
def deep_list(ndarray, list_en):
    """Return the depth in the ndarray given in list_en"""
    output = ndarray
    for i in list_en:
        output = output[i]
    return output

def deep_partition(input_array):
    
    
    assert len(set(len(row) for row in input_array)) == 1 # matrix-shaped
    assert len(input_array) > DIMENSIONS
    mat = np.array(input_array, dtype=np.float64)

    max_mds = MAX_MDS_2
    dims = len(mat[0])
    sp_size = dims + DIMENSIONS
    if max_mds < sp_size * 2:
        max_mds = sp_size * 2
    if max_mds > MAX_MDS:
        raise MDSError
    
    super_arr = [["pre-sub", partition_matrix(mat)]]
    flag = True
    all_ss = [[], [], []]
    while flag:
        curr_loc = all_ss[-1]
        sub_arr = deep_list(super_arr, curr_loc)
        for i, nd_arr in enumerate(sub_arr):
            if nd_arr[0] == "pre-sub":
                nd_arr[0] = "sub"
                all_ss.append(curr_loc + [i, 1])
                break
            if nd_arr[0] == "pre-mds" and len(nd_arr[1]) > max_mds:
                assert len(nd_arr) == 2
                nd_arr[0] = "sub"
                nd_arr[1] = partition_matrix(nd_arr[1])
                all_ss.append(curr_loc + [i, 1])
                break
            if nd_arr[0] == "ind":
                all_ss = all_ss[:-1]
        if all_ss[-1] == []:
            flag = False
    
    return super_arr

def mds_each_part(super_arr):
    """Compute mds of each section created in fast_mds and recombine"""
    
    flag = True
    all_ss = [[], [], []]
    while flag:
        curr_loc = all_ss[-1]
        sub_arr = deep_list(super_arr, curr_loc)
        sup_arr = deep_list(super_arr, all_ss[:-1][-1])
        for i, nd_arr in enumerate(sub_arr):
            if nd_arr[0] == "sub":
                all_ss.append(all_ss[-1] + [i])
                all_ss.append(all_ss[-1] + [1])
                break
            elif nd_arr[0] == "pre-mds":
                sub_arr[i] = ["post-mds", mds(nd_arr[1])]
            elif nd_arr[0] == "ind":
                
                # Create recombined object named flat_mds
                assert sub_arr[-1][0] == "ind"
                assert list(set(row[0] for row in sub_arr[:-1]))[0] == \
                                                            "post-mds"
                assert len(set(row[0] for row in sub_arr[:-1])) == 1
                
                indicate = sub_arr[-1][1]
                sp_size, num_ss = indicate[0], indicate[1]
                dMDS = [row[1] for row in sub_arr[:-2]]
                m_align = sub_arr[-2][1]
                
                sub_dMDS = [mtr[:sp_size] for mtr in dMDS]
                sub_mMDS = [m_align[i*sp_size: (i+1)*sp_size] for i in \
                                                        range(num_ss)]
                
                assert len(sub_mMDS) == len(sub_dMDS)
                for r_i in range(len(sub_mMDS)):
                    assert len(sub_dMDS[r_i]) == len(sub_mMDS[r_i])
                    
                all_mds = []
                for m_i, mtr in enumerate(dMDS):
                    sub_mMDS_i = sub_mMDS[m_i]
                    sub_dMDS_i = np.c_[sub_dMDS[m_i], np.ones(len(
                                                        sub_dMDS[m_i]))]
                    lstsq = np.linalg.lstsq(sub_dMDS_i, sub_mMDS_i)[0]
                    mtr_ones = np.c_[mtr, np.ones(len(mtr))]
                    all_mds.append(np.dot(mtr_ones, lstsq))
                flat_mds = np.array([item for sublist in all_mds for item\
                                                            in sublist])
                
                # Replace list of matrices with recombined object
                sup_arr[0] = "post-mds"
                sup_arr[1] = flat_mds
                all_ss = all_ss[:-2]
                break
            
        if super_arr[0][0] == "post-mds":
            flag = False
    
    assert len(super_arr) == 1
    assert len(super_arr[0]) == 2
    assert super_arr[0][0] == "post-mds"
    assert len(set(len(row) for row in super_arr[0][1])) == 1 # matrix-shaped
    
    return super_arr[0][1]

def fast_mds(inp_matrix):
    """Subsets mds using FastMDS and return mds in 3-dimensions
    
    Need to add in computations for RAM and whether the dataset is small 
    enough for regular mds.  
    """
    
    deep = deep_partition(inp_matrix)
    output = mds_each_part(deep)
    return output

def find_strain(matrix1, matrix2):
    """Return the strain between mat1 and mat2"""
    
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

# For testing the function
rows = 30000
dims = 10
#assert rows * dims < MAX_MDS * MAX_MDS
foo = np.random.rand(rows, dims)
print("done creating")
start = time.time()
bar = fast_mds(foo)
print("Total time taken: {:.2f}".format(time.time() - start))
#print("done creating")
#start = time.time()
#bar_sk = sk_mds(n_components=3)
#print bar_sk.fit(foo)
#print("Total time taken: {:.2f}".format(time.time() - start))

#start = time.time()
#baz = fast_mds_recursion(foo)
#print("Total time taken: {:.2f}".format(time.time() - start))
#
#print (bar == baz).all()