#include "mds.h"
#include "fast_mds.h"
#include "ProgressBar.hpp"

#include <iostream>


/**
    Return a std::vector containing a permutation of the integers in [0, n-1]

    @param n - the number n specifying the range [0, n-1]
    @return - a permutation of the integers in [0, n-1]
*/
std::vector<int> get_permuted_range(int n) {
    std::vector<int> indices;

    for (int i = 0; i < n; i++) {
        indices.push_back(i);
    }

    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::shuffle (indices.begin(), indices.end(), generator);

    return indices;
}


/**
    Return a std:vector containing the indices corresponding to sampling
    without replacement m times from the integers in [0, n-1]

    @param n - the number n specifying the range [0, n-1]
    @param m - the number of sample indices to generate
    @return - std:vector containing the indices corresponding to sampling
    without replacement m times from the integers in [0, n-1]
*/
std::vector<int> sample_no_replacement(int n, int m) {
    // return a vector containing the first m integers from a permutation
    // of the integers in [0, n-1]
    std::vector<int> indices = get_permuted_range(n);
    std::vector<int> result = std::vector<int>(indices.begin(), indices.begin() + m);
    return result;
}


/**
    Return a sample matrix containing q rows (without replacement) from the
    matrix M

    @param M - the matrix to sample from
    @param q - the number of rows to sample
    @return - the sample matrix containing q rows sampled without replacement
    from M
*/
MatrixXd get_sample_matrix(const MatrixXd& M, int q) {
    std::vector<int> indices = sample_no_replacement(M.rows(), q);

    MatrixXd M_sample = MatrixXd(q, M.cols());

    for (int i = 0; i < q; i++) {
        M_sample.row(i) = M.row(indices[i]);
    }

    return M_sample;
}


/**
    Return a pair of matrices containing q rows sampled without
    replacement from each matrix in the pair M_pair

    @param M_pair - the pair of matrices to sample from
    @param q - the number of rows to sample
    @return - the pair of matrices containing q rows sampled without
    replacement from each matrix in the pair M_pair
*/
matrix_pair get_sample_matrices_from_pair(const matrix_pair& M_pair, int q) {
    assert(M_pair.first.rows() == M_pair.second.rows());

    int rows = M_pair.first.rows();

    std::vector<int> indices = sample_no_replacement(rows, q);

    MatrixXd M_sample_1 = MatrixXd(q, M_pair.first.cols());
    MatrixXd M_sample_2 = MatrixXd(q, M_pair.second.cols());

    for (int i = 0; i < q; i++) {
        M_sample_1.row(i) = M_pair.first.row(indices[i]);
        M_sample_2.row(i) = M_pair.second.row(indices[i]);
    }

    matrix_pair result;
    result.first = M_sample_1;
    result.second = M_sample_2;

    return result;
}


/**
    Return a std::vector containing the submatrices obtained by partitioning
    the matrix M into partitions with p rows

    @param M - the matrix to partition
    @param p - the number of rows for the submatrices
    @return - a std::vector containing the submatrices obtained by partitioning
    the matrix M into partitions with p rows
*/
std::vector<MatrixXd> partition_matrix(const MatrixXd& M, int p) {
    // partition the input matrix into (p X num_cols) submatrices
    // note that the final submatrix may have more than p rows if num_rows % p != 0
    // in this case, it will have (p + num_rows % p) rows
    std::vector<MatrixXd> partition;
    int rows = M.rows();

    assert(p < rows);

    int cols = M.cols();
    int partition_size = rows / p;

    for (int i = 0; i < partition_size - 1; i ++) {
        partition.push_back(M.block(i * p, 0, p, cols));
    }

    partition.push_back(M.block((partition_size - 1) * p, 0, p + rows % p, cols));

    return partition;
}


/**
    Return the matrix obtained by appending a vector of 1s to the leftmost
    column of M

    @param M - the matrix to generate the intercept matrix for
    @result - the intercept matrix
*/
MatrixXd get_intercept_matrix(const MatrixXd& M) {
    int rows = M.rows();
    int cols = M.cols();

    MatrixXd M_intercept = MatrixXd(rows, cols + 1);

    M_intercept.block(0, 1, rows, cols) = M;

    M_intercept.col(0) = VectorXd::Constant(rows, 1);

    return M_intercept;
}


/**
    Return a std::vector containing pairs of matrices, where the pairs
    correspond to partitions ai of the matrix M and the associated xi = mds(ai)

    @param M - the matrix M to generate the vector of matrix pairs for
    @param p - the number of rows for the submatrices in the partition
    @param m - the desired dimensionality for the xi = mds(ai) matrices
    @result - a std::vector containing pairs of matrices, where the pairs
    correspond to partitions ai of the matrix M and the associated xi matrices
*/
vector_of_matrix_pairs get_ai_xi_matrices(const MatrixXd& M, int p, int m) {
    std::cerr<<"Starting get_ai_xi_matrices..."<<std::endl;
    std::vector<MatrixXd> partition = partition_matrix(M, p);

    vector_of_matrix_pairs result(partition.size());

    ProgressBar pg;
    pg.start(partition.size());

    std::cerr<<"Entering parallel region of get_ai_xi_matrices..."<<std::endl;
    #pragma omp parallel for
    for(unsigned int i=0;i<partition.size();i++){
        pg.update(i);
        const auto &thismat = partition.at(i);
        const auto ret = mds(thismat, m);
        result.at(i) = std::make_pair(thismat, ret);
    }

    std::cerr<<"get_ai_xi_matrices time = " << pg.stop() << " s"<<std::endl;

    return result;
}


/**
    Return a std::vector of matrix pairs of sample matrices of q rows without
    replacement from the ai, xi pairs with the specified partition size p and
    desired dimenionsality for mds m

    @param M - the input matrix
    @param p - the number of rows for the submatrices in the partition
    @param m - the desired dimensionality for mds
    @result - a std::vector containing pairs of matrices, where the pairs
    correspond to sample matrices of q rows without replacement from the ai, xi
    pairs
*/
vector_of_matrix_pairs get_ai_xi_sample_matrices(const MatrixXd& M, int p, int q, int m) {
    vector_of_matrix_pairs ai_xi_matrices = get_ai_xi_matrices(M, p, m);

    vector_of_matrix_pairs result;

    for (unsigned int i = 0; i < ai_xi_matrices.size(); i++) {
        result.push_back(get_sample_matrices_from_pair(ai_xi_matrices[i], q));
    }

    return result;
}


/**
    Return a std::vector containing the submatrices xi obtained by partitioning
    the matrix M into partitions ai of M with p rows and performing mds on each
    ai with desired dimensionality m

    @param M - the input matrix
    @param p - the number of rows for the submatrices in the partition
    @param m - the desired dimensionality for mds
    @result - a std::vector containing the submatrices xi
*/
std::vector<MatrixXd> get_xi_matrices(const MatrixXd& M, int p, int m) {
    std::vector<MatrixXd> partition = partition_matrix(M, p);

    std::vector<MatrixXd> result;

    for (std::vector<MatrixXd>::iterator it = partition.begin(); it != partition.end(); it++) {
        result.push_back(mds(*it, m));
    }

    return result;
}


/**
    Return a std::vector of the yi sample matrices obtained by performing mds
    on the matrix M and returning the submatrices from the partition of M
    (where M is the matrix of stacked ai sample matrices)

    @param vector_of_matrices - a vector containing the ai sample matrices
    @param m - the desired dimensionality for mds
    @return - a std::vector of the yi sample matrices obtained by performing mds
    on the matrix M and returning the submatrices from the partition of M
*/
std::vector<MatrixXd> get_yi_sample_matrices(std::vector<MatrixXd> vector_of_matrices, int m) {
    // there currently is no check that each matrix has the same number of rows and columns

    int rows = vector_of_matrices[0].rows();
    int cols = vector_of_matrices[0].cols();
    int num_submatrices = vector_of_matrices.size();

    // construct the M matrix (by stacking all of the ai_sample_matrices)
    // this is a (num_submatrices * sample_rows) X num_cols matrix
    MatrixXd stacked_matrices = MatrixXd(num_submatrices * rows, cols);

    for (int i = 0; i < num_submatrices; i++) {
        stacked_matrices.block(i * rows, 0, rows, cols) = vector_of_matrices[i];
    }

    MatrixXd Y =  mds(stacked_matrices, m);

    // return a vector of the yi matrices
    return partition_matrix(Y, rows);
}


/**
    Return a std::vector of the bi matrices, which are the
    solutions to xi_sample_intercept * bi = yi_sample_intercept
    (where xi_sample_intercept, yi_sample_intercept are the associated
    xi sample and yi sample matrices for the matrix M with a vector of 1s
    appended to the leftmost column)

    @param M - the input matrix
    @param p - the number of rows for the submatrices in the partition
    @param q - the number of rows to sample
    @param m - the desired dimensionality for mds
    @result - a std::vector of the bi matrices, which are the
    solutions to xi_sample_intercept * bi = yi_sample_intercept
*/
std::vector<MatrixXd> get_bi_matrices(const MatrixXd& M, int p, int q, int m) {
    vector_of_matrix_pairs ai_xi_sample_matrices = get_ai_xi_sample_matrices(M, p, q, m);

    std::vector<MatrixXd> ai_sample_matrices;
    std::vector<MatrixXd> xi_sample_matrices;
    //
    for (unsigned int i = 0; i < ai_xi_sample_matrices.size(); i++) {
        ai_sample_matrices.push_back(ai_xi_sample_matrices[i].first);
        xi_sample_matrices.push_back(ai_xi_sample_matrices[i].second);
    }

    std::vector<MatrixXd> yi_sample_matrices = get_yi_sample_matrices(ai_sample_matrices, m);

    std::vector<MatrixXd> bi_matrices;

    for (unsigned int i = 0; i < ai_xi_sample_matrices.size(); i++) {
        MatrixXd xi_sample_intercept = get_intercept_matrix(xi_sample_matrices[i]);
        MatrixXd yi_sample_intercept = get_intercept_matrix(yi_sample_matrices[i]);

        MatrixXd bi = xi_sample_intercept.colPivHouseholderQr().solve(yi_sample_intercept);

        bi_matrices.push_back(bi);
    }

    return bi_matrices;
}


/**
    Return a std::vector containing the xi mapped matrices, which are the
    intercept matrices for the matrices xi * bi, i.e., xi * bi with the vector
    of 1s appended to the leftmost column

    @param M - the input matrix
    @param p - the number of rows for the submatrices in the partition
    @param q - the number of rows to sample
    @param m - the desired dimensionality for mds
    @result - a std::vector containing the xi mapped matrices
*/
std::vector<MatrixXd> get_xi_mapped_matrices(const MatrixXd& M, int p, int q, int m) {

    std::cerr<<"get_xi_mapped_matrices -> get_xi_matrices"<<std::endl;
    std::vector<MatrixXd> xi_matrices = get_xi_matrices(M, p, m);
    std::cerr<<"get_xi_mapped_matrices -> get_bi_matrices"<<std::endl;
    std::vector<MatrixXd> bi_matrices = get_bi_matrices(M, p, q, m);

    // TODO determine if this check can be removed
    assert(bi_matrices.size() == xi_matrices.size());

    std::vector<MatrixXd> xi_mapped_matrices;

    std::cerr<<"get_intercept_matrix"<<std::endl;
    for (unsigned int i = 0; i < bi_matrices.size(); i++) {
        xi_mapped_matrices.push_back(get_intercept_matrix(xi_matrices[i]) * bi_matrices[i]);
    }

    return xi_mapped_matrices;
}


/**
    Return the matrix obtained by performing FastMDS on the matrix M with the
    parameters p, q, m

    @param M - the input matrix
    @param p - the number of rows for the submatrices in the partition
    @param q - the number of rows to sample
    @param m - the desired dimensionality for mds
    @result - the matrix obtained by performing FastMDS on the matrix M with the
    parameters p, q, m
*/
MatrixXd fast_mds(
  const MatrixXd& M, 
  const int rows_per_partition,
  const int rows_to_sample, 
  const int desired_dims
){
    Timer tmr;
    std::cout<<"FastMDS"<<std::endl;
    std::cout<<"rows_per_partition = "  <<rows_per_partition<<std::endl; 
    std::cout<<"rows_to_sample     = "  <<rows_to_sample    <<std::endl; 
    std::cout<<"desired_dims       = "  <<desired_dims      <<std::endl; 

    tmr.reset();
    std::cout<<"fast_mds passing control to get_xi_mapped_matrices..."<<std::endl;
    std::vector<MatrixXd> xi_mapped_matrices = get_xi_mapped_matrices(M, rows_per_partition, rows_to_sample, desired_dims);
    std::cout<<"Got xi_mapped_matrices in "<<tmr.elapsed()<<" s."<<std::endl;

    const int M_rows          = M.rows();
    const int cols            = desired_dims + 1;
    const int num_submatrices = xi_mapped_matrices.size();

    tmr.reset();
    MatrixXd result(M_rows, cols);
    for (int i = 0; i < num_submatrices - 1; i++) {
        result.block(i * rows_per_partition, 0, rows_per_partition, cols) = xi_mapped_matrices[i];
    }

    result.block(
        (num_submatrices - 1) * rows_per_partition, 
        0, 
        rows_per_partition + M_rows % rows_per_partition, 
        cols
    ) = xi_mapped_matrices[num_submatrices - 1];

    auto ret = result.block(0, 1, M_rows, cols - 1);

    std::cout<<"Stiched in "<<tmr.elapsed()<<" s."<<std::endl;

    return ret;
}
