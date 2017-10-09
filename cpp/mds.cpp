#include "mds.h"

using namespace Eigen;


/**
    Returns the centering matrix for the matrix M, which is given by
    -1/2 J * M * J
    (where J = I - 1/n * 1, I is the identity matrix, and 1 is the
    constant-valued matrix of 1s)

    @param M - the matrix to generate the centering matrix from
    @return - the centering matrix for M
*/
MatrixXd get_centering_matrix(const MatrixXd& M) {
    assert(M.rows() == M.cols());
    int n                = M.rows();
    MatrixXd identity    = MatrixXd::Identity(n, n);
    MatrixXd one_over_ns = MatrixXd::Constant(n, n, 1.0 / n);
    MatrixXd J           = identity - one_over_ns;
    MatrixXd result      = (-1.0 / 2.0) * J * M * J;
    return result;
}


/**
    Returns the distance squared matrix for the matrix M, where the
    distance is the Euclidean distance

    @param M - the matrix to compute the distance squared matrix for
    @return - the distance squared matrix for M
*/
MatrixXd get_distance_squared_matrix(const MatrixXd& M) {
    int n = M.rows();
    MatrixXd result = MatrixXd(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // since distance matrices are symmetric and 0 on the diagonal,
            // redundant computations could be avoided by setting
            //  result(i, i) = 0, and result(i, j) = result(j, i) when i > j
            result(i, j) = (M.row(i) - M.row(j)).squaredNorm();
        }
    }
    return result;
}


/**
    Returns a multimap (i.e., keys need not be unique) with key, value pairs
    corresponding to the m largest eigenvalues of the matrix M and their
    respective eigenvectors

    @param M - the matrix to compute the eigenvalues/eigenvectors for
    @param m - the number of largest eigenvalues/eigenvectors to populate the
    multimap with
    @return - the multimap containing the eigenvalue, eigenvector key, value
    pairs for the m largest eigenvalues of the matrix M
*/
eigen_multimap get_eigen_map(const MatrixXd& M, int m)  {
    int n = M.rows();
    assert(m <= n);
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(n);
    eigen_solver.compute(M);
    VectorXd eigenvalues  = eigen_solver.eigenvalues();
    MatrixXd eigenvectors = eigen_solver.eigenvectors();

    eigen_multimap eigen_map;
    for (int i = 0; i < eigenvalues.size(); i++) {
        eigen_map.insert(std::make_pair(eigenvalues(i), eigenvectors.col(i)));
    }
    eigen_multimap result;

    for (eigen_multimap::iterator it = eigen_map.begin(); it != std::next(eigen_map.begin(), m); it++) {
        result.insert(std::make_pair((*it).first, (*it).second));
    }
    return result;
}


/**
    Return the X matrix for the matrix M and the integer m, which is given by
    E_m * Lambda_m_sqrt
    (where E_m is the n X m matrix of m eigenvectors corresponding to the m largest
    eigenvalues, and Lambda_m_srt - the m X m diagonal matrix with entries
    corresponding to square roots of the m largest eigenvalues)

    @param M - the matrix to generate the X matrix for
    @param m - the value of m for E_m and Lambda_m_sqrt
    @return - the matrix X = E_m * Lambda_m_sqrt
*/
MatrixXd get_x_matrix(const MatrixXd& M, int m) {
    eigen_multimap eigen_map = get_eigen_map(M, m);
    int n = M.rows();
    // E_m - the (n X m) matrix of m eigenvectors corresponding to the m largest eigenvalues
    MatrixXd E_m = MatrixXd(n , m);
    // Lambda_m_srt - the (m X m) diagonal matrix with entries corresponding to square roots
    // of the m largest eigenvalues
    MatrixXd Lambda_m_sqrt = MatrixXd::Constant(m, m, 0.0);

    int index = 0;
    for (eigen_multimap::iterator it = eigen_map.begin(); it != eigen_map.end(); it++) {
        E_m.col(index) = (*it).second;
        Lambda_m_sqrt(index, index) = sqrt((*it).first);
        index++;
    }

    return E_m * Lambda_m_sqrt;
}


/**
    Returns the result of performing MDS on the matrix M with the given desired
    dimensionality m

    @param M - the matrix M to perform MDS on
    @param m - the desired dimensionality of the result
    @return - the matrix obtained from performing MDS to project M to m
    dimensions
*/
MatrixXd mds(const MatrixXd& M, int m) {
    MatrixXd D = get_distance_squared_matrix(M);
    MatrixXd B = get_centering_matrix(D);
    MatrixXd X = get_x_matrix(B, m);
    return X;
}
