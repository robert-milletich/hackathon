#include <cstdlib>
#include <iostream>
#include "mds.h"
#include "Timer.hpp"
#include <Eigen/Core>
#include <spectra/SymEigsSolver.h> 
#include <stdexcept>
#include <map>

using namespace Eigen;

typedef std::multimap<double, VectorXd, std::greater<double> > eigen_multimap;


MatrixXd ArrayToMatrix(const std::vector<double> &array, const int width, const int height) {
    MatrixXd matrix(height, width);
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            matrix(y, x) = array[y*width + x];
        }
    }
    return matrix;
}

std::vector<double> MatrixToArray(const MatrixXd& mat) {
    const auto height = mat.rows();
    const auto width = mat.cols();
    std::vector<double> flattened(height*width);
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            flattened[y*width + x] = mat(y, x);
        }
    }
    return flattened;
}


/**
    Returns the centering matrix for the matrix M, which is given by
    -1/2 J * M * J
    (where J = I - 1/n * 1, I is the identity matrix, and 1 is the
    constant-valued matrix of 1s)

    @param M - the matrix to generate the centering matrix from
    @return - the centering matrix for M
*/
// MatrixXd get_centering_matrix(const MatrixXd& M) {
//     Timer tmr;
//     assert(M.rows() == M.cols());
//     int n                = M.rows();
//     MatrixXd identity    = MatrixXd::Identity(n, n);
//     MatrixXd one_over_ns = MatrixXd::Constant(n, n, 1.0 / n);
//     MatrixXd J           = identity - one_over_ns;
//     MatrixXd result      = (-1.0 / 2.0) * J * M * J;
//
//     std::cerr << "centering run time = " << tmr.elapsed() << " s" << std::endl;
//     return result;
// 
void center_matrix(std::vector<double> &M, const int N) {
  Timer tmr;

  std::vector<double> row_accum(N, 0.0);
  std::vector<double> col_accum(N, 0.0);

  double matrix_accum = 0;
  for (int y = 0; y < N; y++) {
    double temp = 0;
    for (int x = 0; x < N; x++) {
      row_accum[x] += M[y*N+x];
      temp         += M[y*N+x];
      matrix_accum += M[y*N+x];
    }
    col_accum[y] = temp;
  }

  for(int x=0;x<N;x++) row_accum[x] /= N;
  for(int y=0;y<N;y++) col_accum[y] /= N;

  matrix_accum /= N * N;

  for (int y = 0; y < N; y++) 
  for (int x = 0; x < N; x++) 
    M[y*N+x] += (matrix_accum - (row_accum[x] + col_accum[y]));

  for(int i = 0; i < N * N; i++)
    M[i] *= -0.5;

    std::cerr << "center matrix run time = " << tmr.elapsed() << " s" << std::endl;
}




/**
    Returns the distance squared matrix for the matrix M, where the
    distance is the Euclidean distance

    @param M - the matrix to compute the distance squared matrix for
    @return - the distance squared matrix for M
*/
std::vector<double> get_distance_squared_matrix(const std::vector<double> &M, const int width, const int height) {
  Timer tmr;
  std::vector<double> result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    double temp_sum = 0;
    for(int x = 0; x < width; x++){
      const double temp = M[row1*width+x] - M[row2*width+x];
      temp_sum += temp * temp;
    }
    result[row1*height+row2] = temp_sum;
    result[row2*height+row1] = temp_sum;
  }
    std::cerr << "distance matrix run time = " << tmr.elapsed() << " s" << std::endl;

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

MatrixXd GetEigenProjectedMatrix(const MatrixXd& M, int m) {
    Timer tmr;
    int n = M.rows();

    // Construct matrix operation object using the wrapper class DenseGenMatProd
    Spectra::DenseSymMatProd<double> op(M);

    // Eigensolver
    Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    if(eigs.info() != Spectra::SUCCESSFUL)
        throw std::runtime_error("Sorry about your luck.");

    const auto evalues = eigs.eigenvalues();
    const auto E_m     = eigs.eigenvectors();

    // Lambda_m_srt - the (m X m) diagonal matrix with entries corresponding to square roots
    // of the m largest eigenvalues
    MatrixXd Lambda_m_sqrt = MatrixXd::Constant(m, m, 0.0);
    for (int i=0; i<m; i++){
        Lambda_m_sqrt(i, i) = sqrt(evalues(i));
    }

    std::cerr << "get x run time = " << tmr.elapsed() << " s" << std::endl;
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
// MatrixXd mds(const MatrixXd& M, int m) {
//     Timer tmr;
//     MatrixXd D = get_distance_squared_matrix(M);
//     MatrixXd B = get_centering_matrix(D);
//     MatrixXd X = get_x_matrix(B, m);
//
//     std::cerr << "mds run time = " << tmr.elapsed() << " s\n" << std::endl;
//     return X;
// }

MatrixXd mds(const MatrixXd& M, int m) {
    Timer tmr;
    auto mvec = MatrixToArray(M);
    auto D = get_distance_squared_matrix(mvec, M.rows(), M.cols());
    center_matrix(D, M.rows());
    auto Dmat = ArrayToMatrix(D, M.rows(), M.cols());
    MatrixXd X = GetEigenProjectedMatrix(Dmat, m);

    std::cerr << "mds run time = " << tmr.elapsed() << " s\n" << std::endl;
    return X;
}
