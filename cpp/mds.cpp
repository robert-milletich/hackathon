#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "mds.h"
#include "Timer.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <spectra/SymEigsSolver.h> 
#include <stdexcept>
#include <map>
#include "doctest.h"

using namespace Eigen;

typedef std::multimap<double, VectorXd, std::greater<double> > eigen_multimap;
typedef std::pair<VectorXd, MatrixXd> topeigs_t;

MatrixXd ArrayToMatrix(const std::vector<double> &array, const int width, const int height) {
  MatrixXd matrix(height, width);
  
  for(int y=0; y<height; y++)
  for(int x=0; x<width; x++)
    matrix(y, x) = array.at(y*width + x);

  return matrix;
}

std::vector<double> MatrixToArray(const MatrixXd& mat) {
  const auto height = mat.rows();
  const auto width = mat.cols();

  std::vector<double> flattened(height*width);
  for(int y=0; y<height; y++)
  for(int x=0; x<width; x++)
    flattened.at(y*width + x) = mat(y, x);

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
void center_matrix(std::vector<double> &M, const int N) {
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
}




/**
    Returns the distance squared matrix for the matrix M, where the
    distance is the Euclidean distance

    @param M - the matrix to compute the distance squared matrix for
    @return - the distance squared matrix for M
*/
std::vector<double> get_distance_squared_matrix(const std::vector<double> &M, const int width, const int height) {
  std::vector<double> result(height*height, 0);

  const int CS = 10; //TODO: Find optimal value

  for(int row1c = 0;     row1c<height;row1c+=CS)
  for(int row2c = row1c; row2c<height;row2c+=CS)

  for(int row1 = row1c; row1 < std::min(height,row1c+CS); row1++)
  for(int row2 = row2c; row2 < std::min(height,row2c+CS); row2++){
    double temp_sum = 0;
    for(int x = 0; x < width; x++){
      const double temp = M[row1*width+x] - M[row2*width+x];
      temp_sum += temp * temp;
    }
    result[row1*height+row2] = temp_sum;
    result[row2*height+row1] = temp_sum;
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
  assert(M.cols()==M.rows());
  int n = M.rows();
  assert(m <= n);
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver(n);
  eigen_solver.compute(M);
  VectorXd eigenvalues = eigen_solver.eigenvalues();
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



topeigs_t GetTopEigenValues(const MatrixXd &mat, const int num_eigvals){
  assert(mat.rows()==mat.cols());

  const int CONVERGENCE_PARAM = 6;

  Timer tmr;

  // Construct matrix operation object using the wrapper clagss DenseGenMatProd
  Spectra::DenseSymMatProd<double> op(mat);

  //Rule of thumb from Spectra docs
  assert(2*num_eigvals<=CONVERGENCE_PARAM);

  // Eigensolver
  Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, num_eigvals, CONVERGENCE_PARAM);

  // Initialize and compute
  eigs.init();
  eigs.compute();

  // Retrieve results
  if(eigs.info() != Spectra::SUCCESSFUL)
    throw std::runtime_error("Sorry about your luck.");

  const auto evalues = eigs.eigenvalues();
  const auto E_m     = eigs.eigenvectors();

  return std::make_pair(evalues, E_m);
}



/**
    Return the X matrix for the matrix M and the integer m, which is given by
    E_m * Lambda_m_sqrt
    (where E_m is the n X m matrix of m eigenvectors corresponding to the m largest
    eigenvalues, and Lambda_m_srt - the m X m diagonal matrix with entries
    corresponding to square roots of the m largest eigenvalues)

    @param m - the value of m for E_m and Lambda_m_sqrt
    @return - the matrix X = E_m * Lambda_m_sqrt
*/
MatrixXd GetEigenProjectedMatrix(const topeigs_t &topeigs) {
  const int num_eigvals = topeigs.first.size();

  // Lambda_m_srt - the (m X m) diagonal matrix with entries corresponding to square roots
  // of the m largest eigenvalues
  MatrixXd Lambda_m_sqrt = MatrixXd::Constant(num_eigvals, num_eigvals, 0.0);
  for (int i=0; i<num_eigvals; i++){
      Lambda_m_sqrt(i, i) = sqrt(topeigs.first(i));
  }

  return topeigs.second * Lambda_m_sqrt;
}



/**
    Returns the result of performing MDS on the matrix M with the given desired
    dimensionality m

    @param M - the matrix M to perform MDS on
    @param m - the desired dimensionality of the result
    @return - the matrix obtained from performing MDS to project M to m
    dimensions
*/
MatrixXd mds(const MatrixXd& M, int num_eigvals) {
  const auto mvec = MatrixToArray(M);
  auto D = get_distance_squared_matrix(mvec, M.cols(), M.rows());
  center_matrix(D, M.rows());

  const auto Dmat    = ArrayToMatrix(D, M.rows(), M.rows());
  const auto topeigs = GetTopEigenValues(Dmat, num_eigvals);
  const auto X       = GetEigenProjectedMatrix(topeigs);

  return X;
}



TEST_CASE("mvec display"){
  const int height = 5;
  const int width  = 10;

  Eigen::MatrixXd mat(height,width);
  for(int i=0;i<height*width;i++)
    mat(i) = rand()/(double)RAND_MAX;

  const auto vec     = MatrixToArray(mat);
  const auto matback = ArrayToMatrix(vec, width, height);

  auto same = mat==matback;

  CHECK(same);
}



TEST_CASE("Distance"){
  const int height = 100;
  const int width  = 5;

  Eigen::MatrixXd mat(height,width);
  for(int i=0;i<height*width;i++)
    mat(i) = rand()/(double)RAND_MAX;

  const auto vec     = MatrixToArray(mat);
  const auto distvec = get_distance_squared_matrix(vec,width,height);

  Eigen::MatrixXd result(height,height);
  for (int i = 0; i < mat.rows(); i++)
  for (int j = 0; j < mat.rows(); j++) {
    // since distance matrices are symmetric and 0 on the diagonal,
    // redundant computations could be avoided by setting
    //  result(i, i) = 0, and result(i, j) = result(j, i) when i > j
    result(i, j) = (mat.row(i) - mat.row(j)).squaredNorm();
  }

  const auto distmat = ArrayToMatrix(distvec, height, height);

  REQUIRE(result.rows()==distmat.rows());
  REQUIRE(result.cols()==distmat.cols());

  const auto diff = (distmat-result).array().abs().sum();

  CHECK(diff==doctest::Approx(0));
}



TEST_CASE("Centering"){
  const int N = 100;

  Eigen::MatrixXd mat(N, N);
  for(int i=0; i<N; i++){
    for(int j=i+1; j<N; j++){
      mat(i, j) = mat(j, i) = rand()/(double)RAND_MAX;
    }
  }

  // Centering using eigen
  Eigen::MatrixXd identity    = Eigen::MatrixXd::Identity(N, N);
  Eigen::MatrixXd one_over_ns = Eigen::MatrixXd::Constant(N, N, 1.0 / N);
  Eigen::MatrixXd J           = identity - one_over_ns;
  auto eigen_res       = (-1.0 / 2.0) * J * mat * J;

  // Centering using flattened array
  auto mat_vec = MatrixToArray(mat);
  center_matrix(mat_vec, N);
  auto our_mat = ArrayToMatrix(mat_vec, N, N);

  const auto diff = (eigen_res-our_mat).array().abs().sum();

  CHECK(diff==doctest::Approx(0));
}




TEST_CASE("Top k eigenvalues"){
  // Create symmetric test data
  const int N = 100;
  Eigen::MatrixXd mat(N, N);
  for(int i=0; i<N; i++){
    for(int j=i+1; j<N; j++){
      mat(i, j) = mat(j, i) = rand()/(double)RAND_MAX;
    }
  }
  const int n_eign = 3;

  //Eigen
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(N);
  eigen_solver.compute(mat);
  VectorXd eigenvalues  = eigen_solver.eigenvalues();
  Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors();
  typedef std::pair<double,VectorXd> eig_vec_pair;
  std::vector< eig_vec_pair > eig_vecs;
  for(int i=0;i<eigenvalues.size();i++)
    eig_vecs.emplace_back( eigenvalues(i), eigenvectors.col(i) );
  std::sort(eig_vecs.begin(), eig_vecs.end(), [](const eig_vec_pair &a, const eig_vec_pair &b){ return a.first>b.first; });

  //Spectra
  const auto eigenpairs_spectra = GetTopEigenValues(mat, n_eign);

  //Calculate difference
  double result = 0;
  for(int i=0; i<n_eign; i++){
    result += std::abs(eigenpairs_spectra.first(i) - eig_vecs[i].first);
  }

  CHECK(result==doctest::Approx(0));

}

