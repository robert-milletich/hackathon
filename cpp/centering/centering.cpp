#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
#include <string>
#include <Eigen/Core>

#include "Timer.hpp"

using namespace Eigen;

void PrintVector(std::string id, const std::vector<double> &a, const int N){
  return;
  std::cout<<id<<std::endl;
  for(int y=0;y<N;y++){
    for(int x=0;x<N;x++)
      std::cout<<std::setw(6)<<std::setprecision(3)<<std::fixed<<a[y*N+x]<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}


void center_matrix_eigen(MatrixXd &M) {
  Timer tmr;

  const int N          = M.rows();
  MatrixXd identity    = MatrixXd::Identity(N, N);
  MatrixXd one_over_ns = MatrixXd::Constant(N, N, 1.0 / N);
  MatrixXd J           = identity - one_over_ns;
  M                    = (-1.0 / 2.0) * J * M * J;

  std::cerr << N << " center_matrix_eigen run time = " << std::fixed << std::setprecision(10) << tmr.elapsed() << " s" << std::endl;
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
// }

void center_matrix_STABLE(std::vector<double> &M, const int N) {
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

  std::cerr << N << " center_matrix_STABLE run time = " << std::fixed << std::setprecision(10) << tmr.elapsed() << " s" << std::endl;
}


void center_matrix(std::vector<double> &M, const int N) {
  Timer tmr;

  // Define 
  double *const mvec  = M.data();
  double *const row   = new double[N];
  double *const col   = new double[N];
  double matrix_accum = 0;

  // OpenACC kernels
  #pragma acc kernels copy(mvec[0:N*N])
  {
    for(int i=0;i<N;i++) row[i] = 0;
    for(int i=0;i<N;i++) col[i] = 0;

    // Row sums
    #pragma acc loop collapse(2) independent
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i++) {
        row[i] += mvec[i*N + j];
      }
    }

    // Column sums
    #pragma acc loop independent
    for (int j = 0; j < N; j++) {
      double temp = 0;
      for (int i = 0; i < N; i++) {
        temp += mvec[i*N + j];
      }
      col[j] = temp;
    }

    // Sum of all elements
    #pragma acc loop reduction(+:matrix_accum)
    for (int i = 0; i < N*N; i++) {
      matrix_accum += mvec[i];
    }

    // Averages for rows and columns
    for(int i=0;i<N;i++) row[i] /= N;
    for(int i=0;i<N;i++) col[i] /= N;

    // Grand mean
    matrix_accum /= N * N;

    // Double center each value
    #pragma acc loop collapse(2) independent
    for(int j = 0; j < N; j++) {
      for (int i = 0; i < N; i++) {
        mvec[i*N + j] += (matrix_accum - (row[i] + col[j]));
      }
    }
  }

  for(int i = 0; i < N * N; i++)
    M[i] *= -0.5;  

  delete row;
  delete col;

  std::cerr << N << " center_matrix run time = " << std::fixed << std::setprecision(10) << tmr.elapsed() << " s" << std::endl;
}



double MatDiff(const std::vector<double> &a, const std::vector<double> &b){
  double diff = 0;
  for(int i=0;i<a.size();i++)
    diff += std::abs(a[i] - b[i]);
  return diff;
}


template<class T>
std::vector<double> CenterNewMatrix(T func, std::vector<double> a, const int N){
  func(a, N);
  return a;
}

template<class T>
std::vector<double> EigenTest(T func, std::vector<double> a, const int N){
  MatrixXd mat(N,N);

  for(int y=0;y<N;y++)
  for(int x=0;x<N;x++)
    mat(y,x) = a[y*N+x];

  // std::cout<<"Pre test matrix"<<std::endl;
  // std::cout<<mat<<std::endl;

  func(mat);

  for(int y=0;y<N;y++)
  for(int x=0;x<N;x++)
    a[y*N+x] = mat(y,x);

  PrintVector("Post Eigen Test", a, N);


  return a;
}



int main(int argc, char **argv){

    // Cli
    if(argc!=2){
        std::cout<<"Syntax: "<<argv[0]<<" <SIZE>"<<std::endl;
        return -1;
    }

    // Test data
    const int N = std::stoi(argv[1]);

    std::vector<double> test_matrix(N*N);

    for (int i = 0; i < N * N; i++)
      test_matrix[i] = rand() / (double) RAND_MAX;


    PrintVector("Original data", test_matrix, N);

    // Define vector and run function
    std::vector<std::pair<std::string, std::vector<double> > > ret;
    if(N<500)
      ret.emplace_back("center_matrix_eigen",       EigenTest      (center_matrix_eigen,       test_matrix, N));
    ret.emplace_back("center_matrix_STABLE", CenterNewMatrix(center_matrix_STABLE, test_matrix, N));
    ret.emplace_back("center_matrix",        CenterNewMatrix(center_matrix,        test_matrix, N));

    for(const auto &i: ret)
      PrintVector(i.first, i.second, N);

    // Differences
    for(int i=0;i<ret.size();i++)
    for(int j=i+1;j<ret.size();j++){
        std::cerr<<"Diff between" << " "
                 <<ret.at(i).first   << " "
                 <<ret.at(j).first   << " = "
                 <<MatDiff(ret.at(i).second, ret.at(j).second)
                 <<std::endl;
    }
}


