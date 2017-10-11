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


/**
    Returns the centering matrix for the matrix M, which is given by
    -1/2 J * M * J
    (where J = I - 1/n * 1, I is the identity matrix, and 1 is the
    constant-valued matrix of 1s)

    @param M - the matrix to generate the centering matrix from
    @return - the centering matrix for M
*/
void center_matrix_eigen(MatrixXd &M) {
  Timer tmr;

  const int N          = M.rows();
  MatrixXd identity    = MatrixXd::Identity(N, N);
  MatrixXd one_over_ns = MatrixXd::Constant(N, N, 1.0 / N);
  MatrixXd J           = identity - one_over_ns;
  M                    = (-1.0 / 2.0) * J * M * J;

  std::cerr << N << " center_matrix_eigen run time = " << std::fixed << std::setprecision(10) << tmr.elapsed() << " s" << std::endl;
}



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
  double *mvec  = M.data();
  std::vector<double> rowvec(N);
  std::vector<double> colvec(N);
  double *row   = rowvec.data();
  double *col   = colvec.data();
  double matrix_accum = 0;

  // OpenACC kernels
  #pragma acc kernels copy(mvec[0:N*N]) create(row[0:N], col[0:N])
  {
    for(int x=0;x<N;x++) row[x] = 0;
    for(int y=0;y<N;y++) col[y] = 0;

    //Column sums. Because the matrix is symmetric, we can use these as row sums
    //as well.
    #pragma acc loop independent
    for (int y = 0; y < N; y++) {
      double colsum = 0;
      #pragma acc loop reduction(+:colsum)
      for (int x = 0; x < N; x++) {
        colsum += mvec[y*N + x];
      }
      col[y] = colsum;
    }

    // Sum of all elements
    //#pragma acc loop reduction(+:matrix_accum)
    //#pragma acc loop seq
    for (int i = 0; i < N*N; i++) {
      matrix_accum += mvec[i];
    }

    // Averages for rows and columns
    for(int x=0;x<N;x++) row[x] /= N;
    for(int y=0;y<N;y++) col[y] /= N;

    // Grand mean
    matrix_accum /= N * N;

    // Double center each value
    #pragma acc loop collapse(2) independent
    for(int y = 0; y < N; y++) {
      for (int x = 0; x < N; x++) {
        mvec[y*N + x] += (matrix_accum - (col[x] + col[y]));
      }
    }

    for(int i = 0; i < N * N; i++)
      mvec[i] *= -0.5;  
  }

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

    //Create a symmetric matrix
    for(int y=0;y<N;y++)
    for(int x=0;x<N;x++){
      test_matrix[y*N+x] = rand() / (double) RAND_MAX;
      test_matrix[x*N+y] = test_matrix[y*N+x];
    }

    //Zero the diagonal
    for(int d=0;d<N;d++)
      test_matrix[d*N+d] = 0;


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


