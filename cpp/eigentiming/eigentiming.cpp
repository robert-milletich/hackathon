#include <iostream>
#include <Eigen/Core>
#include <spectra/SymEigsSolver.h>  // Also includes <MatOp/DenseSymMatProd.h>
#include <Eigen/Eigenvalues>
#include <string>
#include <iomanip>
#include <vector>
#include <utility>
#include <map>

#include "Timer.hpp"


typedef std::multimap<double, Eigen::VectorXd, std::greater<double> > eigen_multimap;


using namespace Eigen;

void PrintTimings(const std::string id, const int width, const int height, const double time){
  std::cerr<<"id="<<std::setw(25)<<id<<" width="<<width<<" height="<<height<<" time="<<time<<std::endl;
}

eigen_multimap GetKLargestEigenvalues(const MatrixXd& M, int m)  {
  int n = M.rows();
  assert(m <= n);
  SelfAdjointEigenSolver<MatrixXd> eigen_solver(n);
  eigen_solver.compute(M);
  VectorXd eigenvalues  = eigen_solver.eigenvalues();
  MatrixXd eigenvectors = eigen_solver.eigenvectors();

  eigen_multimap eigen_map;
  for (int i = 0; i < eigenvalues.size(); i++) {
      eigen_map.emplace(eigenvalues(i), eigenvectors.col(i));
  }
  eigen_multimap result;

  for (eigen_multimap::iterator it = eigen_map.begin(); it != std::next(eigen_map.begin(), m); it++) {
      result.emplace((*it).first, (*it).second);
  }
  return result;
}



eigen_multimap GetEigenProjectedMatrix_spectra(const MatrixXd& M, int m) {
  int n = M.rows();

  // Construct matrix operation object using the wrapper class DenseGenMatProd
  Spectra::DenseSymMatProd<double> op(M);
  //std::cerr << "Data structure defined with success!" << std::endl;

  // Eigensolver
  Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 3, 6);

  // Initialize and compute
  eigs.init();
  int nconv = eigs.compute();

  // Retrieve results
  if(eigs.info() != Spectra::SUCCESSFUL)
      throw std::runtime_error("Sorry about your luck.");

  const auto evalues = eigs.eigenvalues();
  const auto E_m    = eigs.eigenvectors();

  eigen_multimap result;
  for(int i=0;i<evalues.size();i++)
    result.emplace(evalues(i), -E_m.col(i));

  return result;
}







template<class T>
eigen_multimap EigenTest(std::string id, T func, const MatrixXd& mat, int m){
  Timer tmr;

  auto ret = func(mat, m);

  PrintTimings(id, mat.cols(), mat.rows(), tmr.elapsed());

  return ret;
}



double MatDiff(const eigen_multimap& a, const eigen_multimap &b){
  auto aiter = a.begin();
  auto biter = b.begin();
  double sum = 0;
  for(int i=0;i<a.size();i++,aiter++,biter++){
    const auto eiga = aiter->first;
    const auto eigb = biter->first;
    sum += std::abs(eiga-eigb);
  }
  return sum;
}



int main(int argc, char **argv){
  if(argc!=2){
      std::cout<<"Syntax: "<<argv[0]<<" <SIZE>"<<std::endl;
      return -1;
  }

  const int TARGET_DIM = 3;

  const int N = std::stoi(argv[1]);

  MatrixXd mat(N,N);
  for(int y=0;y<N;y++)
  for(int x=0;x<N;x++){
    mat(x,y) = rand() / (double) RAND_MAX;
    mat(y,x) = mat(x,y);
  }

  // std::cout<<"Original data: "<<std::endl<<mat<<std::endl;

  std::vector<std::pair<std::string, eigen_multimap > > ret;

  ret.emplace_back("eigen_simple", EigenTest("eigen_simple", GetKLargestEigenvalues,mat,TARGET_DIM));
  ret.emplace_back("eigen_spectra", EigenTest("eigen_spectra", GetEigenProjectedMatrix_spectra,mat,TARGET_DIM));

  //Print results
  for(const auto &x: ret){
    continue;
    std::cout<<x.first<<std::endl;
    for(const auto &eig_vec: x.second)
      std::cout<<eig_vec.first<<std::endl;
    for(const auto &eig_vec: x.second){
      for(unsigned int i=0;i<eig_vec.second.size();i++)
        std::cout<<std::setw(8)<<std::fixed<<std::setprecision(4)<<eig_vec.second(i)<<" ";
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
  }

  for(int i=0;i<ret.size();i++)
  for(int j=i+1;j<ret.size();j++){
    std::cerr<<"Diff between" << " "
             <<std::setw(25)<<ret.at(i).first   << " "
             <<std::setw(25)<<ret.at(j).first   << " = "
             <<MatDiff(ret.at(i).second, ret.at(j).second)
             <<std::endl;
  }

  // for(int bs=1;bs<1000;bs+=5)
  //   distance6(M, width, height, bs);
}
