#include <iostream>
#include <Eigen/Core>
#include <string>
#include <iomanip>
#include <vector>
#include <utility>
#include "../mds.h"

#include "Timer.hpp"

using namespace Eigen;

void PrintVector(std::string id, const std::vector<double> &a, const int height, const int width){
  return;

  std::cout<<id<<std::endl;
  for(int y=0;y<height;y++){
    for(int x=0;x<width;x++)
      std::cout<<std::setw(6)<<std::setprecision(3)<<std::fixed<<a[y*width+x]<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}

void PrintTimings(const std::string id, const int width, const int height, const double time){
  std::cerr<<"id="<<std::setw(25)<<id<<" width="<<width<<" height="<<height<<" time="<<time<<std::endl;
}

MatrixXd GetEigenProjectedMatrix(const MatrixXd&, int);

template<class T>
MatrixXd EigenTest(std::string id, T func, const MatrixXd& mat, int m){
  Timer tmr;

  auto ret = func(mat, m);

  PrintTimings(id, mat.cols(), mat.rows(), tmr.elapsed());

  return ret;
}



double MatDiff(const MatrixXd& a, const MatrixXd& b){
  return (a-b).array().abs().sum();
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

  std::vector<std::pair<std::string,MatrixXd > > ret;

  ret.emplace_back("eigen_simple", EigenTest("eigen_simple", GetEigenProjectedMatrix,mat,TARGET_DIM));

  for(const auto &i: ret){
    continue;
    std::cout<<i.first<<std::endl<<i.second<<std::endl;
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
