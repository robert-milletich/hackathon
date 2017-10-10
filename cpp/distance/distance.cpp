#include <iostream>
#include <Eigen/Core>
#include <string>
#include <iomanip>
#include <vector>
#include <utility>

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


MatrixXd distance1(const MatrixXd& M) {
  Timer tmr;
  // const int BLOCK_SIZE = 32;
  const int N = M.rows();
  MatrixXd result = MatrixXd(N, N);

  for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++) {
    // since distance matrices are symmetric and 0 on the diagonal,
    // redundant computations could be avoided by setting
    //  result(i, i) = 0, and result(i, j) = result(j, i) when i > j
    result(i, j) = (M.row(i) - M.row(j)).squaredNorm();
  }

  std::cerr << N << " distance1 run time = " << tmr.elapsed() << " s" << std::endl;
  return result;
}


std::vector<double> distance2(const std::vector<double> &M, const int width, const int height) {
  Timer tmr;

  std::vector<double> result(height*height, 0);

  for(int row1 = 0; row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    double temp_sum = 0;
    for(int i = 0; i < width; i++){
      const double temp = M[row1*width+i] - M[row2*width+i];
      temp_sum += temp * temp;
    }
    result[row1*width+row2] = temp_sum;
    result[row2*width+row1] = temp_sum;
  }

  std::cerr << height << " distance2 run time = " << tmr.elapsed() << " s" << std::endl;
  return result;
}

std::vector<double> distance4(const std::vector<double> &M, const int width, const int height) {
  Timer tmr;

  std::vector<double> result(height*height, 0);

  for(int row1 = 0; row1 < height; row1++) {
    for(int row2 = row1 + 1; row2 < height; row2++){
      double temp_sum = 0;
      for(int i = 0; i < width; i++){
        const double temp = M[row1*width+i] - M[row2*width+i];
        temp_sum += temp * temp;
      }
      result[row1*height+row2] = temp_sum;
      result[row2*height+row1] = temp_sum;
    }
  }

  std::cerr << height << " distance4 run time = " << tmr.elapsed() << " s" << std::endl;

  return result;
}


std::vector<double> distance5(const std::vector<double> &M, const int width, const int height, const int BS) {
  Timer tmr;

  std::vector<double> result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1+=BS) {
      for(int r1b = row1; r1b < std::min(height, row1+BS); r1b++)
      for(int row2 = r1b + 1; row2 < height; row2++){
          double temp_sum = 0;
          for(int i = 0; i < width; i++){
              const double temp = M[r1b*width+i] - M[row2*width+i];
              temp_sum += temp * temp;
          }
          result[r1b*height+row2] = temp_sum;
          result[row2*height+r1b] = temp_sum;
      }
  }

  std::cerr << height << " distance5 run time = " << tmr.elapsed() << " s" << std::endl;

  return result;
}

std::vector<double> distance6(const std::vector<double> &M, const int width, const int height, const int BS) {
  Timer tmr;

  std::vector<double> result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1 += BS)
  for(int r1b  = 0; r1b < std::min(height, row1 + BS); r1b++)
  for(int row2 = row1; row2 < height; row2+= BS)
  for(int r2b  = 0; r2b < std::min(height, row2 + BS); r2b++){
    double temp_sum = 0;
    for(int i = 0; i < width; i++){
        const double temp = M[r1b*width+i] - M[r2b*width+i];
        temp_sum += temp * temp;
    }
    result[r1b*width+r2b] = temp_sum;
    result[r2b*width+r1b] = temp_sum;
  }

  std::cerr << height << " " << BS << " distance6 run time = " << tmr.elapsed() << " s" << std::endl;
  return result;
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

double MatDiff(const std::vector<double> &a, const std::vector<double> &b){
  double diff = 0;
  for(int i=0;i<a.size();i++)
    diff += std::abs(a[i] - b[i]);
  return diff;
}

int main(int argc, char **argv){
  if(argc!=2){
      std::cout<<"Syntax: "<<argv[0]<<" <SIZE>"<<std::endl;
      return -1;
  }

  const int N = std::stoi(argv[1]);

  std::vector<double> M(N*N);
  for(int i=0;i<N*N;i++)
      M[i] = rand() / (double) RAND_MAX;

  std::vector<std::pair<std::string,std::vector<double> > > ret;

  // distance1(Test_Matrix);
  ret.emplace_back("distance2", distance2(M, N, N));
  //distance3(Test_Matrix, i);
  ret.emplace_back("distance4", distance4(M, N, N));
  ret.emplace_back("distance5", distance5(M, N, N, 500));
  ret.emplace_back("distance6", distance6(M, N, N, 100));

  for(int i=0;i<ret.size();i++)
  for(int j=i+1;j<ret.size();j++){
    std::cerr<<"Diff between" << " "
             <<ret.at(i).first   << " "
             <<ret.at(j).first   << " = "
             <<MatDiff(ret.at(i).second, ret.at(j).second)
             <<std::endl;
  }
}
