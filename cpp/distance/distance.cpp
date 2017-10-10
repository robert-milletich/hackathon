#include <iostream>
#include <Eigen/Core>
#include <string>
#include <iomanip>
#include <vector>
#include <utility>

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

MatrixXd distance_eigen_square(const MatrixXd& M, const int width, const int height) {
  Timer tmr;

  MatrixXd result = MatrixXd(height, height);

  for (int i = 0; i < width; i++)
  for (int j = 0; j < width; j++) {
    // since distance matrices are symmetric and 0 on the diagonal,
    // redundant computations could be avoided by setting
    //  result(i, i) = 0, and result(i, j) = result(j, i) when i > j
    result(i, j) = (M.row(i) - M.row(j)).squaredNorm();
  }

  PrintTimings("distance_eigen_square", width, height, tmr.elapsed());

  return result;
}



std::vector<double> simple_distance(const std::vector<double> &M, const int width, const int height) {
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

  PrintTimings("simple_distance", width, height, tmr.elapsed());

  return result;
}



std::vector<double> distance5(const std::vector<double> &M, const int width, const int height, const int BS) {
  Timer tmr;

  std::vector<double> result(height*height, 0);

  for(int row1 = 0;       row1 < height;                    row1+=BS )
  for(int r1b  = row1;    r1b  < std::min(height, row1+BS); r1b++    )
  for(int row2 = r1b + 1; row2 < height;                    row2++   ){
    double temp_sum = 0;
    for(int i = 0; i < width; i++){
        const double temp = M[r1b*width+i] - M[row2*width+i];
        temp_sum += temp * temp;
    }
    result[r1b*height+row2] = temp_sum;
    result[row2*height+r1b] = temp_sum;
  }

  PrintTimings("distance5", width, height, tmr.elapsed());

  return result;
}



std::vector<double> distance6(const std::vector<double> &M, const int width, const int height, const int BS) {
  Timer tmr;

  std::vector<double> result(height*height, 0);

  for(int row1 = 0;     row1 < height;                      row1 += BS )
  for(int r1b  = row1;  r1b  < std::min(height, row1 + BS); r1b++      )
  for(int row2 = r1b+1; row2 < height;                      row2 += BS )
  for(int r2b  = row2;  r2b  < std::min(height, row2 + BS); r2b++      ){
    double temp_sum = 0;
    for(int i = 0; i < width; i++){
        const double temp = M[r1b*width+i] - M[r2b*width+i];
        temp_sum += temp * temp;
    }
    result[r1b*height+r2b] = temp_sum;
    result[r2b*height+r1b] = temp_sum;
  }

  PrintTimings("distance6", width, height, tmr.elapsed());

  return result;
}



template<class T>
std::vector<double> EigenTest(T func, std::vector<double> a, const int width, const int height){
  MatrixXd mat(height,width);

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
    mat(y,x) = a[y*width+x];

  // std::cout<<"Pre test matrix"<<std::endl;
  // std::cout<<mat<<std::endl;

  mat = func(mat, width, height);

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
    a[y*width+x] = mat(y,x);

  PrintVector("Post Eigen Test", a, width, height);

  return a;
}



double MatDiff(const std::vector<double> &a, const std::vector<double> &b){
  double diff = 0;
  for(int i=0;i<a.size();i++)
    diff += std::abs(a[i] - b[i]);
  return diff;
}



int main(int argc, char **argv){
  if(argc!=3){
      std::cout<<"Syntax: "<<argv[0]<<" <WIDTH> <HEIGHT>"<<std::endl;
      return -1;
  }

  const int width  = std::stoi(argv[1]);
  const int height = std::stoi(argv[2]);

  std::vector<double> M(width*height);
  for(int i=0;i<width*height;i++)
      M[i] = rand() / (double) RAND_MAX;

  PrintVector("Original data", M, width, height);

  std::vector<std::pair<std::string,std::vector<double> > > ret;

  if(width==height)
    ret.emplace_back("distance_eigen_square", EigenTest(distance_eigen_square, M, width, height));
  // distance1(Test_Matrix);
  ret.emplace_back("simple_distance", simple_distance(M, width, height));
  //distance3(Test_Matrix, i);
  ret.emplace_back("distance5", distance5(M, width, height, 500));
  ret.emplace_back("distance6", distance6(M, width, height, 100));

  for(const auto &i: ret)
    PrintVector(i.first, i.second, width, height);

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
