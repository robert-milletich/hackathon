#include <iostream>
#include <Eigen/Core>
#include <string>
#include <iomanip>
#include <vector>
#include <utility>
#include <functional>

// #include <x86intrin.h>  // SSE4

// typedef union{
//   __m128d v;
//   double d[2];
// } v2df_t;

#include "Timer.hpp"
#define GET_VARIABLE_NAME(Variable) (#Variable)

typedef std::vector<double> dvec;

using namespace Eigen;

void PrintVector(std::string id, const dvec &a, const int width, const int height){
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

MatrixXd distance_eigen_square(const MatrixXd& M) {
  Timer tmr;

  MatrixXd result = MatrixXd(M.rows(), M.rows());

  for (int i = 0; i < M.rows(); i++)
  for (int j = 0; j < M.rows(); j++) {
    // since distance matrices are symmetric and 0 on the diagonal,
    // redundant computations could be avoided by setting
    //  result(i, i) = 0, and result(i, j) = result(j, i) when i > j
    result(i, j) = (M.row(i) - M.row(j)).squaredNorm();
  }

  PrintTimings("distance_eigen_square", M.rows(), M.rows(), tmr.elapsed());

  return result;
}



dvec simple_distance0(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

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

  return result;
}


dvec simple_distance_levi1(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

  const int CS = 10;

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



dvec simple_distance_levi2(const dvec &M, const int width, const int height, const int CS) {
  dvec result(height*height, 0);

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
    // result[row2*height+row1] = temp_sum;
  }

  return result;
}



double inline inner_distance1(const dvec &M, const int width, int row1, int row2){
  double temp_sum = 0;

  for(int x = 0; x < width; x++){
    const double temp = M[row1*width+x]-M[row2*width+x];
    temp_sum += temp * temp;
  }

  return temp_sum;
}

dvec simple_distance1(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    const double temp_sum = inner_distance1(M, width, row1, row2);
    result[row1*height+row2] = temp_sum;
    result[row2*height+row1] = temp_sum;
  }

  return result;
}










double inline inner_distance2(const dvec &M, const int width, int row1, int row2){
  double temp_sum = 0;

  row1 *= width;
  row2 *= width;

  for(int x = 0; x < width; x++){
    const double temp = M[row1+x]-M[row2+x];
    temp_sum += temp * temp;
  }

  return temp_sum;
}

dvec simple_distance2(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    const double temp_sum = inner_distance2(M, width, row1, row2);
    result[row1*height+row2] = temp_sum;
    result[row2*height+row1] = temp_sum;
  }

  return result;
}















double inline inner_distance3(const dvec &M, const int width, int row1, int row2){
  double temp_sum = 0;

  const double* const mdata = M.data();
  const double*       rdat1 = mdata+row1*width;
  const double*       rdat2 = mdata+row2*width;

  for(int x = 0; x < width; x++, rdat1++, rdat2++){
    const double temp = *rdat1 - *rdat2;
    temp_sum += temp * temp;
  }

  return temp_sum;
}

dvec simple_distance3(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    const double temp_sum = inner_distance3(M, width, row1, row2);
    result[row1*height+row2] = temp_sum;
    result[row2*height+row1] = temp_sum;
  }

  return result;
}



double inline inner_distance4(const dvec &M, const int width, int row1, int row2){
  double temp_sum = 0;

  const double* const mdata = M.data();
  const double*       rdat1 = mdata+row1*width;
  const double*       rdat2 = mdata+row2*width;

  const int STEP = 4;
  const int cleanwidth = STEP*(width/STEP);

  // std::cerr<<"cleanwidth="<<(cleanwidth)<<std::endl;

  for(int x = 0; x < cleanwidth; x+=STEP, rdat1+=STEP, rdat2+=STEP){
    const double a = *(rdat1+0) - *(rdat2+0);
    const double b = *(rdat1+1) - *(rdat2+1);
    const double c = *(rdat1+2) - *(rdat2+2);
    const double d = *(rdat1+3) - *(rdat2+3);
    temp_sum += a*a+b*b+c*c+d*d;
  }

  // std::cerr<<"Fin start="<<(STEP*cleanwidth)<<std::endl;
  for(int x=cleanwidth;x<width;x++,rdat1++,rdat2++){
    const double temp = *rdat1 - *rdat2;
    temp_sum += temp*temp;
  }

  return temp_sum;
}

dvec simple_distance4(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    const double temp_sum = inner_distance4(M, width, row1, row2);
    result[row1*height+row2] = temp_sum;
    result[row2*height+row1] = temp_sum;
  }

  return result;
}





double inline inner_distance5(const dvec &M, const int width, int row1, int row2){
  double temp_sum = 0;

  const double* const mdata = M.data();
  const double*       rdat1 = mdata+row1*width;
  const double*       rdat2 = mdata+row2*width;

  const int STEP = 8;
  const int cleanwidth = STEP*(width/STEP);

  // std::cerr<<"cleanwidth="<<(cleanwidth)<<std::endl;

  for(int x = 0; x < cleanwidth; x+=STEP, rdat1+=STEP, rdat2+=STEP){
    const double a = *(rdat1+0) - *(rdat2+0);
    const double b = *(rdat1+1) - *(rdat2+1);
    const double c = *(rdat1+2) - *(rdat2+2);
    const double d = *(rdat1+3) - *(rdat2+3);
    const double e = *(rdat1+4) - *(rdat2+4);
    const double f = *(rdat1+5) - *(rdat2+5);
    const double g = *(rdat1+6) - *(rdat2+6);
    const double h = *(rdat1+7) - *(rdat2+7);
    temp_sum += a*a+b*b+c*c+d*d+e*e+f*f+g*g+h*h;
  }

  // std::cerr<<"Fin start="<<(STEP*cleanwidth)<<std::endl;
  for(int x=cleanwidth;x<width;x++,rdat1++,rdat2++){
    const double temp = *rdat1 - *rdat2;
    temp_sum += temp*temp;
  }

  return temp_sum;
}

dvec simple_distance5(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

  for(int row1 = 0;        row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    const double temp_sum = inner_distance5(M, width, row1, row2);
    result[row1*height+row2] = temp_sum;
    result[row2*height+row1] = temp_sum;
  }

  return result;
}







//Attempt at using SIMD and FPUs

// double inline inner_distance6(const dvec &M, const int width, int row1, int row2){
//   double temp_sum = 0;

//   const double* const mdata = M.data();
//   const double*       rdat1 = mdata+row1*width;
//   const double*       rdat2 = mdata+row2*width;

//   const int STEP = 4;
//   const int cleanwidth = STEP*(width/STEP);

//   // std::cerr<<"cleanwidth="<<(cleanwidth)<<std::endl;

//   v2df_t regsum;
//   v2df_t reg1a;
//   v2df_t reg1b;
//   v2df_t reg2a;
//   v2df_t reg2b;

//   regsum.v = _mm_setzero_pd();

//   for(int x = 0; x < cleanwidth; x+=STEP, rdat1+=STEP, rdat2+=STEP){
//     reg1a.v = _mm_loadu_pd( rdat1   );
//     reg1b.v = _mm_loadu_pd( rdat1+1 );
//     reg2a.v = _mm_loadu_pd( rdat2   );
//     reg2b.v = _mm_loadu_pd( rdat2+1 );

//     reg1a.v  -= reg2a.v;
//     reg1a.v  *= reg1a.v;
//     regsum.v += reg1a.v; // _mm_add_pd(regsum.v, reg1a.v);

//     reg1b.v -= reg2b.v;
//     reg1b.v *= reg1b.v;
//     regsum.v += reg1b.v; // _mm_add_pd(regsum.v, reg1a.v);
//   }

//   temp_sum += regsum.d[0] + regsum.d[1];

//   // std::cerr<<"Fin start="<<(STEP*cleanwidth)<<std::endl;
//   for(int x=cleanwidth;x<width;x++,rdat1++,rdat2++){
//     const double temp = *rdat1 - *rdat2;
//     temp_sum += temp*temp;
//   }

//   return temp_sum;
// }

// dvec simple_distance6(const dvec &M, const int width, const int height) {
//   dvec result(height*height, 0);

//   for(int row1 = 0;        row1 < height; row1++)
//   for(int row2 = row1 + 1; row2 < height; row2++){
//     const double temp_sum = inner_distance6(M, width, row1, row2);
//     result[row1*height+row2] = temp_sum;
//     result[row2*height+row1] = temp_sum;
//   }

//   return result;
// }







dvec simple_distance7(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);

  const int RB_SIZE = 64;
  const int cleanwidth = RB_SIZE*(width/RB_SIZE);

  const double *mat = M.data();

  for(int rowblock=0;rowblock<width;rowblock+=RB_SIZE)
  for(int row1i = 0;          row1i < height; row1i++){
    const double *row1 = mat+row1i*width;
    for(int row2i = row1i + 1; row2i < height; row2i++){
      double temp_sum = 0;
      const double *row2 = mat+row2i*width;
      for(int x = rowblock; x < rowblock+RB_SIZE; x++){
        const double temp = *(row1+x) - *(row2+x);
        temp_sum += temp * temp;
      }
      for(int x=RB_SIZE;x<width;x++){
        const double temp = *(row1+x) - *(row2+x);
        temp_sum += temp*temp;
      }
      result[row1i*height+row2i] += temp_sum;
      result[row2i*height+row1i] += temp_sum;
    }
  }



  return result;
}





dvec simple_distance8(const dvec &M, const int width, const int height) {
  dvec result(height*height, 0);
  int i = 0;

  for(int row1 = 0;        row1 < height; row1++)
  for(int row2 = row1 + 1; row2 < height; row2++){
    double temp_sum = 0;
    for(int x = 0; x < width; x++){
      const double temp = M[row1*width+x] - M[row2*width+x];
      temp_sum += temp * temp;
    }
    result[i++] = temp_sum;
  }

  return result;
}







dvec distance_gpu(const dvec& M, const int width, const int height) {
    Timer tmr;

    dvec result(height * height, 0);

    const double *data = M.data();
    double *result_data = result.data();

    #pragma acc kernels default(none) copyin(data[0 : width * height]) copyout(result_data[0: height * height])
    #pragma acc loop independent collapse(2)
    for(int row1 = 0; row1 < height; row1++) {
        // for(int row2 = row1 + 1; row2 < height; row2++){
        for(int row2 = 0; row2 < height; row2++){
            double temp_sum = 0;
            #pragma acc loop reduction(+:temp_sum)
            for(int i = 0; i < width; i++){
                const double temp = data[row1 * width + i] - data[row2 * width + i];
                temp_sum += temp * temp;
            }
            result_data[row1 * height + row2] = temp_sum;
            // result_data[row2 * width + row1] = temp_sum;
        }
    }


    std::cerr << height << " distance gpu run time = " << tmr.elapsed() << " s" << std::endl;
    return result;
}


template<class T>
dvec EigenTest(T func, dvec a, const int width, const int height){
  MatrixXd mat(height,width);

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
    mat(y,x) = a[y*width+x];

  // std::cout<<"Pre test matrix"<<std::endl;
  // std::cout<<mat<<std::endl;

  mat = func(mat);

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
    a[y*width+x] = mat(y,x);

  PrintVector("Post Eigen Test", a, width, height);

  return a;
}



double MatDiff(const dvec &a, const dvec &b){
  double diff = 0;
  for(unsigned int i=0;i<a.size();i++)
    diff += std::abs(a[i] - b[i]);
  return diff;
}

template<class T>
dvec TimeDistance(std::string id, T func, const dvec &M, const int width, const int height){
  Timer tmr;

  const int REPEAT = 1;

  dvec ret;
  for(int i=0;i<REPEAT;i++){
    ret = func(M, width, height);
  }

  std::cout << "func="<<id<<" width="<<width<<" height="<<height<<" time="<<(tmr.elapsed()/REPEAT)<<" s"<<std::endl;
  return ret;
}

template<class T>
dvec TimeDistanceChunk(std::string id, T func, const dvec &M, const int width, const int height, const int CS){
  Timer tmr;

  const int REPEAT = 1;

  dvec ret;
  for(int i=0;i<REPEAT;i++){
    ret = func(M, width, height, CS);
  }

  std::cout << "func="<<id<<" CS="<<CS<<" width="<<width<<" height="<<height<<" time="<<(tmr.elapsed()/REPEAT)<<" s"<<std::endl;
  return ret;
}



int main(int argc, char **argv){
  if(argc!=3){
      std::cout<<"Syntax: "<<argv[0]<<" <WIDTH> <HEIGHT>"<<std::endl;
      return -1;
  }

  const int width  = std::stoi(argv[1]);
  const int height = std::stoi(argv[2]);

  dvec M(width*height);
  for(int i=0;i<width*height;i++)
    M[i] = rand() / (double) RAND_MAX;


  // TimeDistance("simple_distance0",simple_distance0,M,width,height); return -1;

  // TimeDistanceChunk("simple_distance_levi2",simple_distance_levi2,M,width,height,90); return -1;


  PrintVector("Original data", M, width, height);

  std::vector< std::pair<std::string, std::function< dvec(dvec,const int, const int) > > > funcs = {
    {GET_VARIABLE_NAME(simple_distance0), simple_distance0},
    //{GET_VARIABLE_NAME(simple_distance1), simple_distance1},
    //{GET_VARIABLE_NAME(simple_distance2), simple_distance2},
    //{GET_VARIABLE_NAME(simple_distance3), simple_distance3},
    //{GET_VARIABLE_NAME(simple_distance4), simple_distance4},
    //{GET_VARIABLE_NAME(simple_distance5), simple_distance5},
    //{GET_VARIABLE_NAME(simple_distance6), simple_distance6},
    //{GET_VARIABLE_NAME(simple_distance7), simple_distance7},
    //{GET_VARIABLE_NAME(simple_distance8), simple_distance8},
    {GET_VARIABLE_NAME(simple_distance_levi1), simple_distance_levi1}
    //{GET_VARIABLE_NAME(distance_gpu), distance_gpu}
  };

  std::vector<std::pair<std::string,dvec > > ret;

  //ret.emplace_back("distance_eigen_square", EigenTest(distance_eigen_square, M, width, height));
  for(const auto &func: funcs)
    ret.emplace_back(func.first, TimeDistance(func.first, func.second, M, width, height));


  std::cout<<std::endl;


  // for(int cs=1;cs<120;cs++)
  //   TimeDistanceChunk("simple_distance_levi2",simple_distance_levi2,M,width,height,cs);



  for(const auto &i: ret)
    PrintVector(i.first, i.second, height, height);

  for(unsigned int i=0;i<ret.size();i++)
  for(unsigned int j=i+1;j<ret.size();j++){
    std::cerr<<"Diff between" << " "
             <<std::setw(25)<<ret.at(i).first   << " "
             <<std::setw(25)<<ret.at(j).first   << " = "
             <<MatDiff(ret.at(i).second, ret.at(j).second)
             <<std::endl;
  }

  // for(int bs=1;bs<1000;bs+=5)
  //   distance6(M, width, height, bs);
}
