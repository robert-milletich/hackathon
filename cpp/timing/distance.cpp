#include <iostream>
#include <Eigen/Core>
#include <string>

#include "Timer.hpp"

using namespace Eigen;

MatrixXd distance1(const MatrixXd& M) {
    Timer tmr;
    // const int BLOCK_SIZE = 32;
    const int N = M.rows();
    MatrixXd result = MatrixXd(N, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // since distance matrices are symmetric and 0 on the diagonal,
            // redundant computations could be avoided by setting
            //  result(i, i) = 0, and result(i, j) = result(j, i) when i > j
            result(i, j) = (M.row(i) - M.row(j)).squaredNorm();
        }
    }
    std::cerr << N << " distance1 run time = " << tmr.elapsed() << " s" << std::endl;
    return result;
}


MatrixXd distance2(const MatrixXd& M) {
    Timer tmr;

    const int N = M.rows();
    MatrixXd result(N, N);

    for(int i = 0; i < N * N; i++) {
        result(i) = 0;
    }

    for(int row1 = 0;        row1 < N; row1++) {
        for(int row2 = row1 + 1; row2 < N; row2++){
            double temp_sum = 0;
            for(int i = 0; i < N; i++){
                const double temp = M(i, row1) - M(i, row2);
                temp_sum += temp * temp;
            }
            result(row1, row2) = temp_sum;
            result(row2, row1) = temp_sum;
        }
    }

    std::cerr << N << " distance2 run time = " << tmr.elapsed() << " s" << std::endl;
    return result;
}

void distance4(double *M, const int width, const int height) {
    Timer tmr;

    double result[height*height];
    for(int i=0;i<height*height;i++)
        result[i] = 0;

    for(int row1 = 0;        row1 < height; row1++) {
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

    std::cerr << height << " distance2 run time = " << tmr.elapsed() << " s" << std::endl;
    //return result;
}



MatrixXd distance3(const MatrixXd& M, const int BLOCK_SIZE) {
    Timer tmr;

    const int N = M.rows();
    MatrixXd result(N, N);

    for(int i = 0; i < N * N; i++) {
        result(i) = 0;
    }

    for(int row1 = 0;        row1 < N; row1 += BLOCK_SIZE) {
        for(int r1b  = 0; r1b < std::min(N, row1 + BLOCK_SIZE); r1b++) {
            for(int row2 = r1b + 1; row2 < N; row2++){
                double temp_sum = 0;
                for(int i = 0; i < N; i++){
                    const double temp = M(i, r1b) - M(i, row2);
                    temp_sum += temp * temp;
                }
                result(r1b, row2) = temp_sum;
                result(row2, r1b) = temp_sum;
            }
        }
    }

    std::cerr << N << " " << BLOCK_SIZE << " distance3 run time = " << tmr.elapsed() << " s" << std::endl;
    return result;
}


int main(int argc, char **argv){
    if(argc!=2){
        std::cout<<"Syntax: "<<argv[0]<<" <SIZE>"<<std::endl;
        return -1;
    }

    const int N = std::stoi(argv[1]);
    MatrixXd Test_Matrix(N, N);

    for (int i = 0; i < N * N; i++) {
        Test_Matrix(i) = rand() / (double) RAND_MAX;
    }

    double *M = new double[N*N];
    for(int i=0;i<N*N;i++)
        M[i] = Test_Matrix(i);

    // distance1(Test_Matrix);
    distance2(Test_Matrix);
    //distance3(Test_Matrix, i);
    distance4(M, N, N);

}
