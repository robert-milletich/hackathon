#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
#include <string>
#include <Eigen/Core>

#include "Timer.hpp"

using namespace Eigen;


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

void center_matrix_STABLE(MatrixXd& M) {
    Timer tmr;
    assert(M.rows() == M.cols());
    const int N = M.cols();
    std::vector<double> row_accum(N, 0.0);
    std::vector<double> col_accum(N, 0.0);

    double matrix_accum = 0;
    for (int j = 0; j < N; j++) {
        double temp = 0;
        for (int i = 0; i < N; i++) {
            row_accum[i] += M(i, j);
            temp += M(i, j);
            matrix_accum += M(i, j);
        }
        col_accum[j] = temp;
    }

    for (int i = 0; i < N; i++) {
        row_accum[i] /= N;
        col_accum[i] /= N;
    }

    matrix_accum /= N * N;

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            M(i, j) += (matrix_accum - (row_accum[i] + col_accum[j]));
        }
    }

    std::cerr << N << " center_matrix_STABLE run time = " << std::fixed << std::setprecision(10) << tmr.elapsed() << " s" << std::endl;
}


void center_matrix(MatrixXd& M) {
    Timer tmr;
    assert(M.rows() == M.cols());
    const int N = M.cols();
    //std::vector<double> row_accum(N, 0.0);
    //std::vector<double> col_accum(N, 0.0);

    // Define 
    double *mvec = M.data();
    double row[N];
    double col[N];
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

    std::cerr << N << " center_matrix run time = " << std::fixed << std::setprecision(10) << tmr.elapsed() << " s" << std::endl;
}



double MatDiff(const MatrixXd &a, const MatrixXd &b){
    return (a-b).sum();
}


template<class T>
MatrixXd CenterNewMatrix(T func, MatrixXd a){
    func(a);
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
    MatrixXd Test_Matrix(N, N);

    for (int i = 0; i < N * N; i++) {
        Test_Matrix(i) = rand() / (double) RAND_MAX;
    }

    std::vector<double> M(N*N);
    for(int i=0;i<N*N;i++)
        M[i] = Test_Matrix(i);

    // Define vector and run function
    std::vector<std::pair<std::string,MatrixXd> > ret;
    ret.emplace_back("center_matrix_STABLE", CenterNewMatrix(center_matrix_STABLE, Test_Matrix));
    ret.emplace_back("center_matrix", CenterNewMatrix(center_matrix, Test_Matrix));

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


