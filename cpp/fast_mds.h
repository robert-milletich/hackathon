#ifndef FAST_MDS_H_
#define FAST_MDS_H_

#include <algorithm>
#include <random>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/QR>

using namespace Eigen;

typedef std::pair<MatrixXd, MatrixXd> matrix_pair;

typedef std::vector<std::pair<MatrixXd, MatrixXd> > vector_of_matrix_pairs;

std::vector<int> get_permuted_range(int);

std::vector<int> sample_no_replacement(int, int);

MatrixXd get_sample_matrix(const MatrixXd&, int);

matrix_pair get_sample_matrices_from_pair(const matrix_pair&, int q);

std::vector<MatrixXd> partition_matrix(const MatrixXd&, int);

MatrixXd get_intercept_matrix(const MatrixXd&);

vector_of_matrix_pairs get_ai_xi_matrices(const MatrixXd&, int, int);

vector_of_matrix_pairs get_ai_xi_sample_matrices(const MatrixXd&, int, int, int);

std::vector<MatrixXd> get_xi_matrices(const MatrixXd&, int, int);

std::vector<MatrixXd> get_yi_sample_matrices(std::vector<MatrixXd>, int m);

std::vector<MatrixXd> get_bi_matrices(const MatrixXd&, int, int, int);

std::vector<MatrixXd> get_xi_mapped_matrices(const MatrixXd&, int, int, int);

MatrixXd fast_mds(const MatrixXd&, int, int, int);

#endif /* FAST_MDS_H_ */
