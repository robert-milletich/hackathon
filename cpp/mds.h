#ifndef MDS_H_
#define MDS_H_

#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

using namespace Eigen;

//void center_matrix(MatrixXd& M);

MatrixXd GetEigenProjectedMatrix(const MatrixXd&, int);

MatrixXd mds(const MatrixXd&, int);

#endif /* MDS_H_ */
