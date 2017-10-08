#ifndef MDS_H_
#define MDS_H_

#include <cassert>
#include <cmath>
#include <map>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace Eigen;

MatrixXd get_centering_matrix(const MatrixXd&);

MatrixXd get_distance_squared_matrix(const MatrixXd&);

typedef std::multimap<double, VectorXd, std::greater<double> > eigen_multimap;

eigen_multimap get_eigen_map(const MatrixXd&, int);

MatrixXd get_x_matrix(const MatrixXd&, int);

MatrixXd mds(const MatrixXd&, int);

#endif /* MDS_H_ */