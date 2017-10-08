#include <fstream>
#include <string>
#include <Eigen/Core>

using namespace Eigen;


MatrixXd read_matrix(std::string, int, int);

void write_matrix(const MatrixXd&, std::string);
