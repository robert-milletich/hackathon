#include "utils.h"


/**
    Returns a (rows X cols) matrix containing the values obtained from
    filename

    @param filename - the file containing the values for the matrix
    @param rows - the number of rows for the matrix
    @param cols - the number of columns for the matrix
    @return - a (rows X cols) matrix populated with values from filename
*/
MatrixXd read_matrix(std::string filename, int rows, int cols) {
  std::ifstream fin(filename);

  if(!fin.good())
    throw std::runtime_error("Could not open input file '"+filename+"'!");

  fin>>rows>>cols;

  MatrixXd result(rows, cols);

  for(int y=0;y<rows;y++)
  for(int x=0;x<cols;x++)
    fin>>result(y,x);

  return result;
}


/**
    Writes the matrix M to filename

    @param M - the matrix to write
    @param filename - the file to write to
*/
void write_matrix(const MatrixXd& M, std::string filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << M;
    }
}
