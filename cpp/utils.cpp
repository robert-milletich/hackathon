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
    std::ifstream in;
    in.open(filename);

    int i = 0;
    int j = 0;

    MatrixXd result = MatrixXd(rows, cols);

    std::string line;
    while (std::getline(in, line)) {
        // convert line to a string stream for use in std::getline
        std::stringstream line_stream(line);
        std::string num_string;

        while (std::getline(line_stream, num_string, ',')) {
            // convert num_string to a double and assign to result(i, j)
            // discard the delimiter ','
            result(i, j) = std::stod(num_string);
            j++;
            // keep j in the standard range: [0, cols - 1]
            j = j % cols;
        }
        i++;
    }

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
