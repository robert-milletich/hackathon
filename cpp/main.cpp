#include <iostream>
#include <unistd.h>

#include "mds.h"
#include "fast_mds.h"
#include "utils.h"
#include "Timer.hpp"



int main(int argc, char** argv) {
    int op;
    int rows;
    int cols;
    char* in;
    char* out;

    const char* options = "r:c:i:o:";

    while ((op = getopt(argc, argv, options)) != EOF) {
        switch (op) {
        case 'r':
            rows = atoi(optarg);
            break;
        case 'c':
            cols = atoi(optarg);
            break;
        case 'i':
            in = optarg;
            break;
        case 'o':
            out = optarg;
            break;
        }
    }

    // if the required arguments have not been supplied, print the proper usage
    // and return 1
    if (argc < 5) {
        std::cout << "usage: ./main -r num_rows -c num_cols -i input_file -o output_file" << std::endl;
        return 1;
    }

    // m (the desired dimensionality) and q (the sample size to use from each
    // partition) are set to 3, 5, respectively
    int m = 3;
    int q = m + 2;
    // p (the partition size) is set to the max of q + 1 and (num_rows / 10)
    // rounded down
    int p = std::max(q + 1, (int)floor(rows / 10));

    std::string infile(in);
    std::string outfile(out);

    Timer tmr;

    // read in the matrix from infile
    MatrixXd M = read_matrix(infile, rows, cols);

    // perform FastMDS on the read-in matrix with p, q, m as above
    MatrixXd result = fast_mds(M, p, q, m);

    std::cout<<"Calculation time = "<<tmr.elapsed()<<std::endl;

    // write out the result to outfile
    write_matrix(result, outfile);

    return 0;
}
