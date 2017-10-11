#ifndef DOCTEST_CONFIG_DISABLE
    #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif

#include "doctest.h"

#include <iostream>
#include <unistd.h>

#include "mds.h"
#include "fast_mds.h"
#include "utils.h"

#ifdef DOCTEST_CONFIG_DISABLE

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

    const int desired_dim    = 3;                //Desired dimensionality
    const int rows_to_sample = desired_dim + 2;  //Sample size to use from each partition
    const int partition_size = std::max(rows_to_sample + 1, rows / 100);

    std::string infile(in);
    std::string outfile(out);

    // read in the matrix from infile
    MatrixXd M = read_matrix(infile, rows, cols);

    // perform FastMDS on the read-in matrix with partition_size, q, m as above
    MatrixXd result = fast_mds(M, partition_size, rows_to_sample, desired_dim);

    // write out the result to outfile
    write_matrix(result, outfile);

    return 0;
}

#endif
