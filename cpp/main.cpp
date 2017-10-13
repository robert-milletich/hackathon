#ifndef DOCTEST_CONFIG_DISABLE
    #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif

#include "doctest.h"

#include <iostream>

#include "mds.h"
#include "fast_mds.h"
#include "utils.h"
#include "random.hpp"
#include "Timer.hpp"

#ifdef DOCTEST_CONFIG_DISABLE



/**
    Returns a (rows X cols) matrix containing the values obtained from
    filename

    @param filename - the file containing the values for the matrix
    @param rows - the number of rows for the matrix
    @param cols - the number of columns for the matrix
    @return - a (rows X cols) matrix populated with values from filename
*/
MatrixXd ReadMatrix(std::string filename){
  Timer tmr;

  std::cout<<"Reading data..."<<std::endl;

  std::ifstream fin(filename);

  if(!fin.good())
    throw std::runtime_error("Could not open input file '"+filename+"'!");

  int rows;
  int cols;
  fin>>rows>>cols;

  MatrixXd result(rows, cols);

  for(int y=0;y<rows;y++)
  for(int x=0;x<cols;x++)
    fin>>result(y,x);

  std::cout<<"Data read in "<<tmr.elapsed()<<" s"<<std::endl;

  return result;
}


/**
  Writes the matrix M to filename

  @param M - the matrix to write
  @param filename - the file to write to
*/
void write_matrix(const MatrixXd& M, std::string filename) {
  std::ofstream fout(filename);
  if(!fout.good())
    throw std::runtime_error("Could not open output file '"+filename+"'!");
  fout << M;
}



int main(int argc, char** argv) {
  if(argc!=3){
    std::cout<<"Syntax: "<<argv[0]<<" <Input File> <Output File>"<<std::endl;
    return -1;
  }

  // seed_rand(0);
  seed_rand(31);

  std::string infile (argv[1]);
  std::string outfile(argv[2]);

  // read in the matrix from infile
  MatrixXd M = ReadMatrix(infile);

  const int desired_dim    = 3;                //Desired dimensionality
  const int rows_to_sample = desired_dim + 2;  //Sample size to use from each partition
  const int partition_size = std::max(rows_to_sample + 1, ((int)M.rows()) / 100);

  // perform FastMDS on the read-in matrix with partition_size, q, m as above
  MatrixXd result = fast_mds(M, partition_size, rows_to_sample, desired_dim);

  // write out the result to outfile
  write_matrix(result, outfile);

  return 0;
}

#endif
