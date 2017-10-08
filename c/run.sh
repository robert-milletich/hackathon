rm data_mds.txt
make
./gsl_test

echo Plotting MDS solution using matplotlib...
python plot_mds.py data_mds.txt