from __future__ import division, print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np


def calc_error(a, b):
    """ADD"""
    diff = a-b
    diff = np.multiply(diff,diff)
    diff = np.sqrt(np.sum(diff, axis=1))
    return np.max(diff)


def cli():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Plot results of MDS/FastMDS algorithm")
    parser.add_argument("-c",
                    "--classic", 
                    help="MDS/FastMDS output file from classic version",
                    required=True,
                    type=str,
                    metavar="classic.csv")

    parser.add_argument("-o",
                    "--optimized", 
                    help="MDS/FastMDS output file from optimized version",
                    required=True,
                    type=str,
                    metavar="optimized.csv")

    return parser.parse_args()

def main():
    """Main"""
    args = cli()

    # I/O stuff
    try:
        classic = np.loadtxt(args.classic)
    except Exception as e:
        print("Error loading classic file %s because %s" % str(e))

    try:
        optimized = np.loadtxt(args.optimized)
    except Exception as e:
        print("Error loading optimized file %s because %s" % str(e))

    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13, 10))
    ax1.scatter(classic[:, 0], classic[:, 1])
    ax1.set_title('Classic Solution')
    
    ax2.scatter(optimized[:, 0], optimized[:, 1])
    ax2.set_title('Optimized Solution')

    # Calculate error and plot
    eps = calc_error(classic, optimized)
    plt.suptitle("Max Error: {0:.10}".format(eps))
    plt.show()

if __name__ == "__main__":
    main()