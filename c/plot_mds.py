from __future__ import print_function

try:
    import matplotlib.pyplot as plt 
    import numpy as np
    from sklearn import datasets
    import sys
except: 
    raise ImportError("Missing libraries needed to run script")

# Define labels and constants
y = datasets.load_digits(n_class=6).target
ROWS = 1083
COLS = 2


# Main
def main():
    """Plots MDS solution of input data set"""
    if len(sys.argv) < 1:
        print("\nCommand line interface\n\tpython plot_mds.py <str: absolute file path of binary data set>")

    try:
        X_mds = np.fromfile(sys.argv[1]).reshape(ROWS, COLS)
        for label in set(y):
            idx = np.where(y == label)[0]
            plt.scatter(X_mds[idx, 0], X_mds[idx, 1], label="Label = %d" % label)
        plt.legend()
        plt.xlabel("Component 1"); plt.ylabel("Component 2")
        plt.title("MDS Visualization")
        plt.show()
    except Exception as e:
        print("Error loading data set because %s" % str(e))


if __name__ == "__main__":
    main()