#!/usr/bin/env python3

import argparse
import csv
import random


parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type     = str,
    default  = 'matrix.csv',
    help     = "filename to write output to",
    required = True
    )
parser.add_argument(
    "--rows",
    type     = int,
    help     = "number of rows in input file",
    required = True
    )
parser.add_argument(
    "--cols",
    type     = int,
    help     = "number of columns in input file",
    required = True
    )
parser.add_argument(
    "--low",
    type    = int,
    default = -20,
    help    = "lowest possible value for generated data"
    )
parser.add_argument(
    "--high",
    type    = int,
    default = 20,
    help    = "highest possible value for generated data"
    )


def write_data_file(filename, rows, cols, lowest_value, highest_value):
    with open(filename, 'w') as fout:
        fout.write("{rows}\n{cols}\n".format(rows=rows,cols=cols))
        csv_writer = csv.writer(fout, delimiter=' ')
        for row in range(0, rows):
            csv_writer.writerow(
                [random.uniform(lowest_value, highest_value) for _ in range(cols)]
                )


def main():

    args = vars(parser.parse_args())

    filename = args['filename']
    rows     = args['rows']
    cols     = args['cols']
    low      = args['low']
    high     = args['high']

    write_data_file(filename, rows, cols, low, high)


if __name__ == '__main__':
    main()
