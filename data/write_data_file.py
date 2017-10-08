import argparse
import csv
from random import randint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    default='matrix.csv',
    help="filename to write output to"
    )
parser.add_argument(
    "--rows",
    type=int,
    help="number of rows in input file"
    )
parser.add_argument(
    "--cols",
    type=int,
    help="number of columns in input file"
    )
parser.add_argument(
    "--low",
    type=int,
    default=-20,
    help="lowest possible value for generated data"
    )
parser.add_argument(
    "--high",
    type=int,
    default=20,
    help="highest possible value for generated data"
    )


def write_data_file(filename, rows, cols, lowest_value, highest_value):
    with open(filename, 'w') as file:
        csv_writer = csv.writer(file, delimiter=',')
        for row in range(0, rows):
            csv_writer.writerow(
                [randint(lowest_value, highest_value) for _ in range(cols)]
                )


def main():

    args = vars(parser.parse_args())

    filename = args['filename']
    rows = args['rows']
    cols = args['cols']
    low = args['low']
    high = args['high']

    write_data_file(filename, rows, cols, low, high)


if __name__ == '__main__':
    main()
