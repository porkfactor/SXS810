#!/usr/bin/python3

import argparse
import numpy
from pathlib import Path
from astropy.table import Table

def fits_to_csv(input: Path, output: Path):
    Table.read(input).write(output, format='ascii.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, nargs=1, help='input file')
    parser.add_argument('output', type=str, nargs=1, help='output file')

    args = parser.parse_args()

    fits_to_csv(Path(args.input[0]), Path(args.output[0]))
