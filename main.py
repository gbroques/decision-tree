from argparse import ArgumentParser
from typing import List

from decision_tree import DecisionTree


def main():
    data = read_data()
    design_matrix = [row[:-1] for row in data]
    target_values = [row[-1] for row in data]
    decision_tree = DecisionTree()
    decision_tree.fit(design_matrix, target_values)
    predictions = decision_tree.predict(design_matrix)
    decision_tree.print()


def read_data() -> List[List]:
    """Read in data from a text file."""
    filename = get_filename()
    with open(filename) as f:
        f.readline()  # Skip first line
        data = [line.split() for line in f]
    return data


def get_filename() -> str:
    """Get the filename to read from as the first command line argument."""
    parser = ArgumentParser(description='Build a decision tree from data stored in a text file.')
    parser.add_argument('filename',
                        metavar='filename',
                        type=str,
                        help='Data in the format described in specification.pdf')
    args = parser.parse_args()
    filename = vars(args)['filename']
    return filename


if __name__ == '__main__':
    main()
