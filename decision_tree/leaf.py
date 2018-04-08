from typing import List

from .helpers import count_labels


class Leaf:
    """A Leaf node classifies data.

    Holds a dictionary of labels to counts.
    """

    def __init__(self, rows: List):
        self.predictions = count_labels(rows)

    def __eq__(self, other):
        return (isinstance(other, Leaf) and
                self.predictions == other.predictions)
