from typing import List

from .util import count_labels
from .node import Node


class Leaf(Node):
    """A Leaf node classifies data.

    Holds a dictionary of labels to counts.
    """

    def __init__(self, rows: List[List]):
        super().__init__()
        self.predictions = count_labels(rows)

    def __eq__(self, other):
        return (isinstance(other, Leaf) and
                self.predictions == other.predictions)
