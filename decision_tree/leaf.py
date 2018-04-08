from typing import List

from .node import Node
from .util import count_labels


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

    def __str__(self):
        total = sum(self.predictions.values()) * 1.0
        probabilities = {}
        for label in self.predictions.keys():
            total_counts = int(self.predictions[label])
            probabilities[label] = str(total_counts / total * 100) + '%'
        return str(probabilities)
