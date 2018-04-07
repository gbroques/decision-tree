from collections import Counter
from typing import List, Set


def get_unique_values(rows: List, column_index: int) -> Set:
    """Find the unique values for a column in a dataset.

    Args:
        rows: The rows of the dataset.
        column_index: Column index to get the unique values of.

    Returns:
        A set of the unique values for the given column index.
    """
    return set([row[column_index] for row in rows])


def count_labels(rows: List) -> Counter:
    """Counts the number of each label in the dataset.

    Assumes the label is the last element in the row.

    Args:
        rows: The rows of the dataset.

    Returns:
        A Counter mapping each label to it's count.
    """
    class_counts = Counter()
    for row in rows:
        label = row[-1]
        class_counts[label] += 1
    return class_counts
