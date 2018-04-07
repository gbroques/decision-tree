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
