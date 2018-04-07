from typing import List, Set


def get_unique_values(rows: List, column_index: int) -> Set:
    return set([row[column_index] for row in rows])
