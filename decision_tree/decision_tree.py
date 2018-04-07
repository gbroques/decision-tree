from typing import List, Tuple

from .question import Question


class DecisionTree:

    @staticmethod
    def partition(rows: List, question: Question) -> Tuple[List, List]:
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
