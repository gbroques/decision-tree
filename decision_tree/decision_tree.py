from typing import List, Tuple

from .helpers import count_labels
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

    @staticmethod
    def gini(rows: List) -> float:
        label_counts = count_labels(rows)
        impurity = 1
        for count in label_counts.values():
            probability = count / float(len(rows))
            impurity -= probability ** 2
        return impurity
