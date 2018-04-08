from typing import List, Tuple, Union

from .decision_node import DecisionNode
from .helpers import count_labels
from .leaf import Leaf
from .question import Question


class DecisionTree:

    def build_tree(self, rows: List) -> Union[Leaf, DecisionNode]:
        """Builds the tree.

        Args:
            rows: The rows of the dataset.

        Returns:
            Root decision node.
        """
        info_gain, question = self.find_best_split(rows)

        if info_gain == 0:
            return Leaf(rows)

        true_rows, false_rows = self.partition(rows, question)

        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)

        return DecisionNode(question, true_branch, false_branch)

    @classmethod
    def find_best_split(cls, rows: List) -> Tuple[float, Question]:
        """Find the best question to ask by iterating over every feature,
        and calculating the information gain.

        Args:
            rows: The rows of the dataset.

        Returns:
            The best information gain, and the question to split on.
        """
        best_info_gain = 0
        best_question = None
        current_uncertainty = cls.gini(rows)
        num_features = len(rows[0]) - 1

        for column_index in range(num_features):
            values = set([row[column_index] for row in rows])

            for value in values:
                question = Question(column_index, value)

                true_rows, false_rows = cls.partition(rows, question)

                # Skip this split if it doesn't divide the dataset
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                info_gain = cls.info_gain(true_rows, false_rows, current_uncertainty)

                if info_gain >= best_info_gain:
                    best_info_gain, best_question = info_gain, question

            return best_info_gain, best_question

    @staticmethod
    def partition(rows: List, question: Question) -> Tuple[List, List]:
        """Partition a dataset.

        Args:
            rows: The rows of the dataset.
            question: The question to partition the dataset on.

        Returns:
            A tuple containing two lists.

            One list containing the rows for which the question was True,
            and another list containing the rows for which the question was False.
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    @classmethod
    def info_gain(cls, left: List, right: List, current_uncertainty: float) -> float:
        """Calculate the information gain.

        Information gain determines the goodness of a split.

        Args:
            left: The children to left of the parent.
            right: The children to right of the parent.
            current_uncertainty: The impurity of the parent node.

        Returns:
            Information gain, the goodness of the split.
        """
        p = float(len(left)) / (len(left) + len(right))
        weighted_sum_of_children = p * cls.gini(left) + (1 - p) * cls.gini(right)
        return current_uncertainty - weighted_sum_of_children

    @staticmethod
    def gini(rows: List) -> float:
        """Calculate the Gini impurity for a list of rows.

        See:
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

        Args:
            rows: The rows of the dataset.

        Returns:
            Gini impurity measure.
        """
        label_counts = count_labels(rows)
        impurity = 1
        for count in label_counts.values():
            probability = count / float(len(rows))
            impurity -= probability ** 2
        return impurity
