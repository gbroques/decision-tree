from typing import List, Tuple, Union

from .decision_node import DecisionNode
from .decision_node import classify
from .leaf import Leaf
from .node import Node
from .question import Question
from .util import count_labels


class DecisionTree:

    def __init__(self, feature_names: List[str] = None):
        self._root = None
        self._feature_names = feature_names

    def fit(self, design_matrix: List[List], target_values: List) -> 'DecisionTree':
        """Fit model according to design matrix and target values.

        Args:
            design_matrix: Training examples with dimension m x n,
                           where m is the number of examples,
                           and n is the number of features.
            target_values: Target values with dimension m,
                           where m is the number of examples

        Returns: self
        """
        rows = [row + [target_values[i]] for i, row in enumerate(design_matrix)]
        self._root = self._build_tree(rows)
        return self

    def predict(self, test_set: List[List]) -> List:
        """Predict target values for test set.

        Args:
            test_set: Test set with dimension m xn,
                      where m is the number of examples,
                      and n is the number of features.
        Returns: Predicted target values for the test set with dimension m,
                 where m is the number of examples.
        """
        predictions = []
        for i in range(len(test_set)):
            prediction = classify(test_set[i], self._root)
            best_prediction = max(prediction, key=lambda key: prediction[key])
            predictions.append(best_prediction)

        return predictions

    def predict_example(self, test_example: List):
        """Predict target value for a single test example.

        Args:
            test_example: Test example with dimension n x 1,
                          where n is the number of features.
        Returns: Predicted target values for the test example with dimension m,
                 where m is the number of examples.
        """
        prediction = classify(test_example, self._root)
        best_prediction = max(prediction, key=lambda key: prediction[key])
        return best_prediction

    def print(self) -> None:
        """Print the decision tree."""
        print(self._root)

    def _build_tree(self, rows: List[List]) -> Node:
        """Builds the tree.

        Args:
            rows: The rows of the dataset.

        Returns:
            Root decision node.
        """
        info_gain, question = self._find_best_split(rows)

        if info_gain == 0:
            return Leaf(rows)

        true_rows, false_rows = self._partition(rows, question)

        true_branch = self._build_tree(true_rows)
        false_branch = self._build_tree(false_rows)

        return DecisionNode(question, true_branch, false_branch)

    def _find_best_split(self, rows: List[List]) -> Tuple[float, Question]:
        """Find the best question to ask by iterating over every feature,
        and calculating the information gain.

        Args:
            rows: The rows of the dataset.

        Returns:
            The best information gain, and the question to split on.
        """
        best_info_gain = 0
        best_question = None
        current_uncertainty = self._gini(rows)
        num_features = len(rows[0]) - 1

        for column_index in range(num_features):
            values = set([row[column_index] for row in rows])

            for value in values:
                feature_name = self._get_feature_name(column_index)
                question = Question(column_index, value, feature_name)

                true_rows, false_rows = self._partition(rows, question)

                # Skip this split if it doesn't divide the dataset
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                info_gain = self.info_gain(true_rows, false_rows, current_uncertainty)

                if info_gain >= best_info_gain:
                    best_info_gain, best_question = info_gain, question

        return best_info_gain, best_question

    @staticmethod
    def _partition(rows: List[List], question: Question) -> Tuple[List, List]:
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
    def info_gain(cls, left: List[List], right: List[List], current_uncertainty: float) -> float:
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
        weighted_sum_of_children = p * cls._gini(left) + (1 - p) * cls._gini(right)
        return current_uncertainty - weighted_sum_of_children

    @staticmethod
    def _gini(rows: List[List]) -> float:
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

    def _get_feature_name(self, column_index: int) -> Union[str, None]:
        if self._feature_names is None:
            return None
        else:
            return self._feature_names[column_index]
