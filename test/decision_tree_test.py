import unittest
from collections import Counter

from decision_tree import DecisionTree
from decision_tree.decision_node import DecisionNode
from decision_tree.decision_node import classify
from decision_tree.leaf import Leaf
from decision_tree.question import Question


class DecisionTreeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.training_data = [
            ['Green', 3, 'Apple'],
            ['Yellow', 3, 'Apple'],
            ['Red', 1, 'Grape'],
            ['Red', 1, 'Grape'],
            ['Yellow', 3, 'Lemon']
        ]
        cls.decision_tree = DecisionTree()
        cls.design_matrix = [row[:-1] for row in cls.training_data]
        cls.target_values = [row[-1] for row in cls.training_data]
        cls.decision_tree.fit(cls.design_matrix, cls.target_values)

    def test_fit_and_predict(self):
        expected_predictions = ['Apple', 'Apple', 'Grape', 'Grape', 'Apple']
        predictions = self.decision_tree.predict(self.design_matrix)
        self.assertEqual(expected_predictions, predictions)

    def test_partition(self):
        expected_true_rows = [['Red', 1, 'Grape'], ['Red', 1, 'Grape']]
        expected_false_rows = [['Green', 3, 'Apple'], ['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon']]

        question = Question(0, 'Red')
        true_rows, false_rows = DecisionTree._partition(self.training_data, question)

        self.assertEqual(expected_true_rows, true_rows)
        self.assertEqual(expected_false_rows, false_rows)

    def test_gini_with_no_mixing(self):
        no_mixing = [['Apple'], ['Apple']]
        self.assertEqual(0.0, DecisionTree._gini(no_mixing))

    def test_gini_with_mixing(self):
        mixed = [['Apple'], ['Grape']]
        self.assertEqual(0.5, DecisionTree._gini(mixed))

    def test_gini_with_a_lof_of_mixing(self):
        mixed = [['Apple'], ['Grape'], ['Banana'], ['Lemon'], ['Orange']]
        self.assertAlmostEqual(0.8, DecisionTree._gini(mixed))

    def test_info_gain(self):
        expected_info_gain = 0.14
        question = Question(0, 'Green')
        true_rows, false_rows = DecisionTree._partition(self.training_data, question)
        current_uncertainty = DecisionTree._gini(self.training_data)
        info_gain = DecisionTree.info_gain(true_rows, false_rows, current_uncertainty)
        self.assertAlmostEqual(expected_info_gain, info_gain)

    def test_info_gain_with_better_split(self):
        expected_info_gain = 0.3733333
        question = Question(0, 'Red')
        true_rows, false_rows = DecisionTree._partition(self.training_data, question)
        current_uncertainty = DecisionTree._gini(self.training_data)
        info_gain = DecisionTree.info_gain(true_rows, false_rows, current_uncertainty)
        self.assertAlmostEqual(expected_info_gain, info_gain)

    def test_find_best_split(self):
        expected_best_question = Question(1, 3)
        best_info_gain, best_question = self.decision_tree._find_best_split(self.training_data)
        self.assertEqual(expected_best_question, best_question)

    def test_find_best_split_with_no_best_question(self):
        rows = [['Green', 3, 'Apple']]
        best_info_gain, best_question = self.decision_tree._find_best_split(rows)
        self.assertIsNone(best_question)

    def test_build_tree(self):
        expected_tree = self.build_expected_tree()
        decision_tree = DecisionTree()
        tree = decision_tree._build_tree(self.training_data)
        self.assertEqual(expected_tree, tree)

    @staticmethod
    def build_expected_tree() -> DecisionNode:
        question = Question(1, 3)
        false_branch = Leaf([['Red', 1, 'Grape'], ['Red', 1, 'Grape']])
        child_question = Question(0, 'Yellow')
        child_true_branch = Leaf([['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon']])
        child_false_branch = Leaf([['Green', 3, 'Apple']])
        true_branch = DecisionNode(child_question, child_true_branch, child_false_branch)

        return DecisionNode(question, true_branch, false_branch)

    def test_classify_with_false_branch(self):
        expected_prediction = Counter({'Apple': 1})
        decision_tree = DecisionTree()
        tree = decision_tree._build_tree(self.training_data)
        prediction = classify(self.training_data[0], tree)
        self.assertEqual(expected_prediction, prediction)

    def test_classify_with_true_branch(self):
        expected_prediction = Counter({'Grape': 2})
        decision_tree = DecisionTree()
        tree = decision_tree._build_tree(self.training_data)
        prediction = classify(self.training_data[2], tree)
        self.assertEqual(expected_prediction, prediction)


if __name__ == '__main__':
    unittest.main()
