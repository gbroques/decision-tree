import unittest

from decision_tree import DecisionTree
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

    def test_partition(self):
        expected_true_rows = [['Red', 1, 'Grape'], ['Red', 1, 'Grape']]
        expected_false_rows = [['Green', 3, 'Apple'], ['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon']]

        question = Question(0, 'Red')
        true_rows, false_rows = DecisionTree.partition(self.training_data, question)

        self.assertEqual(expected_true_rows, true_rows)
        self.assertEqual(expected_false_rows, false_rows)

    def test_gini_with_no_mixing(self):
        no_mixing = [['Apple'], ['Apple']]
        self.assertEqual(0.0, DecisionTree.gini(no_mixing))

    def test_gini_with_mixing(self):
        mixed = [['Apple'], ['Grape']]
        self.assertEqual(0.5, DecisionTree.gini(mixed))

    def test_gini_with_a_lof_of_mixing(self):
        mixed = [['Apple'], ['Grape'], ['Banana'], ['Lemon'], ['Orange']]
        self.assertAlmostEqual(0.8, DecisionTree.gini(mixed))

    def test_info_gain(self):
        expected_info_gain = 0.14
        question = Question(0, 'Green')
        true_rows, false_rows = DecisionTree.partition(self.training_data, question)
        current_uncertainty = DecisionTree.gini(self.training_data)
        info_gain = DecisionTree.info_gain(true_rows, false_rows, current_uncertainty)
        self.assertAlmostEqual(expected_info_gain, info_gain)

    def test_info_gain_with_better_split(self):
        expected_info_gain = 0.3733333
        question = Question(0, 'Red')
        true_rows, false_rows = DecisionTree.partition(self.training_data, question)
        current_uncertainty = DecisionTree.gini(self.training_data)
        info_gain = DecisionTree.info_gain(true_rows, false_rows, current_uncertainty)
        self.assertAlmostEqual(expected_info_gain, info_gain)

    def test_find_best_split(self):
        expected_best_question = Question(0, 'Red')
        best_info_gain, best_question = DecisionTree.find_best_split(self.training_data)
        self.assertEqual(expected_best_question, best_question)


if __name__ == '__main__':
    unittest.main()
