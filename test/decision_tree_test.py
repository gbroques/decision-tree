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


if __name__ == '__main__':
    unittest.main()
