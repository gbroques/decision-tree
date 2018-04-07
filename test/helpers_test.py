import unittest
from collections import Counter

from decision_tree.helpers import count_labels
from decision_tree.helpers import get_unique_values
from decision_tree.helpers import is_numeric


class HelpersTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.training_data = [
            ['Green', 3, 'Apple'],
            ['Yellow', 3, 'Apple'],
            ['Red', 1, 'Grape'],
            ['Red', 1, 'Grape'],
            ['Yellow', 3, 'Lemon']
        ]

    def test_unique_values(self):
        expected_unique_values = {'Green', 'Yellow', 'Red'}
        unique_values = get_unique_values(self.training_data, 0)
        self.assertEqual(expected_unique_values, unique_values)

    def test_count_labels(self):
        expected_label_counts = Counter({'Apple': 2, 'Grape': 2, 'Lemon': 1})
        label_counts = count_labels(self.training_data)
        self.assertEqual(expected_label_counts, label_counts)

    def test_is_numeric(self):
        self.assertTrue(is_numeric(0.5))
        self.assertTrue(is_numeric(6))

        self.assertFalse(is_numeric('hi'))
        self.assertFalse(is_numeric({'hello': 'world'}))
        self.assertFalse(is_numeric([1, 2, 3]))


if __name__ == '__main__':
    unittest.main()
