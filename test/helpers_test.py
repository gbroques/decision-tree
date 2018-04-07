import unittest

from decision_tree.helpers import get_unique_values


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


if __name__ == '__main__':
    unittest.main()
