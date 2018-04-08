import unittest

from decision_tree.leaf import Leaf


class LeafTest(unittest.TestCase):

    def test_leaf_constructor(self):
        expected_predictions = {'Grape': 2}
        rows = [['Red', 1, 'Grape'], ['Red', 1, 'Grape']]
        leaf = Leaf(rows)
        self.assertEqual(expected_predictions, leaf.predictions)

    def test_equals(self):
        rows = [['Red', 1, 'Grape'], ['Red', 1, 'Grape']]
        self.assertEqual(Leaf(rows), Leaf(rows))


if __name__ == '__main__':
    unittest.main()
