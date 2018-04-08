import unittest

from decision_tree.decision_node import DecisionNode
from decision_tree.decision_node import compare_trees
from decision_tree.leaf import Leaf
from decision_tree.question import Question


class DecisionNodeTest(unittest.TestCase):

    def test_decision_node_constructor(self):
        question = Question(0, 'Red')
        true_branch = [['Red', 1, 'Grape'], ['Red', 1, 'Grape']]
        false_branch = [
            ['Green', 3, 'Apple'],
            ['Yellow', 3, 'Apple'],
            ['Yellow', 3, 'Lemon']
        ]

        decision_node = DecisionNode(question, true_branch, false_branch)

        self.assertEqual(question, decision_node.question)
        self.assertEqual(true_branch, decision_node.true_branch)
        self.assertEqual(false_branch, decision_node.false_branch)

    def test_compare_trees_equal(self):
        tree = self.build_tree()
        self.assertTrue(compare_trees(tree, tree))

    def test_compare_trees_not_equal(self):
        tree = self.build_tree()
        question = Question(0, 'Red')
        true_branch = Leaf([['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon']])
        not_equal_tree = DecisionNode(question, true_branch, None)
        self.assertFalse(compare_trees(tree, not_equal_tree))

    @staticmethod
    def build_tree() -> DecisionNode:
        question = Question(0, 'Red')
        true_branch = Leaf([['Red', 1, 'Grape'], ['Red', 1, 'Grape']])
        child_question = Question(0, 'Yellow')
        child_true_branch = Leaf([['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon']])
        child_false_branch = Leaf([['Green', 3, 'Apple']])
        false_branch = DecisionNode(child_question, child_true_branch, child_false_branch)
        return DecisionNode(question, true_branch, false_branch)


if __name__ == '__main__':
    unittest.main()
