import unittest

from decision_tree.decision_node import DecisionNode
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


if __name__ == '__main__':
    unittest.main()
