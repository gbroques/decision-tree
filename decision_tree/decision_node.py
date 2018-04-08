from .leaf import Leaf
from .question import Question


class DecisionNode:
    """A Decision Node asks a question.

    Holds a reference to the question and two child nodes.
    """

    def __init__(self,
                 question: Question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __eq__(self, other):
        return compare_trees(self, other)


def compare_trees(a: DecisionNode, b: DecisionNode) -> bool:
    if isinstance(a, DecisionNode) and isinstance(b, DecisionNode):
        return (a.question == b.question and
                compare_trees(a.true_branch, b.true_branch) and
                compare_trees(a.false_branch, b.false_branch))
    elif isinstance(a, Leaf) and isinstance(b, Leaf):
        return a == b
    else:
        return False


def print_tree(node: DecisionNode, spacing=''):
    if isinstance(node, Leaf):
        print(spacing + 'Predict', node.predictions)
        return

    print(spacing + str(node.question))

    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
