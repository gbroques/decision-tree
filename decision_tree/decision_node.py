from .leaf import Leaf
from .node import Node
from .question import Question


class DecisionNode(Node):
    """A Decision Node asks a question.

    Holds a reference to the question and two child nodes.
    """

    def __init__(self,
                 question: Question,
                 true_branch,
                 false_branch):
        super(DecisionNode, self).__init__(question, true_branch, false_branch)

    def __eq__(self, other):
        return compare_trees(self, other)

    def __str__(self):
        return print_tree(self)


def compare_trees(a: Node, b: Node) -> bool:
    if isinstance(a, DecisionNode) and isinstance(b, DecisionNode):
        return (a.question == b.question and
                compare_trees(a.true_branch, b.true_branch) and
                compare_trees(a.false_branch, b.false_branch))
    elif isinstance(a, Leaf) and isinstance(b, Leaf):
        return a == b
    else:
        return False


def print_tree(node: Node, spacing='') -> str:
    return print_tree_helper(node, '', spacing)


def print_tree_helper(node: Node, string: str, spacing=''):
    if isinstance(node, Leaf):
        string += spacing + 'Predict ' + str(node.predictions) + '\n'
        return string

    assert isinstance(node, DecisionNode)

    string += spacing + str(node.question) + '\n'

    string += spacing + '--> True:\n'
    string = print_tree_helper(node.true_branch, string, spacing + '  ')

    string += spacing + '--> False:\n'
    string = print_tree_helper(node.false_branch, string, spacing + '  ')
    return string
