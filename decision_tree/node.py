from abc import ABC

from .question import Question


class Node(ABC):
    """Abstract type for decision tree node.

    Primarily used for typing.
    """

    def __init__(self,
                 question: Question = None,
                 true_branch=None,
                 false_branch=None):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

