from typing import List

from .question import Question


class DecisionNode:
    """A Decision Node asks a question.

    Hods a reference to the question and two child nodes.
    """

    def __init__(self,
                 question: Question,
                 true_branch: List,
                 false_branch: List):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
