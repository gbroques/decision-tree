from typing import List

from .util import is_numeric


class Question:
    """A Question is used to partition a dataset."""

    def __init__(self, column_index: int, value, feature_name: str = None):
        self.column_index = column_index
        self.value = value
        self.feature_name = feature_name
        if feature_name is None:
            self.feature_name = 'feature ' + str(column_index)

    def match(self, example: List):
        value = example[self.column_index]
        if is_numeric(value):
            return value >= self.value
        else:
            return value == self.value

    def __repr__(self):
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        return 'Is %s %s %s?' % (self.feature_name,
                                 condition,
                                 str(self.value))

    def __eq__(self, other):
        return (isinstance(other, Question) and
                self.value == other.value and
                self.column_index == other.column_index)
