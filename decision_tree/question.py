from typing import List

from .util import is_numeric


class Question:
    """A Question is used to partition a dataset."""

    def __init__(self, column_index: int, value):
        self.column_index = column_index
        self.value = value

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
        # TODO: Figure out how to make header configurable
        header = ['color', 'diameter', 'label']
        return 'Is %s %s %s?' % (header[self.column_index],
                                 condition,
                                 str(self.value))

    def __eq__(self, other):
        return (isinstance(other, Question) and
                self.value == other.value and
                self.column_index == other.column_index)
