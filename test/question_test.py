import unittest

from decision_tree.question import Question


class QuestionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_data = [
            ['Green', 3, 'Apple'],
            ['Yellow', 3, 'Apple'],
            ['Red', 1, 'Grape'],
            ['Red', 1, 'Grape'],
            ['Yellow', 3, 'Lemon']
        ]

    def test_question_constructor(self):
        question = Question(0, 'Green')
        self.assertEqual(question.column_index, 0)
        self.assertEqual(question.value, 'Green')

    def test_match(self):
        question = Question(0, 'Green')
        example = self.training_data[0]
        self.assertTrue(question.match(example))

    def test_does_not_match(self):
        question = Question(0, 'Red')
        example = self.training_data[0]
        self.assertFalse(question.match(example))

    def test_match_with_numeric_value(self):
        question = Question(1, 3)
        example = self.training_data[0]
        self.assertTrue(question.match(example))

    def test_does_not_match_with_numeric_value(self):
        question = Question(1, 4)
        example = self.training_data[0]
        self.assertFalse(question.match(example))

    def test_repr(self):
        expected_repr = 'Is color == Red?'
        question = Question(0, 'Red')
        self.assertEqual(expected_repr, repr(question))

    def test_repr_with_numeric_value(self):
        expected_repr = 'Is diameter >= 2?'
        question = Question(1, 2)
        self.assertEqual(expected_repr, repr(question))


if __name__ == '__main__':
    unittest.main()
