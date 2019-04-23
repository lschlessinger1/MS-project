from unittest import TestCase

from sympy import Symbol

from src.autoks.symbolic.util import apply_add_mul_operator, postfix_tokens_to_symbol


class TestSymbolicUtil(TestCase):

    def test_apply_add_mul_operator(self):
        operator = '+'
        operand_1 = 5
        operand_2 = 10

        expected = operand_1 + operand_2
        actual = apply_add_mul_operator(operator, operand_1, operand_2)
        self.assertEqual(expected, actual)

    def test_postfix_tokens_to_symbol(self):
        x = Symbol('x')
        y = Symbol('y')
        postfix_tokens = [x, y, '*']
        actual = postfix_tokens_to_symbol(postfix_tokens)
        expected = x * y
        self.assertEqual(expected, actual)
