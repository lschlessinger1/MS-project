from typing import List, Union

from sympy import Symbol


def apply_add_mul_operator(operator: str, operand_1, operand_2):
    if operator == "*":
        return operand_1 * operand_2
    elif operator == "+":
        return operand_1 + operand_2


def postfix_tokens_to_symbol(postfix_token_symbols: List[Union[str, Symbol]]) -> Symbol:
    operand_stack = []
    for token in postfix_token_symbols:
        if isinstance(token, Symbol):
            # token is operand
            operand_stack.append(token)
        else:
            # token is operator
            operand_2 = operand_stack.pop()
            operand_1 = operand_stack.pop()
            result = apply_add_mul_operator(token, operand_1, operand_2)
            operand_stack.append(result)
    return operand_stack.pop()
