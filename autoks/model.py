from autoks.kernel import kernel_to_infix_tokens, tokens_to_str
from evalg.encoding import infix_tokens_to_postfix_tokens, postfix_tokens_to_binexp_tree


def model_to_infix_tokens(model):
    return kernel_to_infix_tokens(model.kern)


def model_to_infix(model):
    infix_tokens = model_to_infix_tokens(model)
    return tokens_to_str(infix_tokens)


def model_to_binexptree(model):
    infix_tokens = model_to_infix_tokens(model)
    postfix_tokens = infix_tokens_to_postfix_tokens(infix_tokens)
    tree = postfix_tokens_to_binexp_tree(postfix_tokens)
    return tree
