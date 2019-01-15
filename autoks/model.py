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


def BIC(model):
    """
    Calculate the Bayesian Information Criterion (BIC) for a GPy `model` with maximum likelihood hyperparameters on a given dataset.
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    # model.log_likelihood() is the natural logarithm of the marginal likelihood of the Gaussian process.
    # len(model.X) is the number of data points.
    # model._size_transformed() is the number of optimisation parameters.
    # BIC = ln(n)k - 2ln(L^)
    import numpy as np
    n = len(model.X)
    k = model._size_transformed()
    l_hat = model.log_likelihood()
    return np.log(n) * k - 2 * np.log(l_hat)
