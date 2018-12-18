from evalg.selection import select_k_best


class BaseGrammar:

    def __init__(self, k):
        self.k = k

    def initialize(self, kernel_families, n_models, n_dims):
        """ Initialize models

        :param kernel_families:
        :param n_models:
        :param n_dims:
        :return:
        """
        raise NotImplementedError('initialize must implemented in a subclass')

    def expand(self, models, model_scores, kernel_families):
        """ Get next round of candidate models from current models

        :param models:
        :param model_scores:
        :param kernel_families:
        :return:
        """
        raise NotImplementedError('expand must be implemented in a subclass')

    def select(self, models, model_scores):
        """ Select next round of models (default is top k models by objective)

        :param models:
        :param model_scores:
        :return:
        """
        return select_k_best(models, model_scores, self.k)


def sort_kernel_tree(kernel_tree):
    """ Sort kernel tree into canonical form

    :param kernel_tree:
    :return:
    """
    pass


def remove_duplicates(kernel_trees):
    """ Remove duplicate kernel trees (after sorting)

    :param kernel_trees:
    :return:
    """
    pass
