import numpy as np

from evalg.util import swap


def mutate_bit_flip(individual, gene_mut_prob=None):
    """Bit-flip mutation
    """
    indiv_mut = individual.copy()
    L = individual.size

    if gene_mut_prob is None:
        gene_mut_prob = 1 / L

    mutated = np.random.uniform(0, 1, size=L) < gene_mut_prob
    indiv_mut = np.where(mutated, ~indiv_mut.astype(bool), indiv_mut)

    return indiv_mut


def mutate_interchange(individual):
    """Interchange mutation
    """
    indiv_mut = individual.copy()
    L = indiv_mut.size
    ind = np.random.randint(0, L, size=2)
    # swap first and second genes
    indiv_mut = swap(indiv_mut, ind[0], ind[1])

    return indiv_mut


def mutate_reverse(individual):
    """Reverse mutation
    """
    indiv_mut = individual.copy()
    L = indiv_mut.size
    ind = np.random.randint(0, L)
    indiv_mut = swap(indiv_mut, ind, ind - 1)

    return indiv_mut


def mutate_gaussian(individual):
    """Gaussian mutation
    """
    raise NotImplementedError("mutate_gaussian not yet implemented")


def mutate_boundary(individual):
    """Boundary mutation
    """
    raise NotImplementedError("mutate_boundary not yet implemented")


def mutate_uniform(individual):
    """Uniform mutation
    """
    raise NotImplementedError("mutate_uniform not yet implemented")


def mutate_non_uniform(individual):
    """Non-Uniform mutation
    """
    raise NotImplementedError("mutate_non_uniform not yet implemented")


def mutate_shrink(individual):
    """Shrink mutation
    """
    raise NotImplementedError("mutate_shrink not yet implemented")
