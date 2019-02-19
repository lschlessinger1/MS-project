import numpy as np

from evalg.util import swap


class Mutator:

    def mutate(self, individual):
        raise NotImplementedError("Implement mutate in a child class.")


class BitFlipMutator(Mutator):

    def __init__(self, gene_mut_prob: float = None):
        self.gene_mut_prob = gene_mut_prob

    def mutate(self, individual: np.array):
        """Bit-flip mutation"""
        gene_size = individual.size
        if self.gene_mut_prob is None:
            self.gene_mut_prob = 1 / gene_size

        mutated = np.random.uniform(0, 1, size=gene_size) < self.gene_mut_prob
        indiv_mut = np.where(mutated, ~individual.astype(bool), individual)

        return indiv_mut

    def __repr__(self):
        return f'gene_mut_prob={self.gene_mut_prob!r})'


class InterchangeMutator(Mutator):

    def mutate(self, individual: np.array):
        """Interchange mutation"""
        gene_size = individual.size
        ind = np.random.randint(0, gene_size, size=2)
        # swap first and second genes
        indiv_mut = swap(individual, ind[0], ind[1])

        return indiv_mut


class ReverseMutator(Mutator):

    def mutate(self, individual: np.array):
        """Reverse mutation."""
        gene_size = individual.size
        ind = np.random.randint(0, gene_size)
        indiv_mut = swap(individual, ind, ind - 1)

        return indiv_mut


class GaussianMutator(Mutator):

    def mutate(self, individual: np.array):
        """Gaussian mutation."""
        raise NotImplementedError("Gaussian mutation not yet implemented")


class BoundaryMutator(Mutator):

    def mutate(self, individual: np.array):
        """Boundary mutation."""
        raise NotImplementedError("Boundary mutation not yet implemented")


class UniformMutator(Mutator):

    def mutate(self, individual: np.array):
        """Uniform mutation."""
        raise NotImplementedError("Uniform mutation not yet implemented")


class NonuniformMutator(Mutator):

    def mutate(self, individual: np.array):
        """Non-Uniform mutation."""
        raise NotImplementedError("Non-Uniform mutation not yet implemented")


class ShrinkMutator(Mutator):

    def mutate(self, individual: np.array):
        """Shrink mutation."""
        raise NotImplementedError("Shrink mutation not yet implemented")
